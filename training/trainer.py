from typing import *
from transformers import Trainer
from transformers.trainer_callback import CallbackHandler
from transformers.integrations import WandbCallback
import torch
from torch.nn import functional as F
from torch.utils.data import (
    DataLoader,
    RandomSampler,
)


class ZippedDataloader(DataLoader):
    """
    This exists to make zipped DL len-able, which is checked in training loop.
    """

    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders
        dl_lengths = [len(dl) for dl in self.dataloaders]
        self.dl_iters = [iter(dl) for dl in self.dataloaders]
        self._length = min(dl_lengths)

    @property
    def dataset(self):
        """
        Compatibility hack with Huggingface, return from only first DL
        """
        return self.dataloaders[0].dataset

    @property
    def sampler(self):
        """
        Compatibility hack with Huggingface, return from only first DL
        """
        return self.dataloaders[0].sampler

    def __len__(self):
        return self._length

    def __iter__(self):
        self.dl_iters = [iter(dl) for dl in self.dataloaders]
        return self

    def __next__(self):
        return [next(dl) for dl in self.dl_iters]


class WandbLogHandler(CallbackHandler):
    """
    This class avoids the annoying printouts from ProgressCallback
    """

    def __init__(self, cbh):
        """
        Pass in the original callback handler to init
        """
        # Copy all attributes from other handler
        self.__dict__.update(cbh.__dict__)
        # Then filter for only callbacks we want
        self.callbacks = [c for c in self.callbacks if isinstance(c, WandbCallback)]


class OODEvaluationTrainer(Trainer):
    """
    Handles OOD examples at validation time.
    """
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Wrap prediction_step function because OOD set doesn't have labels
        We mark them as -1 so it will error if we were to pass them in.
        """
        labels = inputs["labels"]
        del inputs["labels"]
        loss, logits, _ = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys
        )
        return loss, logits, labels

class AugmentedTrainer(OODEvaluationTrainer):
    """
    Subclassing Trainer to accept an additional dataset, an OOD set
    Doesn't actually do any augmented loss, override compute_loss to do that.
    """

    def __init__(self, ood_dataset, do_loss_logging=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.do_loss_logging = do_loss_logging
        self.ood_dataset = ood_dataset
        self.wandb_cbh = WandbLogHandler(self.callback_handler)

    def _get_ood_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """
        Returns sampler for OOD set, following get_train_sampler
        """
        generator = torch.Generator()
        generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        return RandomSampler(self.ood_dataset, generator=generator)

    def _get_ood_dataloader(self) -> DataLoader:
        """
        Returns dataloader for OOD set, following get_train_dataloader
        """
        ood_dataset = self.ood_dataset
        ood_dataset = self._remove_unused_columns(ood_dataset, description="ood")
        ood_sampler = self._get_ood_sampler()
        return DataLoader(
            ood_dataset,
            batch_size=self.args.train_batch_size,
            sampler=ood_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


    def get_train_dataloader(self) -> DataLoader:
        """
        Combine train and OOD dataloader into one.
        Kind of a hack but this avoids needing to modify the training loop.
        """
        train_dataloader = super().get_train_dataloader()
        ood_dataloader = self._get_ood_dataloader()
        return ZippedDataloader(train_dataloader, ood_dataloader)

    def _prepare_inputs(
        self, inputs: Union[list, Dict[str, Union[torch.Tensor, Any]]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Instead of preparing one batch of inputs, prepare a sequence of batches.
        Necessary because we pass in both ID and OOD batches as a tuple.
        """
        if isinstance(inputs, list):
            new_inputs = []
            for input_batch in inputs:
                input_batch = self._prepare_input(input_batch)
                if self.args.past_index >= 0 and self._past is not None:
                    input_batch["mems"] = self._past
                new_inputs.append(input_batch)
            return new_inputs
        inputs = self._prepare_input(inputs)
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past
        return inputs


class OETrainer(AugmentedTrainer):
    """
    Train with an OOD dataset and Outlier Exposure (OE)
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        OE Loss is computed as the cross-entropy between the uniform and predicted distribution
        """
        inputs, inputs_ood = inputs
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # Compute ID Loss
        if labels is not None:
            id_loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            id_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Compute OOD Loss
        outputs_ood = model(**inputs_ood)
        lprobs = F.log_softmax(
            outputs_ood.logits - torch.max(outputs_ood.logits, dim=1, keepdim=True)[0],
            dim=-1,
        )
        oe_loss = -1 * lprobs.mean()

        loss = id_loss + oe_loss

        # Log if necessary
        if self.do_loss_logging:
            self.control = self.wandb_cbh.on_log(
                self.args, self.state, self.control, {"id_loss": id_loss}
            )
            self.control = self.wandb_cbh.on_log(
                self.args, self.state, self.control, {"oe_loss": oe_loss}
            )
            self.control = self.wandb_cbh.on_log(
                self.args, self.state, self.control, {"total_loss": loss}
            )

        return (loss, outputs) if return_outputs else loss


class CCLTrainer(AugmentedTrainer):
    """
    Train with an OOD dataset and Confidence Contrastive Loss (CCL)
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        CCL loss is defined as the positive part of conf_ood - conf_id
        """
        inputs, inputs_ood = inputs
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # Compute ID Loss
        if labels is not None:
            id_loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            id_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # Compute OOD Loss
        outputs_ood = model(**inputs_ood)
        pairwise_conf_diffs = torch.max(outputs_ood.logits, dim=-1)[0].unsqueeze(
            1
        ) - torch.max(outputs.logits, dim=-1)[0].unsqueeze(0)
        pos_pairwise_conf_diffs = torch.nn.functional.relu(pairwise_conf_diffs)
        ccl_loss = pos_pairwise_conf_diffs.mean()

        loss = id_loss + ccl_loss

        # Log if necessary
        if self.do_loss_logging:
            self.control = self.wandb_cbh.on_log(
                self.args, self.state, self.control, {"id_loss": id_loss}
            )
            self.control = self.wandb_cbh.on_log(
                self.args, self.state, self.control, {"ccl_loss": ccl_loss}
            )
            self.control = self.wandb_cbh.on_log(
                self.args, self.state, self.control, {"total_loss": loss}
            )

        return (loss, outputs) if return_outputs else loss
