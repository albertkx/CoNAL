import os
import sys
import click
from datasets import DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    AutoConfig,
    DebertaV2Tokenizer,
)
import wandb
from metrics import sp_auac, ood_det_auroc, top1_acc, metric2hf
from training.trainer import CCLTrainer, OETrainer, OODEvaluationTrainer


def get_compute_metrics(ood_label):
    def compute_metrics(eval_pred):
        return {
            "sp_auac": metric2hf(sp_auac)(eval_pred),
            "accuracy": metric2hf(top1_acc, ood_label=ood_label)(eval_pred),
            "ood_det_auroc": metric2hf(ood_det_auroc, ood_label=ood_label)(eval_pred),
        }
    return compute_metrics

def get_model_hyperparams(model_type):
    """
    Return hyperparams of various models and the HF name
    All values from original papers
    """
    if model_type == "deberta":
        base_model = "microsoft/deberta-v3-base"
        batch_size = 20
        adam_eps = 1e-6
        adam_betas = (0.9, 0.999)
        lr = 2e-05
        grad_steps = 2
        weight_decay = 0.01
    elif model_type == "deberta_large":
        base_model = "microsoft/deberta-v3-large"
        batch_size = 10
        adam_eps = 1e-6
        adam_betas = (0.9, 0.999)
        lr = 2e-05
        grad_steps = 4
        weight_decay = 0.01
    elif model_type.startswith("roberta"):
        base_model = "roberta-base"
        batch_size = 40
        adam_eps = 1e-6
        adam_betas = (0.9, 0.98)
        lr = 2e-05
        grad_steps = 1
        weight_decay = 0.01
    elif model_type == "roberta_large":
        base_model = "roberta-large"
        batch_size = 40
        adam_eps = 1e-6
        adam_betas = (0.9, 0.98)
        lr = 1e-05
        grad_steps = 1
        weight_decay = 0.1
    elif model_type == "bert_large":
        base_model = "bert-large-cased"
        batch_size = 40
        adam_eps = 1e-8
        adam_betas = (0.9, 0.999)
        lr = 2e-05
        grad_steps = 1
        weight_decay = 0.01
    elif model_type.startswith("bert"):
        base_model = "bert-base-cased"
        lr = 2e-05
        adam_eps = 1e-8
        adam_betas = (0.9, 0.999)
        batch_size = 40
        grad_steps = 1
        weight_decay = 0.01
    else:
        raise ValueError("Model not found")
    return {
        "base_model": base_model,
        "lr": lr,
        "adam_eps": adam_eps,
        "adam_betas": adam_betas,
        "batch_size": batch_size,
        "grad_steps": grad_steps,
        "weight_decay": weight_decay,
    }


@click.command()
@click.argument("data_path")
@click.argument("model_path")
@click.option("--ood_training_type", type=str, default="vanilla")
@click.option("--ood_path", type=str, default=None)
@click.option("--limit_train_ood", type=int, default=None)
@click.option("--limit_train_id", type=int, default=None)
@click.option("--num_training_steps", type=int, default=10000)
@click.option("--checkpoint_steps", type=int, default=5000)
@click.option("--label_smoothing/--no_label_smoothing", type=bool, default=False)
@click.option("--do_early_stopping/--no_early_stopping", type=bool, default=False)
@click.option("--model_type", type=str, default="bert")
@click.option("--seed", type=int, default=123)
@click.option("--retrain_if_exists/--skip_if_exists", default=False)
@click.option("--do_loss_logging/--no_loss_logging", default=False)
def main(
    data_path,
    model_path,
    ood_training_type,
    ood_path,
    limit_train_ood,
    limit_train_id,
    num_training_steps,
    checkpoint_steps,
    label_smoothing,
    do_early_stopping,
    model_type,
    seed,
    retrain_if_exists,
    do_loss_logging,
):
    # Init
    if not retrain_if_exists and os.path.exists(
        os.path.join(model_path, "pytorch_model.bin")
    ):
        print(f"Model at {model_path} exists, skipping training!")
        sys.exit(0)
    os.makedirs(model_path, exist_ok=True)
    wandb.init(
        project="cnl",
        name=model_path,
        settings=wandb.Settings(start_method="fork"),
    )

    # Load model name & hyperparams
    hyperparams = get_model_hyperparams(model_type)
    base_model = hyperparams["base_model"]

    # Tokenizer Setup
    if model_type.startswith("deberta"):
        tokenizer = DebertaV2Tokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [f"[unused{k}]" for k in range(1000)]}
    )

    def tokenize_function(batch):
        if "text" in batch.keys():
            for k, v in tokenizer(
                batch["text"], truncation=True, padding=True, max_length=128
            ).items():
                batch[k] = v
        elif "sentence1" in batch.keys() and "sentence2" in batch.keys():
            for k, v in tokenizer(
                batch["sentence1"],
                batch["sentence2"],
                padding=True,
                truncation=True,
                max_length=128,
            ).items():
                batch[k] = v
        return batch

    # Tokenize ID & OOD datasets
    dataset = DatasetDict.load_from_disk(data_path)
    if limit_train_id is None:
        limit_train_id = float("inf")
    dataset = dataset.shuffle(seed=seed)
    dataset["train"] = dataset["train"].select(range(min(limit_train_id, len(dataset["train"]))))
    dataset = dataset.map(tokenize_function, batched=False)

    if ood_path is not None:
        ood_dataset = Dataset.load_from_disk(ood_path)
        if "label" in ood_dataset.features:
            ood_dataset = ood_dataset.remove_columns(["label"])
        if limit_train_ood is None:
            limit_train_ood = float("inf")
        ood_dataset = ood_dataset.shuffle(seed=seed)
        ood_dataset = ood_dataset.select(range(min(limit_train_ood, len(ood_dataset))))
        ood_dataset = ood_dataset.map(tokenize_function, batched=False)

    # Build training args first
    lr = hyperparams["lr"]
    adam_eps = hyperparams["adam_eps"]
    adam_betas = hyperparams["adam_betas"]
    batch_size = hyperparams["batch_size"]
    grad_steps = hyperparams["grad_steps"]
    weight_decay = hyperparams["weight_decay"]
    training_args = TrainingArguments(
        output_dir=model_path,
        max_steps=num_training_steps,
        warmup_steps=500,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_steps,
        learning_rate=lr,
        adam_epsilon=adam_eps,
        adam_beta1=adam_betas[0],
        adam_beta2=adam_betas[1],
        weight_decay=weight_decay,
        label_smoothing_factor=0.1 * label_smoothing,  # 0.1 else 0
        max_grad_norm=float("inf"),  # Empirically works better
        save_strategy="steps",
        save_total_limit=1,
        evaluation_strategy="steps",
        save_steps=checkpoint_steps,
        eval_steps=checkpoint_steps,
        load_best_model_at_end=do_early_stopping,
        metric_for_best_model="accuracy",
        report_to="wandb",
        seed=seed,
        fp16=True,
    )

    # HF Trainer requires labels to be consecutive, or it will throw a weird error.
    labels = list(set(dataset["train"]["label"]))
    label2id = {lb: i for i, lb in enumerate(labels)}
    id2label = {i: lb for lb, i in label2id.items()}
    ood_label = -1

    def map_label2id(ex):
        if ex["label"] in label2id:
            ex["label"] = label2id[ex["label"]]
        else:
            ex["label"] = ood_label
        return ex

    for phase in ["train", "validation"]:
        dataset[phase] = dataset[phase].map(map_label2id)

    # Setup
    config = AutoConfig.from_pretrained(
        base_model, id2label=id2label, label2id=label2id, num_labels=len(labels)
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, config=config
    )
    if base_model.startswith("roberta") or base_model.startswith("deberta"):
        model.resize_token_embeddings(len(tokenizer))

    # Initialize Trainer based on defined loss
    if ood_training_type == "vanilla":
        print(f"Starting training with {len(dataset['train'])} ID examples")
        trainer = OODEvaluationTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
            compute_metrics=get_compute_metrics(ood_label),
        )
    elif ood_training_type == "ccl":
        print(
            f"Starting CCL training with {len(dataset['train'])} ID examples and {len(ood_dataset)} OOD examples!"
        )
        trainer = CCLTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            ood_dataset=ood_dataset,
            tokenizer=tokenizer,
            compute_metrics=get_compute_metrics(ood_label),
            do_loss_logging=do_loss_logging,
        )
    elif ood_training_type == "oe":
        print(
            f"Starting OE training with {len(dataset['train'])} ID examples and {len(ood_dataset)} OOD examples!"
        )
        trainer = OETrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            ood_dataset=ood_dataset,
            tokenizer=tokenizer,
            compute_metrics=get_compute_metrics(ood_label),
            do_loss_logging=do_loss_logging,
        )
    else:
        raise ValueError

    trainer.train()
    model.save_pretrained(model_path)


if __name__ == "__main__":
    main()
