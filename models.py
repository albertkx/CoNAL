from tqdm import tqdm
from scipy.stats import entropy
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
    DebertaV2Config,
    BertConfig,
    RobertaConfig,
)


class UncertaintyClassifier:
    """
    Wraps model, instantiating from a checkpoint.
    Classifies and yields an associated confidence score.
    """

    def __init__(self, model_path, label_path=None, cuda=False, max_length=128):
        # Model Setup
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        if isinstance(self.model.config, BertConfig):
            tokenizer_name = "bert-base-cased"
        elif isinstance(self.model.config, RobertaConfig):
            tokenizer_name = "roberta-base"
            self.model.resize_token_embeddings(len(self.tokenizer))
        elif isinstance(self.model.config, DebertaV2Config):
            tokenizer_name = "microsoft/deberta-v3-base"
        else:
            raise ValueError("Unknown model")
        # Tokenizer setup
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, kwargs={"model_max_length": 128}
        )
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [f"[unused{k}]" for k in range(1000)]}
        )
        self.max_length = max_length
        # Move to GPU if necessary
        if cuda:
            self.model = self.model.to('cuda')
            self.p = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device="cuda:0",
            )
        else:
            self.p = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device="cpu",
            )
        if isinstance(label_path, list):
            self.labels = dict(zip(range(len(label_path)), label_path))
        else:
            self.labels = self.model.config.id2label

    def __call__(self, sent):
        """
        Call on a single example and return maxprob confidence
        """
        model_output = self.p(sent, truncation=True, max_length=self.max_length)
        return (
            max(model_output, key=lambda x: x["score"])["label"],
            model_output[0]["score"],
        )

    def ood_mass(self, sent, ood_labels):
        """
        Returns total probability mass on OOD labels
        """
        model_output = self.p(
            sent, return_all_scores=True, truncation=True, max_length=self.max_length
        )[0]
        mass = sum(x["score"] for x in model_output if x["label"] in ood_labels)
        return mass

    def cuda(self):
        self.p.device = "cuda:0"
        return self

    def batched_call(self, sents, batch_size=5):
        """
        Call on a batch of examples and return outputs and confidences for all examples
        """
        model_outputs = self.p(
            sents,
            batch_size=1,
            truncation=True,
            max_length=self.max_length
        )
        # TODO: Check using log_softmax here for stability?
        labels = [output["label"] for output in model_outputs]
        confidences = [output["score"] for output in model_outputs]
        return labels, confidences


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]
