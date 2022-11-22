import numpy as np
from sklearn import metrics
from torch.nn import functional as F
import torch
from datasets import load_metric

def top1_acc(eval_pred):
    """
    Returns the top1 exact-match accuracy of the model
    If an ood_label is passed in, remove those instances first
    """
    labels, predictions, ood_label = eval_pred["labels"], eval_pred["predictions"], eval_pred["ood_label"]
    if ood_label is not None:
        keep_indices = labels != ood_label
        predictions = predictions[keep_indices]
        labels = labels[keep_indices]
    metric = load_metric("accuracy")
    return metric.compute(predictions=predictions, references=labels)["accuracy"]


def ood_det_auroc(eval_pred):
    """
    Computes the area under the ROC curve of classifying whether test examples are ID or OOD
    """
    labels, confidences, ood_label = eval_pred["labels"], eval_pred["confidences"], eval_pred["ood_label"]
    assert ood_label is not None
    id_indicators = labels != ood_label
    return metrics.roc_auc_score(id_indicators, confidences)


def sp_auroc(eval_pred):
    """
    Computes the area under the ROC curve of selective prediction
    Equals to 1 if the model perfectly predicts which examples it gets right.
    """
    corrects, confidences = eval_pred["corrects"], eval_pred["confidences"]
    return metrics.roc_auc_score(corrects, confidences)


def sp_auac(eval_pred):
    """
    Computes the Area under the Accuracy-Coverage curve for open-set selective classificaiton
    Cannot equal 1 unless the model achieves 100% test acc.
    """
    corrects, confidences = eval_pred["corrects"], eval_pred["confidences"]
    _, corrects = list(
        zip(*sorted(zip(confidences, corrects), reverse=True, key=lambda x: x[0]))
    )  # Sort by conf
    x = np.arange(1, len(corrects) + 1)
    cumulative_accs = np.cumsum(corrects) / x
    return metrics.auc(x / len(x), cumulative_accs)


def metric2hf(metric, ood_label=None):
    """
    Returns a callable metric for Huggingface Trainer
    """
    def hf_metric(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        corrects = predictions == labels
        # TODO: Check log_softmax here
        confidences = np.max(F.softmax(torch.Tensor(logits), dim=-1).numpy(), axis=-1)
        corrects = (predictions == labels)
        inference = {"labels": labels,
                "predictions": predictions,
                "confidences": confidences,
                "corrects": corrects,
                "ood_label": ood_label}
        return metric(inference) 
    return hf_metric

def metric2score(metric, ood_label=None):
    """
    Returns an easy callable for our post-hoc scoring.
    """
    def score_metric(preds, confs, labels):
        assert isinstance(preds, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert isinstance(confs, np.ndarray)
        assert all(isinstance(x, np.int64) for x in preds)
        assert all(isinstance(x, np.int64) for x in labels)
        assert all(isinstance(x, np.float64) for x in confs)
        corrects = (preds == labels)
        inference = {"labels": labels,
                "predictions": preds,
                "confidences": confs,
                "corrects": corrects,
                "ood_label": ood_label}
        return metric(inference) 
    return score_metric

