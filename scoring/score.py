import os
import sys
import pickle
import click
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from metrics import metric2score, sp_auac, ood_det_auroc, top1_acc
from models import UncertaintyClassifier


class NoPrint:
    """
    Wraps code to suppress all printing.
    From: https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Evaluator:
    """
    Evaluate:
    (1) a given model
    (2) w.r.t. OOD detection/OSSC metrics
    (3) on some dataset+split
    """

    def __init__(self, model_path, train_dataset_path):
        self.model_path = model_path
        train_dataset = Dataset.load_from_disk(train_dataset_path)
        self.train_labels = set(train_dataset["label"])

    def score(self, test_dataset_path, save_path=None):
        """
        Evaluate on a test dataset, with some number of new labels
        Returns a dict of form:
        Dict[ood_label (str): Dict[["id-acc", "auroc", "auac"]]]
        """
        preds, confs, labels = self._evaluate(test_dataset_path)
        if save_path is not None:
            with open(save_path, "wb") as f:
                pickle.dump({"preds": preds, "confs": confs, "labels": labels}, f)
        ood_labels = list(set(labels).difference(set(self.train_labels)))
        results = {label: {} for label in ood_labels}
        for i, ood_label in enumerate(ood_labels):
            results_per_label = results[ood_label]
            metrics = {
                "id-acc": metric2score(top1_acc, ood_label=ood_label),
                "auroc": metric2score(ood_det_auroc, ood_label=ood_label),
                "auac": metric2score(sp_auac),
            }
            for metric_name, metric in metrics.items():
                results_per_label[metric_name] = metric(preds, confs, labels)
                #print(results_per_label)
        return results

    def _evaluate(self, test_dataset_path):
        test_dataset = Dataset.load_from_disk(test_dataset_path)
        test_dataset, test_golds = test_dataset["text"], test_dataset["label"]

        with NoPrint():
            model = UncertaintyClassifier(self.model_path, cuda=True)
        preds, confs_maxprob = model.batched_call(test_dataset)
        return np.array(preds), np.array(confs_maxprob), np.array(test_golds)


def get_splits(dataset_name):
    if dataset_name == "trec10":
        train_splits = ["0", "1", "3", "4", "5"]
    elif dataset_name == "emotion":
        train_splits = ["0", "1", "3", "4"]
    elif dataset_name == "agnews":
        train_splits = ["0", "1", "2", "3"]
    elif dataset_name == "tacred":
        train_splits = ["0"]
    return train_splits


def aggregate_df(df):
    """
    macro-average over splits
    """
    means = []
    stds = []
    counts = []
    for _, split in df.iterrows():
        means.append(np.nanmean(split))
        stds.append(np.nanstd(split))
        counts.append(split.count())
    return 100 * np.nanmean(means), 100 * np.nanmean(stds) / np.sqrt(min(counts))


@click.command()
@click.argument("model_name")
@click.argument("dataset")
@click.option("--num_trials", default=5)
@click.option("--debug/--no_debug", type=bool, default=False)
@click.option("--save_path", type=str, default=None)
def main(model_name, dataset, num_trials, debug, save_path):
    train_splits = get_splits(dataset)
    trials = [str(x) for x in range(1, num_trials + 1)]
    all_results = {"auac": [], "auroc": [], "id-acc": []}
    for split in tqdm(train_splits):
        results_row = {"auac": [], "auroc": [], "id-acc": []}
        for t in trials:
            # Try to score the split
            try:
                model_path = f"models/{dataset}/{split}/{model_name}_seed_{t}/"
                train_data_path = f"data/{dataset}/{split}/train/"
                test_data_path = f"data/{dataset}/{split}/test-final/"
                model_eval = Evaluator(model_path, train_data_path)
                assert os.path.exists(model_path) and os.listdir(model_path), f"Couldn't find {model_path}"
                if save_path:
                    eval_save_path = os.path.join(save_path, dataset, split)
                    os.makedirs(eval_save_path, exist_ok=True)
                    eval_results = model_eval.score(test_data_path, os.path.join(eval_save_path, f"{model_name}_seed_{t}"))
                else:
                    eval_results = model_eval.score(test_data_path)
                assert len(eval_results) == 1  # Should be only 1 OOD label
                eval_results = list(eval_results.values())[0]
                # Add item to row of DF
                for metric_name, score in eval_results.items():
                    results_row[metric_name].append(score)
            except AssertionError as _:
                #raise
                print("Split", split, "Trial", t, "errored.")
                for metric_name, score in results_row.items():
                    score.append(np.NaN)
            except Exception as _:
                if debug:
                    raise
                else:
                    pass
        # Add completed row to DF and convert to DF
        for metric_name, row in results_row.items():
            all_results[metric_name].append(row)
    for metric_name in all_results.keys():
        all_results[metric_name] = pd.DataFrame(all_results[metric_name])

    print(f"*** All stats: {model_name} on {dataset} ***")
    if debug:
        for name, df in all_results.items():
            print(name)
            print(df)
    all_results = {name: aggregate_df(df) for name, df in all_results.items()}
    print(
        "auac:",
        np.round(all_results["auac"][0], 1),
        "+-",
        np.round(all_results["auac"][1], 1),
    )
    print(
        "auroc:",
        np.round(all_results["auroc"][0], 1),
        "+-",
        np.round(all_results["auroc"][1], 1),
    )
    print(
        "id-acc:",
        np.round(all_results["id-acc"][0], 1),
        "+-",
        np.round(all_results["id-acc"][1], 1),
    )


if __name__ == "__main__":
    main()
