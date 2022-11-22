import datasets
from datasets import DatasetDict
import click
import os
import json
import pandas as pd
import pickle
from collections import Counter


def download_wikitext(out):
    d = datasets.load_dataset("wikitext", "wikitext-103-v1")
    d = d["train"]
    d.save_to_disk(os.path.join(out, "wikitext"))


def get_special_token(w, special_tokens):
    if w not in special_tokens:
        special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
    return special_tokens[w]


def label_to_num(label, labels):
    if label not in labels:
        labels[label] = len(labels)
    return labels[label]


def preprocess_tacred(sample, special_tokens, labels):
    """
    Returns a list of sequence tokens with spans replaced by POS tag following Zhang et al. 2017
    We need to get the masked sentence, unmasked sentence, tokens to be masked [s+o], label
    From https://github.com/yuhaozhang/tacred-relation
    """
    new_tokens = []
    tokens = sample["token"]
    ner_tag_s = sample["subj_type"]
    ner_tag_o = sample["obj_type"]
    SUBJECT_START = get_special_token("SUBJ_START", special_tokens)
    SUBJECT_END = get_special_token("SUBJ_END", special_tokens)
    OBJECT_START = get_special_token("OBJ_START", special_tokens)
    OBJECT_END = get_special_token("OBJ_END", special_tokens)
    SUBJECT_NER = get_special_token("SUBJ=%s" % ner_tag_s, special_tokens)
    # SUBJECT_NER = get_special_token("SUBJ", special_tokens)
    OBJECT_NER = get_special_token("OBJ=%s" % ner_tag_o, special_tokens)
    # OBJECT_NER = get_special_token("OBJ", special_tokens)
    ss, se, os, oe = [
        sample[x] for x in ["subj_start", "subj_end", "obj_start", "obj_end"]
    ]
    subj_tokens = []
    obj_tokens = []
    for i, token in enumerate(tokens):
        if i == ss:
            new_tokens.append(SUBJECT_START)
            new_tokens.append(SUBJECT_NER)
            new_tokens.append(SUBJECT_END)
        if i == os:
            new_tokens.append(OBJECT_START)
            new_tokens.append(OBJECT_NER)
            new_tokens.append(OBJECT_END)
        if i >= ss and i <= se:
            subj_tokens.append(tokens[i])
        elif i >= os and i <= oe:
            obj_tokens.append(tokens[i])
        else:
            new_tokens.append(tokens[i])
    return {
        "text": " ".join(new_tokens),
        "label": label_to_num(sample["relation"], labels),
        "original_text": " ".join(tokens),
        "subject": " ".join(subj_tokens),
        "object": " ".join(obj_tokens),
    }


def build_split(
    dataset_name, dataset_out_name, out, split, seed, tacred_path=None, ood_multipliers=[1.0]
):
    """
    Build and save to disk a single split of one dataset, with some set of labels held out
    """
    mask_labels = [split]
    # Load Datasets
    if dataset_name in ["ag_news", "trec", "emotion"]:
        data = datasets.load_dataset(dataset_name)
        for phase in data:
            data[phase]._info.__dict__["task_templates"] = []
    elif dataset_name == "tacred":
        assert tacred_path is not None
        splits = ["train", "dev", "test"]
        special_tokens = {}
        data = datasets.DatasetDict(
            {
                split: datasets.Dataset.from_pandas(
                    pd.DataFrame(
                        json.load(
                            open(os.path.join(tacred_path, f"data/json/{split}.json"))
                        )
                    )
                )
                for split in splits
            }
        )
        data["test"] = datasets.concatenate_datasets([data["dev"], data["test"]])
        counts = Counter(data["test"]["relation"])
        label_set = set(data["test"]["relation"])
        sorted_label_set = sorted(label_set, key=lambda x: counts[x], reverse=True)
        n = 6  # Top n kept as ID, rest as OOD
        labels = {
            label: i + 1 if i < n else 0 for i, label in enumerate(sorted_label_set)
        }
        data = data.map(lambda ex: preprocess_tacred(ex, special_tokens, labels))
        os.makedirs(os.path.join(out, dataset_out_name, split), exist_ok=True)
        with open(os.path.join(out, dataset_out_name, split, "ner_tag_map.pkl"), "wb") as f:
            pickle.dump(special_tokens, f)
        with open(os.path.join(out, dataset_out_name, split, "labels_map.pkl"), "wb") as f:
            pickle.dump(labels, f)
    # Process Datasets
    delete_labels = []
    delete_columns = []
    if dataset_name == "ag_news":
        process_example = lambda ex: ex
        label_column_name = "label"
        split_p = {"train": 0.7, "validation": 0.1, "itest": 0.1, "test": 0.1}
    elif dataset_name == "tacred":
        process_example = lambda ex: ex
        delete_columns = [
            "id",
            "docid",
            "relation",
            "token",
            "subj_start",
            "subj_end",
            "obj_start",
            "obj_end",
            "subj_type",
            "obj_type",
            "stanford_pos",
            "stanford_ner",
            "stanford_head",
            "stanford_deprel",
        ]
        label_column_name = "label"
        split_p = {"train": 0.7, "validation": 0.1, "itest": 0.1, "test": 0.1}
    elif dataset_name == "trec":
        process_example = lambda ex: ex
        delete_labels = [2]
        data = data.remove_columns("label-fine").rename_column("label-coarse", "label")
        label_column_name = "label"
        split_p = {"train": 0.6, "validation": 0.1, "itest": 0.15, "test": 0.15}
    elif dataset_name == "emotion":
        process_example = lambda ex: ex
        delete_labels = [2, 5]
        label_column_name = "label"
        data["train"] = datasets.concatenate_datasets(
            [data["train"], data["validation"]]
        )
        split_p = {"train": 0.7, "validation": 0.1, "itest": 0.1, "test": 0.1}
    data = data.shuffle(seed=seed)
    # Turn all labels into numbers if necessary
    if all([label.isdigit() for label in mask_labels]):
        mask_labels = [int(label) for label in mask_labels if label.isdigit()]
    # Delete some undesired classes
    data = data.filter(lambda ex: ex[label_column_name] not in delete_labels)
    data = data.remove_columns(delete_columns)
    # Merge all the splits so we can resplit it ourselves
    if "test" in data.keys():
        data = datasets.concatenate_datasets([data["train"], data["test"]])
    else:
        data = data["train"]
    data = data.shuffle(seed=seed)
    # Hold out ALL the OOD data
    ood_data = data.filter(lambda ex: ex[label_column_name] in mask_labels)
    id_data = data.filter(lambda ex: ex[label_column_name] not in mask_labels)
    # Split off train
    id_data = id_data.train_test_split(train_size=split_p["train"], shuffle=False)
    train, nontrain_data = id_data["train"], id_data["test"]
    full_split = {"train": train}
    for ood_multiplier in ood_multipliers:
        # Join rest of data with selected OOD percent
        ood_p = (1 - split_p["train"]) * ood_multiplier
        assert ood_p <= 1, "OOD multiplier is too high!"
        rest_data = datasets.concatenate_datasets(
            [nontrain_data, ood_data.select(range(int(len(ood_data) * ood_p)))]
        )
        rest_data = rest_data.shuffle(seed=seed)
        heldout_data = ood_data.select(range(int(len(ood_data) * ood_p), len(ood_data)))
        # Split up rest of data
        rest_data = rest_data.train_test_split(
            train_size=(
                split_p["validation"]
                / (split_p["validation"] + split_p["itest"] + split_p["test"])
            ),
            shuffle=False,
        )
        validation, rest_data = rest_data["train"], rest_data["test"]
        rest_data = rest_data.train_test_split(
            train_size=(split_p["itest"] / (split_p["itest"] + split_p["test"])),
            shuffle=False,
        )
        itest, test = rest_data["train"], rest_data["test"]
        if ood_multiplier == 1.0:
            suffix = ""
        elif ood_multiplier.is_integer():
            suffix = "_" + str(int(ood_multiplier)) + "x"
        else:
            suffix = "_" + str(ood_multiplier) + "x"
        full_split["validation" + suffix] = validation
        full_split["test" + suffix] = itest
        full_split["test-final" + suffix] = test
        full_split["heldout" + suffix] = heldout_data
    full_split = DatasetDict(full_split)
    # Save discrim data
    full_split.save_to_disk(os.path.join(out, dataset_out_name, split))


@click.command()
@click.option("--out_path", type=str, required=True)
@click.option("--tacred_data_path", default="raw_data/tacred/")
def main(out_path, tacred_data_path):
    settings = {
        "emotion": ["emotion", ["0", "1", "3", "4"]],
        "trec": ["trec10", ["0", "1", "3", "4", "5"]],
        "ag_news": ["agnews", ["0", "1", "2", "3"]],
        "tacred": ["tacred", ["0"]],
    }
    seed = 1  # Hardcode for determinism
    for dataset, ds in settings.items():
        dataset_out_name, splits = ds
        for split in splits:
            print(f"Constructing {dataset} split {split}, with random seed {seed}")
            if dataset == "tacred":
                build_split(
                    dataset, dataset_out_name, out_path, split, seed, tacred_path=tacred_data_path
                )
            else:
                build_split(dataset, dataset_out_name, out_path, split, seed)
    download_wikitext(out_path)


if __name__ == "__main__":
    main()
