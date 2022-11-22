import os
import click
import numpy as np
import datasets
from datasets import DatasetDict

@click.command()
@click.option("--out_path", type=str)
@click.option("--data_path", type=str)
@click.option("--dataset", type=str, default="all")
@click.option("--seed", type=int, default=1)
def main(out_path, data_path, dataset, seed):
    settings = {
        "emotion": ["0", "1", "3", "4"],
        "trec10": ["0", "1", "3", "4", "5"],
        "agnews": ["0", "1", "2", "3"],
        "tacred": ["0"],
    }
    id_ps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    all_dsets = settings.keys()
    if dataset == "all":
        dsets = all_dsets
    else:
        assert dataset in all_dsets
        dsets = [dataset]
    for dataset_name in dsets:
        splits = settings[dataset_name]
        for split in splits:
            dd = DatasetDict()
            split_out_path = os.path.join(out_path, dataset_name, split)
            id_dataset_path = os.path.join(data_path, dataset_name, split)
            os.makedirs(split_out_path, exist_ok=True)
            dataset = datasets.load_from_disk(id_dataset_path)
            dataset = dataset.shuffle(seed=seed)
            #id_dataset = dataset["train"]
            id_labels = set(dataset['train']['label'])
            id_dataset = dataset['test'].filter(lambda x: x['label'] in id_labels)
            ood_dataset = dataset["heldout"]
            max_length = min(len(id_dataset), len(ood_dataset))
            for id_p in id_ps:
                id_selected = id_dataset.select(range(1, int(max_length * id_p)))
                ood_selected = ood_dataset.select(range(1, int(max_length * (1 - id_p))))
                noised_dataset = datasets.concatenate_datasets([id_selected, ood_selected])
                dd[str(id_p)] = noised_dataset
            dd.save_to_disk(os.path.join(split_out_path, "noised_ood_set"))
            print(f"wrote ID noised OOD set of {dataset_name}/{split}")
    print("done")

if __name__ == "__main__":
    main()
