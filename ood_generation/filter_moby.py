import pickle
from itertools import chain
import click
import numpy as np
import datasets
import mobypy
from ood_generation.generation_utils import load_label_mapping

@click.command()
@click.argument("dataset")
@click.option("--input_dataset", default="generations_100k")
@click.option("--output_dataset", default="generations_100k_filtered")
def main(dataset, input_dataset, output_dataset):
    label_to_name_mapping = load_label_mapping(dataset)
    fraction_deleted = []

    for ood_class in label_to_name_mapping.keys():
        input_dataset = f"{dataset}/{ood_class}/{input_dataset}/"
        output_dataset = f"{dataset}/{ood_class}/{output_dataset}/"
        id_labels = [
            label for i, label in label_to_name_mapping.items() if i != ood_class
        ]
        d = datasets.load_from_disk(input_dataset)
        labels = list(set(d["label"]))
        synonyms = list(
            set(chain.from_iterable([mobypy.synonyms(label) for label in id_labels]))
        )
        keep_labels = [label for label in labels if label not in synonyms]
        removed_labels = [label for label in synonyms if label in labels]
        print(f"Went from {len(labels)} labels to {len(keep_labels)}")
        fraction_deleted.append(len(removed_labels) / len(labels))
        print("Deleted labels", removed_labels)
        new_d = d.filter(lambda ex: ex["label"] in keep_labels)
        print(f"Went from {len(d)} to {len(new_d)} examples for split {ood_class}")
        new_d.save_to_disk(output_dataset)

    print("Average deleted", np.mean(fraction_deleted))


if __name__ == "__main__":
    main()
