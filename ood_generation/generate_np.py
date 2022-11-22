import os
import random
import click
import datasets
from transformers import set_seed
import openai
from ood_generation.generation_utils import (
    GPT3Model,
    GeneratorModel,
    LabelGenerator,
    ExampleGenerator,
    preprocess_tacred,
    load_ner_tags,
    load_label_mapping,
    is_not_noise,
)
from api_keys import OPENAI_API_KEY


@click.command()
@click.option(
    "--dataset",
    type=str,
    required=True,
    help="Name of dataset to generate based on, e.g., trec10",
)
@click.option(
    "--gold_ood_class",
    type=int,
    required=True,
    help="Index of gold OOD class to holdout from generation",
)
@click.option(
    "--example_generator_size", 
    type=str, 
    required=True, 
    help="Model sizes, see generation_utils.py"
)
@click.option(
    "--label_generator_size", 
    type=str, 
    required=True, 
    help="Model sizes, see generation_utils.py"
)
@click.option(
    "--num_generations", 
    type=int, 
    required=True, 
    help="Number of examples to generate"
)
@click.option(
    "--output_dir_name",
    type=str,
    required=True,
    help="Where to save examples (as a HF Dataset)",
)
@click.option(
    "--class_to_generate",
    type=int,
    default=None,
    help="Used to specify a gold class index to generate",
)
@click.option(
    "--do_moby/--no_moby",
    type=bool,
    default=False,
    help="Whether to use Moby Thesaurus to filter labels",
)
@click.option(
    "--generation_batch_size",
    type=int,
    default=1,
    help="Number of examples to generate per each prompt",
)
@click.option(
    "--label_generation_iterations",
    type=int,
    default=5,
    help="Number of label generation sets to union together",
)
@click.option(
    "--match_labels_path",
    type=str,
    default=None,
    help="Skip label generation and copy labels from a list",
)
@click.option(
    "--max_num_prompt_context",
    type=int,
    default=None,
    help="Restrict maximum number of IC examples. Prevents exceeding seqlen",
)
@click.option(
    "--num_context_per_class",
    type=int,
    default=1,
    help="Number of in-context examples to put in the prompt",
)
@click.option(
    "--seed", 
    type=int, 
    default=1, 
    help="Random seed"
)
def main(
    dataset,
    gold_ood_class,
    example_generator_size,
    label_generator_size,
    num_generations,
    output_dir_name,
    class_to_generate,
    do_moby,
    generation_batch_size,
    label_generation_iterations,
    match_labels_path,
    max_num_prompt_context,
    num_context_per_class,
    seed,
):
    # 0. Setup
    class2label = load_label_mapping(dataset)
    openai.api_key = OPENAI_API_KEY
    train_dataset_path = f"data/{dataset}/{gold_ood_class}/train/"
    set_seed(seed)
    random.seed(seed)

    label_to_generate = (
        class2label[class_to_generate] if class_to_generate is not None else None
    )
    gold_ood_label = class2label[gold_ood_class]
    train_dataset = datasets.load_from_disk(train_dataset_path)
    id_classes = list(set(train_dataset["label"]))
    id_labels = [class2label[cls] for cls in id_classes]
    if dataset == "tacred":  # Hack to combine the text + subj/obj together
        token2ner = load_ner_tags(
            f"data/tacred/{gold_ood_class}/ner_tag_map.pkl"
        )
        train_dataset = train_dataset.map(
            lambda x: preprocess_tacred(x, token2ner)
        )
    else:
        token2ner = None
    # 1. Label Generation
    if match_labels_path is not None:
        print("*** Matching labels from: ***", match_labels_path)
        reference_d = datasets.load_from_disk(match_labels_path)
        ood_labels = list(set(reference_d["label"]))
        print(ood_labels)
    elif class_to_generate is not None:
        print("*** Generating the class: ***")
        print(class_to_generate)
        ood_labels = [class_to_generate]
    else:
        if label_generator_size == "gpt3":
            label_generator_model = GPT3Model("text-davinci-002")
        else:
            label_generator_model = GeneratorModel(label_generator_size)
        label_generator = LabelGenerator(label_generator_model)
        # TODO: Add kwargs for extra hyperparams
        ood_labels = label_generator(
            dataset, id_labels, gold_ood_label, filter_moby=do_moby, num_iterations=label_generation_iterations,
        )
        ood_labels = [lb for lb in ood_labels if is_not_noise(lb)]
        if dataset != "tacred":
            ood_labels = [lb for lb in ood_labels if " " not in lb]
        print("*** Generating from the following labels! ***")
        print(ood_labels)
    del label_generator
    del label_generator_model

    # 2. Example Generation
    prompt_base = """Given a label, generate a corresponding example:\n"""
    generator_model = GeneratorModel(example_generator_size)
    example_generator = ExampleGenerator(generator_model)
    all_generations, generations_label_reference = example_generator(
        train_dataset=train_dataset,
        dataset_name=dataset,
        ood_labels=ood_labels,
        prompt_base=prompt_base,
        num_generations=num_generations,
        generation_batch_size=generation_batch_size,
        context_per_class=num_context_per_class,
        max_num_context=max_num_prompt_context,
        id_classes=id_classes,
        class2label=class2label,
        token_to_ner_mapping=token2ner,
    )
    print(f"*** Generated {len(all_generations)} examples total! ***")
    print("Some example generations:\n" + "\n".join(all_generations[:10]))
    # Save to disk
    generated_dataset = datasets.Dataset.from_dict(
        {"text": all_generations, "label": generations_label_reference}
    )
    os.makedirs(f"generations/{dataset}/{gold_ood_class}/{output_dir_name}", exist_ok=True)
    generated_dataset.save_to_disk(
        f"generations/{dataset}/{gold_ood_class}/{output_dir_name}"
    )
    # Write a text file for easy interpretation
    with open(
        f"generations/{dataset}/{gold_ood_class}/{output_dir_name}/generations.txt", "w"
    ) as f:
        for line in all_generations:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
