import datasets
import itertools
import random
import click
import torch
import pickle
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from generation_utils import call

@click.command()
@click.option("--dataset", type=str)
@click.option("--true_split_number", type=int)
@click.option("--num_generations", type=int)
@click.option("--generation_batch_size", type=int)
@click.option("--output_dir_name", type=str, default=None)
def main(dataset, true_split_number, num_generations, output_dir_name, generation_batch_size):
    # Parameters
    seed = 1 # Random seed

    # Setup
    if dataset == "emotion":
        prompt = """Generate a tweet:"""
    elif dataset == "agnews":
        prompt = """Generate a news title:"""
    elif dataset == "trec10":
        prompt = """Generate a question:"""
    elif dataset == "tacred":
        prompt = """Generate a sentence with a relation:"""

    set_seed(seed)
    random.seed(seed)
    global model, tokenizer
    model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B",
            max_length=256).cuda()
    tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-j-6B",
            use_fast=False)
    all_generations = []
    pbar = tqdm(total = num_generations)
    while len(all_generations) < num_generations:
        step = len(all_generations)
        pbar.n = step
        pbar.refresh()
        if step == 0:
            print("*** Example Prompt: ***")
            print(prompt)
        # Query GPTJ
        completions = call(prompt, model, num_to_generate=generation_batch_size)
        for completion in completions:
            generation = completion.split("\n")[0].strip()
            generation = generation.strip("_")
            generation = generation.strip()
            if generation != "":
                all_generations.append(generation)

    print(f"*** Generated {len(all_generations)} examples total! ***")
    print("Some example generations:\n" + "\n".join(all_generations[:10]))
    # Convert to dataset
    generated_dataset = datasets.Dataset.from_dict({"text": all_generations})
    if output_dir_name is None:
        output_dir_name = "generations"
    generated_dataset.save_to_disk(f"generations/{dataset}/{true_split_number}/{output_dir_name}")
    with open(f"generations/{dataset}/{true_split_number}/{output_dir_name}/generations.txt", "w") as f:
        for line in all_generations:
            f.write(line + "\n")

if __name__ == "__main__":
    main()
