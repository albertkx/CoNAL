import datasets
import itertools
import random
import click
import torch
import pickle
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from generation_utils import call

def make_tacred_example(sample, token_to_ner_mapping):
    text = sample["text"]
    subj_start_token = [token for token, ner in token_to_ner_mapping.items() if ner == "SUBJ_START"][0]
    subj_end_token = [token for token, ner in token_to_ner_mapping.items() if ner == "SUBJ_END"][0]
    obj_start_token = [token for token, ner in token_to_ner_mapping.items() if ner == "OBJ_START"][0]
    obj_end_token = [token for token, ner in token_to_ner_mapping.items() if ner == "OBJ_END"][0]
    text = text.replace(subj_start_token + " ", "[Subject: ")
    text = text.replace(" " + subj_end_token, "]")
    text = text.replace(obj_start_token + " ", "[Object: ")
    text = text.replace(" " + obj_end_token, "]")
    for token, ner in token_to_ner_mapping.items():
        if "SUBJ_" not in ner and "OBJ_" not in ner:
            ner = ner.split("=")[1].lower().replace("_", " ")
            text = text.replace(token, ner)
    return {"text": text}
    #return {"text": sample["original_text"] + ", " + sample["subject"] + ", " + sample["object"]}

def slice_assign(string, span, replacement):
    start_idx, end_idx = span
    return string[:start_idx] + replacement + string[end_idx:]

def postprocess_tacred(generation, token_to_ner_mapping):
    new_mapping = {ner.split("=")[1].lower().replace("_", " "): token 
            for token, ner 
            in token_to_ner_mapping.items()
            if "SUBJ_" not in ner and "OBJ_" not in ner}
    subj_start_token = [token for token, ner in token_to_ner_mapping.items() if ner == "SUBJ_START"][0]
    subj_end_token = [token for token, ner in token_to_ner_mapping.items() if ner == "SUBJ_END"][0]
    obj_start_token = [token for token, ner in token_to_ner_mapping.items() if ner == "OBJ_START"][0]
    obj_end_token = [token for token, ner in token_to_ner_mapping.items() if ner == "OBJ_END"][0]
    if generation.count("[Subject:") != 1 or generation.count("[Object:") != 1 or generation.count("]") != 2:
        return None
    # Subject
    subj_start_idx = generation.index("[Subject:")
    if "]" not in generation[subj_start_idx:]:
        # didn't generate a closing bracket anywhere after the subject start
        return None
    subj_end_idx =  generation.index("]", subj_start_idx)
    generated_ner = generation[subj_start_idx + 9: subj_end_idx].strip().replace(" ", "_")
    if generated_ner not in new_mapping.keys():
        # didn't generate a real ner token
        return None
    subj_replacement = subj_start_token + " " + new_mapping[generated_ner] + " " + subj_end_token
    generation = slice_assign(generation, (subj_start_idx, subj_end_idx + 1), subj_replacement)
    # Object
    obj_start_idx = generation.index("[Object:")
    if "]" not in generation[obj_start_idx:]:
        # didn't generate a closing bracket anywhere after the object start
        return None
    obj_end_idx =  generation.index("]", obj_start_idx)
    generated_ner = generation[obj_start_idx + 8: obj_end_idx].strip().replace(" ", "_")
    if generated_ner not in new_mapping.keys():
        # didn't generate a real ner token
        return None
    obj_replacement = obj_start_token + " " + new_mapping[generated_ner] + " " + obj_end_token
    generation = slice_assign(generation, (obj_start_idx, obj_end_idx + 1), obj_replacement)
    return generation

def check(generation, dataset):
    if "=>" in generation:
        return False
    return True

@click.command()
@click.option("--dataset", type=str)
@click.option("--true_split_number", type=int)
@click.option("--num_generations", type=int)
@click.option("--generation_batch_size", type=int)
@click.option("--seed", type=int)
@click.option("--max_num_context", type=int)
@click.option("--output_dir_name", type=str, default=None)
def main(dataset, true_split_number, num_generations, output_dir_name, generation_batch_size, max_num_context, seed):
    # Parameters
    
    train_dataset_path = f"data/{dataset}/{true_split_number}/train/"
    train_dataset = datasets.load_from_disk(train_dataset_path)
    train_labels = list(set(train_dataset["label"]))
    train_examples = train_dataset["text"]
    # Setup
    if dataset == "emotion":
        prompt = """Generate a tweet:"""
    elif dataset == "agnews":
        prompt = """Generate a news title:"""
    elif dataset == "trec10":
        prompt = """Generate a question:"""
    elif dataset == "tacred":
        prompt = """Generate a sentence with a relation:"""
        ner_tags_path = f"data/tacred/{true_split_number}/ner_tag_map.pkl"
        with open(ner_tags_path, "rb") as f:
            ner_to_token_mapping = pickle.load(f)
        token_to_ner_mapping = {tok: ner for ner, tok in ner_to_token_mapping.items()}
        train_dataset = train_dataset.map(lambda x: make_tacred_example(x, token_to_ner_mapping))
    else:
        raise ValueError("Dataset not found")
    sorted_train_dataset = sorted(
            list(train_dataset),
            key=lambda ex: ex["label"])
    grouped_train_dataset = itertools.groupby(
            sorted_train_dataset,
            key=lambda ex: ex["label"])
    grouped_train_examples = {label: [ex["text"] for ex in examples]
            for label, examples
            in grouped_train_dataset}

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
        # Build Prompt
        formatted_prompt = prompt
        for i, label in enumerate(train_labels):
            if max_num_context is not None and i >= max_num_context:
                break
            example = random.choice(grouped_train_examples[label])
            formatted_prompt += f"\nExample: {example}"
        formatted_prompt += "\nExample:"
        if step == 0:
            print("*** Example Prompt: ***")
            print(formatted_prompt)
        # Query GPTJ
        completions = call(formatted_prompt, model, num_to_generate=generation_batch_size)
        for completion in completions:
            generation = completion.split("\n")[0].strip()
            generation = generation.strip("_")
            generation = generation.strip()
            if dataset == "tacred":
                generation = postprocess_tacred(generation, token_to_ner_mapping)
                if generation is None:
                    continue
            if generation != "" and check(generation, dataset):
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
