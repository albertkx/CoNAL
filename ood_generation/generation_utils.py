import pickle
import random
from itertools import chain, groupby
from tqdm import tqdm
import openai
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
)
import mobypy


class LabelGenerator:
    """
    Runs label generation phase for a given LM generator
    """

    def __init__(self, generator):
        assert isinstance(generator, GPT3Model) or isinstance(generator, GeneratorModel)
        self.generator = generator

    def __call__(
        self,
        dataset,
        id_labels,
        gold_ood_label,
        num_iterations=1,
        filter_moby=False,
        **kwargs,
    ):
        prompt = self._get_label_generation_prompt(dataset, id_labels)
        all_labels = id_labels + [gold_ood_label]
        ood_labels = set()
        for _ in tqdm(range(num_iterations)):
            generation = self.generator(prompt, stop_seq="Label:", **kwargs)[0]
            generation = generation.replace("]", "")
            ood_labels_generated = generation.split(",")
            ood_labels_generated = set(
                [
                    label.strip().lower()
                    for label in ood_labels_generated
                    if label.strip().lower() not in all_labels
                    and "\n" not in label.strip()
                    and label.strip() != ""
                ]
            )
            ood_labels = ood_labels.union(ood_labels_generated)
        ood_labels = list(ood_labels)
        if filter_moby:
            ood_labels = self._filter_moby(ood_labels, id_labels)
        return ood_labels

    @staticmethod
    def _filter_moby(ood_labels, id_labels):
        num_labels_before = len(ood_labels)
        synonyms = list(
            set(chain.from_iterable([mobypy.synonyms(label) for label in id_labels]))
        )
        ood_labels = [label for label in ood_labels if label not in synonyms]
        num_labels_after = len(ood_labels)
        print("Removed", num_labels_before - num_labels_after, "labels!")
        return ood_labels

    @staticmethod
    def _get_label_generation_prompt(dataset, id_labels):
        if dataset == "emotion":
            prompt = """Generate a diverse list of emotions:\n["""
        elif dataset == "agnews":
            prompt = """Generate a diverse list of news genres:\n["""
        elif dataset == "trec10":
            prompt = """Generate a diverse list of entity types:\n["""
        elif dataset == "tacred":
            prompt = """Generate a diverse list of relations between entities:\n["""
        else:
            prompt = """Generate a complete list of labels for a dataset:\n["""
        for label in random.sample(id_labels, k=len(id_labels)):
            prompt += label + ", "
        return prompt


class ExampleGenerator:
    """
    Runs example generation phase for a given LM generator
    """

    def __init__(self, generator):
        self.generator = generator

    def __call__(
        self,
        train_dataset,
        dataset_name,
        ood_labels,
        prompt_base,
        num_generations,
        generation_batch_size,
        context_per_class,
        max_num_context,
        id_classes,
        class2label,
        token_to_ner_mapping,
    ):
        grouped_train_examples = self._group_train_examples(train_dataset)
        all_generations = []
        label_per_generation = []
        pbar = tqdm(total=num_generations)
        while len(all_generations) < num_generations:
            step = len(all_generations)
            pbar.n = step
            pbar.refresh()
            prompt = prompt_base
            # Add Context
            for _ in range(context_per_class):
                for i, cls in enumerate(id_classes):
                    if max_num_context is not None and i >= max_num_context:
                        break
                    example = random.choice(grouped_train_examples[cls])
                    label_name = class2label[cls]
                    prompt += f"{label_name}\n{example}\n"
            # Add Final Label
            ood_label = random.choice(ood_labels)
            prompt += f"{ood_label}\n"
            # Query LM Generator
            generations = self.generator(
                prompt, num_to_generate=generation_batch_size, stop_seq="\n"
            )
            for generation in generations:
                generation = generation.strip("_")
                generation = generation.strip()
                if dataset_name == "tacred":
                    generation = postprocess_tacred(generation, token_to_ner_mapping)
                    if generation is None:
                        continue
                if generation != "" and not "\n" in generation:
                    all_generations.append(generation)
                    label_per_generation.append(ood_label)
        return all_generations, label_per_generation

    @staticmethod
    def _group_train_examples(train_dataset):
        sorted_train_dataset = sorted(list(train_dataset), key=lambda ex: ex["label"])
        grouped_train_dataset = groupby(
            sorted_train_dataset, key=lambda ex: ex["label"]
        )
        grouped_train_examples = {
            label: [ex["text"] for ex in examples]
            for label, examples in grouped_train_dataset
        }
        return grouped_train_examples


class GPT3Model:
    """
    Wraps the openai gpt3 api
    """

    def __init__(self, engine_name):
        self.engine_name = engine_name

    def __call__(
        self,
        prompt,
        temperature=0.9,
        num_to_generate=1,
        max_tokens=64,
        stop_seq="Label:",
    ):
        completion = openai.Completion.create(
            engine=self.engine_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_seq,
            n=num_to_generate,
        )
        generations = [comp.text.strip() for comp in completion.choices]
        return generations


class GeneratorModel:
    """
    Wraps a LM for easy generation.
    """

    def __init__(self, model_size):
        if model_size == "tiny":
            model_name = "EleutherAI/gpt-neo-125M"
        elif model_size == "small":
            model_name = "gpt2-large"
        elif model_size == "medium":
            model_name = "EleutherAI/gpt-neo-1.3B"
        elif model_size == "xmedium":
            model_name = "EleutherAI/gpt-neo-2.7B"
        elif model_size == "large":
            model_name = "EleutherAI/gpt-j-6B"
        else:
            raise ValueError("unknown model size")
        model = AutoModelForCausalLM.from_pretrained(model_name, max_length=256).cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = model
        self.tokenizer = tokenizer

    def __call__(
        self,
        prompt,
        temperature=1.0,
        num_to_generate=1,
        max_tokens=64,
        stop_seq="Label:",
    ):
        input_ids = self.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.cuda()
        generated_ids = self.model.generate(
            input_ids,
            do_sample=True,
            num_return_sequences=num_to_generate,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        prompt_length = input_ids.shape[1]
        decoded_generations = self.tokenizer.batch_decode(
            generated_ids[:, prompt_length:], skip_special_tokens=True
        )
        generations = [gen.split(stop_seq)[0].strip() for gen in decoded_generations]
        return generations


def load_label_mapping(dataset):
    if dataset == "emotion":
        class2label = {
            0: "sadness", 
            1: "joy", 
            3: "anger", 
            4: "fear"
        }
    elif dataset == "trec10":
        class2label = {
            0: "description",
            1: "entity",
            3: "human",
            4: "number",
            5: "location",
        }
    elif dataset == "agnews":
        class2label = {
            0: "world", 
            1: "sports", 
            2: "business", 
            3: "sci/tech"
        }
    elif dataset == "tacred":
        labels_path = "data/tacred/0/labels_map.pkl"
        with open(labels_path, "rb") as f:
            name_to_label_mapping = pickle.load(f)
        class2label = {0: []}
        for name, label in name_to_label_mapping.items():
            if label == 0:
                class2label[0].append(name)
            else:
                class2label[label] = name
        # Process class2label
        def normalize(name):
            return name.replace("org:", "").replace("per:", "").replace("_", " ")

        for label, name in class2label.items():
            if isinstance(name, list):
                class2label[label] = [normalize(n) for n in name]
            else:
                class2label[label] = normalize(name)
    else:
        raise ValueError
    return class2label


def preprocess_tacred(sample, token_to_ner_mapping):
    text = sample["text"]
    subj_start_token = [
        token for token, ner in token_to_ner_mapping.items() if ner == "SUBJ_START"
    ][0]
    subj_end_token = [
        token for token, ner in token_to_ner_mapping.items() if ner == "SUBJ_END"
    ][0]
    obj_start_token = [
        token for token, ner in token_to_ner_mapping.items() if ner == "OBJ_START"
    ][0]
    obj_end_token = [
        token for token, ner in token_to_ner_mapping.items() if ner == "OBJ_END"
    ][0]
    text = text.replace(subj_start_token + " ", "[Subject: ")
    text = text.replace(" " + subj_end_token, "]")
    text = text.replace(obj_start_token + " ", "[Object: ")
    text = text.replace(" " + obj_end_token, "]")
    for token, ner in token_to_ner_mapping.items():
        if "SUBJ_" not in ner and "OBJ_" not in ner:
            ner = ner.split("=")[1].lower().replace("_", " ")
            text = text.replace(token, ner)
    return {"text": text}


def postprocess_tacred(generation, token_to_ner_mapping):
    def slice_assign(string, span, replacement):
        start_idx, end_idx = span
        return string[:start_idx] + replacement + string[end_idx:]

    new_mapping = {
        ner.split("=")[1].lower().replace("_", " "): token
        for token, ner in token_to_ner_mapping.items()
        if "SUBJ_" not in ner and "OBJ_" not in ner
    }
    subj_start_token = [
        token for token, ner in token_to_ner_mapping.items() if ner == "SUBJ_START"
    ][0]
    subj_end_token = [
        token for token, ner in token_to_ner_mapping.items() if ner == "SUBJ_END"
    ][0]
    obj_start_token = [
        token for token, ner in token_to_ner_mapping.items() if ner == "OBJ_START"
    ][0]
    obj_end_token = [
        token for token, ner in token_to_ner_mapping.items() if ner == "OBJ_END"
    ][0]
    if (
        generation.count("[Subject:") != 1
        or generation.count("[Object:") != 1
        or generation.count("]") != 2
    ):
        return None
    # Subject
    subj_start_idx = generation.index("[Subject:")
    if "]" not in generation[subj_start_idx:]:
        # didn't generate a closing bracket anywhere after the subject start
        return None
    subj_end_idx = generation.index("]", subj_start_idx)
    generated_ner = (
        generation[subj_start_idx + 9 : subj_end_idx].strip().replace(" ", "_")
    )
    if generated_ner not in new_mapping.keys():
        # didn't generate a real ner token
        return None
    subj_replacement = (
        subj_start_token + " " + new_mapping[generated_ner] + " " + subj_end_token
    )
    generation = slice_assign(
        generation, (subj_start_idx, subj_end_idx + 1), subj_replacement
    )
    # Object
    obj_start_idx = generation.index("[Object:")
    if "]" not in generation[obj_start_idx:]:
        # didn't generate a closing bracket anywhere after the object start
        return None
    obj_end_idx = generation.index("]", obj_start_idx)
    generated_ner = (
        generation[obj_start_idx + 8 : obj_end_idx].strip().replace(" ", "_")
    )
    if generated_ner not in new_mapping.keys():
        # didn't generate a real ner token
        return None
    obj_replacement = (
        obj_start_token + " " + new_mapping[generated_ner] + " " + obj_end_token
    )
    generation = slice_assign(
        generation, (obj_start_idx, obj_end_idx + 1), obj_replacement
    )
    return generation


def load_ner_tags(ner_tags_path):
    with open(ner_tags_path, "rb") as f:
        ner_to_token_mapping = pickle.load(f)
    token_to_ner_mapping = {tok: ner for ner, tok in ner_to_token_mapping.items()}
    return token_to_ner_mapping

def is_not_noise(label):
    disallowed_seqs = ["_", ".", "{", "}", "=", "[", "]", "(", ")", "$", "\\", "/", "<", ">", "|", "~", '"', "\t", "\n", "'", "-"]
    for ds in disallowed_seqs:
        if ds in label:
            return False
    for digit in range(10):
        if str(digit) in label:
            return False
    return True
