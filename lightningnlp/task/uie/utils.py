import json
import math
import random
import re

import numpy as np

from lightningnlp.callbacks import Logger, tqdm

logger = Logger("UIE")


def get_span(start_ids, end_ids, with_prob=False):
    """
    Get spn set from position start and end list.
    Args:
        start_ids (L.ist[int]/List[tuple]): The start index list.
        end_ids (List[int]/List[tuple]): The end index list.
        with_prob (bool): If True, each element for start_ids and end_ids is a tuple alike: (index, probability).
    Returns:
        set: The spn set without overlapping, every id can only be used once .
    """
    if with_prob:
        start_ids = sorted(start_ids, key=lambda x: x[0])
        end_ids = sorted(end_ids, key=lambda x: x[0])
    else:
        start_ids = sorted(start_ids)
        end_ids = sorted(end_ids)

    start_pointer = 0
    end_pointer = 0
    len_start = len(start_ids)
    len_end = len(end_ids)
    couple_dict = {}
    while start_pointer < len_start and end_pointer < len_end:
        if with_prob:
            start_id = start_ids[start_pointer][0]
            end_id = end_ids[end_pointer][0]
        else:
            start_id = start_ids[start_pointer]
            end_id = end_ids[end_pointer]

        if start_id == end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            end_pointer += 1
            continue
        if start_id < end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            continue
        if start_id > end_id:
            end_pointer += 1
            continue
    result = [(couple_dict[end], end) for end in couple_dict]
    return set(result)


def get_bool_ids_greater_than(probs, limit=0.5, return_prob=False):
    """
    Get idx of the last dimension in probability arrays, which is greater than a limitation.
    Args:
        probs (List[List[float]]): The input probability arrays.
        limit (float): The limitation for probability.
        return_prob (bool): Whether to return the probability
    Returns:
        List[List[int]]: The index of the last dimension meet the conditions.
    """
    probs = np.array(probs)
    dim_len = len(probs.shape)
    result = []
    if dim_len > 1:
        result.extend(get_bool_ids_greater_than(p, limit, return_prob) for p in probs)
    else:
        for i, p in enumerate(probs):
            if p > limit:
                if return_prob:
                    result.append((i, p))
                else:
                    result.append(i)
    return result


def unify_prompt_name(prompt):
    # The classification labels are shuffled during finetuning, so they need
    # to be unified during evaluation.
    if re.search(r'\[.*?\]$', prompt):
        prompt_prefix = prompt[:prompt.find("[", 1)]
        cls_options = re.search(r'\[.*?\]$', prompt).group()[1:-1].split(",")
        cls_options = sorted(list(set(cls_options)))
        cls_options = ",".join(cls_options)
        prompt = f"{prompt_prefix}[{cls_options}]"
        return prompt
    return prompt


def get_id_and_prob(spans, offset_map):
    prompt_length = 0
    for i in range(1, len(offset_map)):
        if offset_map[i] != [0, 0]:
            prompt_length += 1
        else:
            break

    for i in range(1, prompt_length + 1):
        offset_map[i][0] -= (prompt_length + 1)
        offset_map[i][1] -= (prompt_length + 1)

    sentence_id = []
    prob = []
    for start, end in spans:
        prob.append(start[1] * end[1])
        sentence_id.append(
            (offset_map[start[0]][0], offset_map[end[0]][1]))
    return sentence_id, prob


def get_relation_type_dict(relation_data, schema_lang="ch"):

    def compare(a, b, schema_lang="ch"):
        if schema_lang == "ch":
            a = a[::-1]
            b = b[::-1]

        res = ''
        for i in range(min(len(a), len(b))):
            if a[i] == b[i]:
                res += a[i]
            else:
                break
        if res == "":
            return res
        if schema_lang == "ch" and res[::-1][0] == "的":
            return res[::-1][1:]
        elif schema_lang == "en" and res[-3:] == " of":
            return res[:-3]
        return ""

    relation_type_dict = {}
    added_list = []
    for i in range(len(relation_data)):
        added = False
        if relation_data[i][0] not in added_list:
            for j in range(i + 1, len(relation_data)):
                match = compare(relation_data[i][0], relation_data[j][0], schema_lang=schema_lang)
                if match != "":
                    match = unify_prompt_name(match)
                    if relation_data[i][0] not in added_list:
                        added_list.append(relation_data[i][0])
                        relation_type_dict.setdefault(match, []).append(relation_data[i][1])
                    added_list.append(relation_data[j][0])
                    relation_type_dict.setdefault(match, []).append(relation_data[j][1])
                    added = True
            if not added:
                added_list.append(relation_data[i][0])
                if schema_lang == "ch":
                    suffix = relation_data[i][0].rsplit("的", 1)[1]
                    suffix = unify_prompt_name(suffix)
                    relation_type = suffix
                else:
                    prefix = relation_data[i][0].split(" of ", 1)[0]
                    prefix = unify_prompt_name(prefix)
                    relation_type = prefix
                relation_type_dict.setdefault(relation_type, []).append(relation_data[i][1])
    return relation_type_dict


def add_entity_negative_example(examples, texts, prompts, label_set, negative_ratio):
    negative_examples = []
    positive_examples = []
    with tqdm(total=len(prompts)) as pbar:
        for i, prompt in enumerate(prompts):
            redundants = sorted(set(label_set) ^ set(prompt))
            num_positive = len(examples[i])
            if num_positive != 0:
                actual_ratio = math.ceil(len(redundants) / num_positive)
            else:
                # Set num_positive to 1 for text without positive example
                num_positive, actual_ratio = 1, 0

            if actual_ratio <= negative_ratio or negative_ratio == -1:
                idxs = list(range(len(redundants)))
            else:
                idxs = random.sample(range(len(redundants)), negative_ratio * num_positive)

            for idx in idxs:
                negative_result = {
                    "content": texts[i],
                    "result_list": [],
                    "prompt": redundants[idx]
                }
                negative_examples.append(negative_result)
            positive_examples.extend(examples[i])
            pbar.update(1)
    return positive_examples, negative_examples


def add_relation_negative_example(redundants, text, num_positive, ratio):
    added_example = []
    rest_example = []

    if num_positive != 0:
        actual_ratio = math.ceil(len(redundants) / num_positive)
    else:
        # Set num_positive to 1 for text without positive example
        num_positive, actual_ratio = 1, 0

    all_idxs = list(range(len(redundants)))
    if actual_ratio <= ratio or ratio == -1:
        idxs = all_idxs
        rest_idxs = []
    else:
        idxs = random.sample(range(len(redundants)), ratio * num_positive)
        rest_idxs = list(set(all_idxs) ^ set(idxs))

    for idx in idxs:
        negative_result = {
            "content": text,
            "result_list": [],
            "prompt": redundants[idx]
        }
        added_example.append(negative_result)

    for rest_idx in rest_idxs:
        negative_result = {
            "content": text,
            "result_list": [],
            "prompt": redundants[rest_idx]
        }
        rest_example.append(negative_result)

    return added_example, rest_example


def add_full_negative_example(examples,
                              texts,
                              relation_prompts,
                              predicate_set,
                              subject_goldens,
                              schema_lang="ch"):
    with tqdm(total=len(relation_prompts)) as pbar:
        for i, relation_prompt in enumerate(relation_prompts):
            negative_sample = []
            for subject in subject_goldens[i]:
                for predicate in predicate_set:
                    # The relation prompt is constructed as follows:
                    # subject + "的" + predicate -> Chinese
                    # predicate + " of " + subject -> English
                    if schema_lang == "ch":
                        prompt = f"{subject}的{predicate}"
                    else:
                        prompt = f"{predicate} of {subject}"
                    if prompt not in relation_prompt:
                        negative_result = {
                            "content": texts[i],
                            "result_list": [],
                            "prompt": prompt
                        }
                        negative_sample.append(negative_result)
            examples[i].extend(negative_sample)
            pbar.update(1)
    return examples


def generate_cls_example(text, labels, prompt_prefix, options):
    random.shuffle(options)
    cls_options = ",".join(options)
    prompt = f"{prompt_prefix}[{cls_options}]"

    result_list = []
    example = {"content": text, "result_list": result_list, "prompt": prompt}
    for label in labels:
        start = prompt.rfind(label) - len(prompt) - 1
        end = start + len(label)
        result = {"text": label, "start": start, "end": end}
        example["result_list"].append(result)
    return example


def convert_cls_examples(raw_examples,
                         prompt_prefix="情感倾向",
                         options=["正向", "负向"]):
    """
    Convert labeled data export from doccano for classification task.
    """
    examples = []
    logger.info("Converting doccano data...")
    with tqdm(total=len(raw_examples)) as pbar:
        for line in raw_examples:
            items = json.loads(line)
            # Compatible with doccano >= 1.6.2
            if "data" in items.keys():
                text, labels = items["data"], items["label"]
            else:
                text, labels = items["text"], items["label"]
            example = generate_cls_example(text, labels, prompt_prefix, options)
            examples.append(example)
    return examples


def convert_ext_examples(raw_examples,
                         negative_ratio,
                         prompt_prefix="情感倾向",
                         options=["正向", "负向"],
                         separator="##",
                         is_train=True,
                         schema_lang="ch"):
    """
    Convert labeled data export from doccano for extraction and aspect-level classification task.
    """

    def _sep_cls_label(label, separator):
        label_list = label.split(separator)
        if len(label_list) == 1:
            return label_list[0], None
        return label_list[0], label_list[1:]

    texts = []
    entity_examples = []
    relation_examples = []
    entity_cls_examples = []
    entity_prompts = []
    relation_prompts = []
    entity_label_set = []
    entity_name_set = []
    predicate_set = []
    subject_goldens = []
    inverse_relation_list = []
    predicate_list = []

    logger.info(f"Converting doccano data...")
    with tqdm(total=len(raw_examples)) as pbar:
        for line in raw_examples:
            items = json.loads(line)
            entity_id = 0
            if "data" in items.keys():
                relation_mode = False
                if isinstance(items["label"],
                              dict) and "entities" in items["label"].keys():
                    relation_mode = True
                text = items["data"]
                entities = []
                relations = []
                if not relation_mode:
                    # Export file in JSONL format which doccano < 1.7.0
                    # e.g. {"data": "", "label": [ [0, 2, "ORG"], ... ]}
                    for item in items["label"]:
                        entity = {
                            "id": entity_id,
                            "start_offset": item[0],
                            "end_offset": item[1],
                            "label": item[2]
                        }
                        entities.append(entity)
                        entity_id += 1
                else:
                    # Export file in JSONL format for relation labeling task which doccano < 1.7.0
                    # e.g. {"data": "", "label": {"relations": [ {"id": 0, "start_offset": 0, "end_offset": 6, "label": "ORG"}, ... ], "entities": [ {"id": 0, "from_id": 0, "to_id": 1, "type": "foundedAt"}, ... ]}}
                    entities.extend(
                        [entity for entity in items["label"]["entities"]])
                    if "relations" in items["label"].keys():
                        relations.extend([
                            relation for relation in items["label"]["relations"]
                        ])
            else:
                # Export file in JSONL format which doccano >= 1.7.0
                # e.g. {"text": "", "label": [ [0, 2, "ORG"], ... ]}
                if "label" in items.keys():
                    text = items["text"]
                    entities = []
                    for item in items["label"]:
                        entity = {
                            "id": entity_id,
                            "start_offset": item[0],
                            "end_offset": item[1],
                            "label": item[2]
                        }
                        entities.append(entity)
                        entity_id += 1
                    relations = []
                else:
                    # Export file in JSONL (relation) format
                    # e.g. {"text": "", "relations": [ {"id": 0, "start_offset": 0, "end_offset": 6, "label": "ORG"}, ... ], "entities": [ {"id": 0, "from_id": 0, "to_id": 1, "type": "foundedAt"}, ... ]}
                    text, relations, entities = items["text"], items[
                        "relations"], items["entities"]
            texts.append(text)

            entity_example = []
            entity_prompt = []
            entity_example_map = {}
            entity_map = {}  # id to entity name
            for entity in entities:
                entity_name = text[entity["start_offset"]:entity["end_offset"]]
                entity_map[entity["id"]] = {
                    "name": entity_name,
                    "start": entity["start_offset"],
                    "end": entity["end_offset"]
                }

                entity_label, entity_cls_label = _sep_cls_label(
                    entity["label"], separator)

                # Define the prompt prefix for entity-level classification
                # xxx + "的" + 情感倾向 -> Chinese
                # Sentiment classification + " of " + xxx -> English
                if schema_lang == "ch":
                    entity_cls_prompt_prefix = entity_name + "的" + prompt_prefix
                else:
                    entity_cls_prompt_prefix = prompt_prefix + " of " + entity_name
                if entity_cls_label is not None:
                    entity_cls_example = generate_cls_example(
                        text, entity_cls_label, entity_cls_prompt_prefix,
                        options)

                    entity_cls_examples.append(entity_cls_example)

                result = {
                    "text": entity_name,
                    "start": entity["start_offset"],
                    "end": entity["end_offset"]
                }
                if entity_label not in entity_example_map.keys():
                    entity_example_map[entity_label] = {
                        "content": text,
                        "result_list": [result],
                        "prompt": entity_label
                    }
                else:
                    entity_example_map[entity_label]["result_list"].append(
                        result)

                if entity_label not in entity_label_set:
                    entity_label_set.append(entity_label)
                if entity_name not in entity_name_set:
                    entity_name_set.append(entity_name)
                entity_prompt.append(entity_label)

            for v in entity_example_map.values():
                entity_example.append(v)

            entity_examples.append(entity_example)
            entity_prompts.append(entity_prompt)

            subject_golden = []  # Golden entity inputs
            relation_example = []
            relation_prompt = []
            relation_example_map = {}
            inverse_relation = []
            predicates = []
            for relation in relations:
                predicate = relation["type"]
                subject_id = relation["from_id"]
                object_id = relation["to_id"]
                # The relation prompt is constructed as follows:
                # subject + "的" + predicate -> Chinese
                # predicate + " of " + subject -> English
                if schema_lang == "ch":
                    prompt = entity_map[subject_id]["name"] + "的" + predicate
                    inverse_negative = entity_map[object_id][
                        "name"] + "的" + predicate
                else:
                    prompt = predicate + " of " + entity_map[subject_id]["name"]
                    inverse_negative = predicate + " of " + entity_map[
                        object_id]["name"]

                if entity_map[subject_id]["name"] not in subject_golden:
                    subject_golden.append(entity_map[subject_id]["name"])
                result = {
                    "text": entity_map[object_id]["name"],
                    "start": entity_map[object_id]["start"],
                    "end": entity_map[object_id]["end"]
                }

                inverse_relation.append(inverse_negative)
                predicates.append(predicate)

                if prompt not in relation_example_map.keys():
                    relation_example_map[prompt] = {
                        "content": text,
                        "result_list": [result],
                        "prompt": prompt
                    }
                else:
                    relation_example_map[prompt]["result_list"].append(result)

                if predicate not in predicate_set:
                    predicate_set.append(predicate)
                relation_prompt.append(prompt)

            for v in relation_example_map.values():
                relation_example.append(v)

            relation_examples.append(relation_example)
            relation_prompts.append(relation_prompt)
            subject_goldens.append(subject_golden)
            inverse_relation_list.append(inverse_relation)
            predicate_list.append(predicates)
            pbar.update(1)

    logger.info(f"Adding negative samples for first stage prompt...")
    positive_examples, negative_examples = add_entity_negative_example(
        entity_examples, texts, entity_prompts, entity_label_set,
        negative_ratio)
    if len(positive_examples) == 0:
        all_entity_examples = []
    else:
        all_entity_examples = positive_examples + negative_examples

    all_relation_examples = []
    if len(predicate_set) != 0:
        logger.info(f"Adding negative samples for second stage prompt...")
        if is_train:

            positive_examples = []
            negative_examples = []
            per_n_ratio = negative_ratio // 3

            with tqdm(total=len(texts)) as pbar:
                for i, text in enumerate(texts):
                    negative_example = []
                    collects = []
                    num_positive = len(relation_examples[i])

                    # 1. inverse_relation_list
                    redundants1 = inverse_relation_list[i]

                    # 2. entity_name_set ^ subject_goldens[i]
                    redundants2 = []
                    if len(predicate_list[i]) != 0:
                        nonentity_list = list(
                            set(entity_name_set) ^ set(subject_goldens[i]))
                        nonentity_list.sort()

                        if schema_lang == "ch":
                            redundants2 = [
                                nonentity + "的" +
                                predicate_list[i][random.randrange(
                                    len(predicate_list[i]))]
                                for nonentity in nonentity_list
                            ]
                        else:
                            redundants2 = [
                                predicate_list[i][random.randrange(
                                    len(predicate_list[i]))] + " of " +
                                nonentity for nonentity in nonentity_list
                            ]

                    # 3. entity_label_set ^ entity_prompts[i]
                    redundants3 = []
                    if len(subject_goldens[i]) != 0:
                        non_ent_label_list = list(
                            set(entity_label_set) ^ set(entity_prompts[i]))
                        non_ent_label_list.sort()

                        if schema_lang == "ch":
                            redundants3 = [
                                subject_goldens[i][random.randrange(
                                    len(subject_goldens[i]))] + "的" +
                                non_ent_label
                                for non_ent_label in non_ent_label_list
                            ]
                        else:
                            redundants3 = [
                                non_ent_label + " of " +
                                subject_goldens[i][random.randrange(
                                    len(subject_goldens[i]))]
                                for non_ent_label in non_ent_label_list
                            ]

                    redundants_list = [redundants1, redundants2, redundants3]

                    for redundants in redundants_list:
                        added, rest = add_relation_negative_example(
                            redundants,
                            texts[i],
                            num_positive,
                            per_n_ratio,
                        )
                        negative_example.extend(added)
                        collects.extend(rest)

                    num_sup = num_positive * negative_ratio - len(
                        negative_example)
                    if num_sup > 0 and collects:
                        if num_sup > len(collects):
                            idxs = [k for k in range(len(collects))]
                        else:
                            idxs = random.sample(range(0, len(collects)),
                                                 num_sup)
                        for idx in idxs:
                            negative_example.append(collects[idx])

                    positive_examples.extend(relation_examples[i])
                    negative_examples.extend(negative_example)
                    pbar.update(1)
            all_relation_examples = positive_examples + negative_examples
        else:
            relation_examples = add_full_negative_example(
                relation_examples,
                texts,
                relation_prompts,
                predicate_set,
                subject_goldens,
                schema_lang=schema_lang)
            all_relation_examples = [
                r for relation_example in relation_examples
                for r in relation_example
            ]
    return all_entity_examples, all_relation_examples, entity_cls_examples
