import json
import asyncio
import pandas as pd
import string
import re
from typing import List, Tuple, Callable, Dict, Any, Set, Union
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm.asyncio import tqdm_asyncio

from examples.ags.benchmark.utils import generate_random_indices

global cost
cost = 0

def _remove_articles(text: str) -> str:
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)


def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


EXCLUDE = set(string.punctuation)


def _remove_punc(text: str) -> str:
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text


def _lower(text: str) -> str:
    return text.lower()


def _tokenize(text: str) -> List[str]:
    return re.split(" |-", text)


def _normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    parts = [
        _white_space_fix(_remove_articles(_normalize_number(_remove_punc(_lower(token)))))
        for token in _tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    normalized = " ".join(parts).strip()
    return normalized


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    else:
        return text


def _answer_to_bags(
    answer: Union[str, List[str], Tuple[str, ...]]
) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if not (precision == 0.0 and recall == 0.0)
        else 0.0
    )
    return f1


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def get_drop_metrics(
    predicted: Union[str, List[str], Tuple[str, ...]], gold: Union[str, List[str], Tuple[str, ...]]
) -> Tuple[float, float]:
    """
    Takes a predicted answer and a gold answer (that are both either a string or a list of
    strings), and returns exact match and the DROP F1 metric for the prediction.  If you are
    writing a script for evaluating objects in memory (say, the output of predictions during
    validation, or while training), this is the function you want to call, after using
    :func:`answer_json_to_strings` when reading the gold answer from the released data file.
    """
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)

    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        exact_match = 1.0
    else:
        exact_match = 0.0

    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)

    return exact_match, f1


def answer_json_to_strings(answer: Dict[str, Any]) -> Tuple[Tuple[str, ...], str]:
    """
    Takes an answer JSON blob from the DROP data release and converts it into strings used for
    evaluation.
    """
    if "number" in answer and answer["number"]:
        return tuple([str(answer["number"])]), "number"
    elif "spans" in answer and answer["spans"]:
        return tuple(answer["spans"]), "span" if len(answer["spans"]) == 1 else "spans"
    elif "date" in answer:
        return (
            tuple(
                [
                    "{0} {1} {2}".format(
                        answer["date"]["day"], answer["date"]["month"], answer["date"]["year"]
                    ).strip()
                ]
            ),
            "date",
        )
    else:
        raise ValueError(
            f"Answer type not found, should be one of number, spans or date at: {json.dumps(answer)}"
        )


def answer_json_to_string(answer_json):
    return json.dumps(answer_json_to_strings(answer_json))


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def fuzzy_match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)

    if s1 == "" or s2 == "":
        return s1 == s2

    return s1 in s2 or s2 in s1


def drop_metric(sample: str, reference: list[str]) -> Tuple[float, float]:
    em_scores = []
    f1_scores = []
    for answer in reference:
        if answer.strip() != "":
            em, f1 = get_drop_metrics(sample, answer)
            em_scores.append(em)
            f1_scores.append(f1)
    return (max(em_scores), max(f1_scores))

async def evaluate_problem(inputs: str, answers: List[Dict[str, Any]], graph: Callable) -> Tuple[str, str, float]:

    max_retries = 5
    retries = 0

    while retries < max_retries:
        try:
            global cost
            prediction, cost = await graph(inputs)

            f1_scores = []
            best_answer = None

            for answer in answers:
                if answer.strip() != "":
                    _, f1_score = get_drop_metrics(prediction, answer)
                    f1_scores.append(f1_score)

            max_score = max(f1_scores)

            # matches = [
            #         fuzzy_match(prediction, answer)
            #         for answer in answers
            # ]

            # score = True in matches

            score = max_score

            break

        except Exception as e:
            retries += 1
            print(f"Error generating prediction: {e}. Retrying... ({retries}/{max_retries})")

            if retries == max_retries:
                print("Maximum retries reached. Skipping this sample.")
                prediction = None
                score = 0.0
                break

    return prediction, score

async def evaluate_all_questions(annotations: List[Tuple[str, Dict[str, Any]]], graph: Callable, max_concurrent_tasks: int = 50) -> List[List[Any]]:
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    results = []

    async def sem_evaluate(annotation: Dict[str, Any]):
        async with semaphore:
            inputs = annotation["context"]
            answers = annotation["targets"]
            prediction, score = await evaluate_problem(inputs, answers, graph)
            results.append([annotation["id"], prediction, answers, score])

    tasks = [sem_evaluate(annotation) for annotation in annotations]
    await tqdm_asyncio.gather(*tasks, desc="Evaluating DROP passages", total=len(annotations))

    return results

def save_results_to_csv(results: List[List[Any]], path: str) -> float:
    df = pd.DataFrame(results, columns=["id", "prediction", "answers", "score"])
    average_score = df["score"].mean()

    output_file = f"{path}/{average_score:.5f}.csv"
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    return average_score

# -- From ADAS --

def load_drop(file_path, samples, test=False, total_length=928):
    import gzip
    with gzip.open(file_path, "rb") as file:
        data = [json.loads(line) for line in file]

    random_indices = generate_random_indices(len(data), total_length, False)
    random_indices = random_indices[:samples] if not test else random_indices[samples:]
    examples = [data[i] for i in random_indices]

    for example in examples:
        example["targets"] = example["ref_text"].split("|")    

    return examples

async def drop_evaluation(graph: Callable, file_path: str, samples: int, path: str, test=False) -> float:
    # data = load_data(file_path, samples, test=test)
    data = load_drop(file_path, samples, test=test)
    results = await evaluate_all_questions(data, graph, max_concurrent_tasks=8)
    average_score = save_results_to_csv(results, path=path)
    print(f"Average score on DROP dataset: {average_score:.5f}")
    global cost
    print(f"Total cost: {cost: .5f}")
    print(f"Cost per sample: {(cost / len(data)):.9f}")
    return average_score
