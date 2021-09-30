"""Script evaluates multimodal disambiguation using GT labels.
Expected JSON format:
[
    "dialog_id": <dialog_id>,
    "predictions": [
        {
            "turn_id": <turn_id>,
            "disambiguation_label": <bool>,
        }
        ...
    ]
    ...
]
Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json

import numpy as np


def evaluate_disambiguation(gt_labels, model_results):
    """Evaluates disambiguation using golden labels and model predictions.
    Args:
        gt_labels: Ground truth labels.
        model_results: Generated labels.
    """
    gt_label_pool = {ii["dialogue_idx"]: ii for ii in gt_labels["dialogue_data"]}

    predictions = []
    num_evaluations = 0
    for model_datum in model_results:
        dialog_id = model_datum["dialog_id"]
        for round_datum in model_datum["predictions"]:
            round_id = round_datum["turn_id"]
            predicted_label = round_datum["disambiguation_label"]
            gt_datum = gt_label_pool[dialog_id]["dialogue"][round_id]

            assert "disambiguation_label" in gt_datum, "Turn not to be evaluated!"
            gt_label = gt_datum["disambiguation_label"]
            predictions.append(gt_label == predicted_label)

    print(f"# Instances evaluated: {len(predictions)}")
    return np.mean(predictions), np.std(predictions) / np.sqrt(len(predictions))


def main(args):
    print("Reading: {}".format(args["data_json_path"]))
    with open(args["data_json_path"], "r") as file_id:
        gt_labels = json.load(file_id)
    print("Reading: {}".format(args["model_result_path"]))
    with open(args["model_result_path"], "r") as file_id:
        model_results = json.load(file_id)

    accuracy, accuracy_std_err = evaluate_disambiguation(gt_labels, model_results)
    print(f"Disambiguation Accuracy: {accuracy:.3f} +- {accuracy_std_err:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disambiguation Evaluation")
    parser.add_argument(
        "--data_json_path",
        default="../data/simmc2_dials_dstc10_devtest.json",
        help="Data with gold label for disambiguation",
    )
    parser.add_argument(
        "--model_result_path",
        default="../scripts/disambiguation_result.json",
        help="Disambiguation labels generated by the model",
    )
    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
