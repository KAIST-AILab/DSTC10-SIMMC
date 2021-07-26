#! /usr/bin/env python
"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the LICENSE file in the
root directory of this source tree.

Reads SIMMC 2.0 dataset, creates train, devtest, dev formats for disambiguation.

Author(s): Satwik Kottur
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import copy
import json
import os


SPLITS = ["train", "dev", "devtest"]


def main(args):
    for split in SPLITS:
        read_path = args[f"simmc_{split}_json"]
        print(f"Reading: {read_path}")
        with open(read_path, "r") as file_id:
            dialogs = json.load(file_id)

        # Reformat into simple strings with positive and negative labels.
        # (dialog string, label)
        disambiguate_data = []
        for dialog_id, dialog_datum in enumerate(dialogs["dialogue_data"]):
            history = []
            for turn_datum in dialog_datum["dialogue"]:
                history.append(turn_datum["transcript"])

                label = turn_datum.get("disambiguation_label", None)
                if label is not None:
                    disambiguate_data.append((copy.deepcopy(history), label))
                history.append(turn_datum["system_transcript"])
        print(f"# instances [{split}]: {len(disambiguate_data)}")
        save_path = os.path.join(
            args["disambiguate_save_path"], f"simmc2_disambiguate_dstc10_{split}.json"
        )
        print(f"Saving: {save_path}")
        with open(save_path, "w") as file_id:
            json.dump(disambiguate_data, file_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simmc_train_json", default=None, help="Path to SIMMC train file"
    )
    parser.add_argument("--simmc_dev_json", default=None, help="Path to SIMMC dev file")
    parser.add_argument(
        "--simmc_devtest_json", default=None, help="Path to SIMMC devtest file"
    )
    parser.add_argument(
        "--disambiguate_save_path",
        required=True,
        help="Path to save SIMMC disambiguate JSONs",
    )

    try:
        parsed_args = vars(parser.parse_args())
    except (IOError) as msg:
        parser.error(str(msg))
    main(parsed_args)
