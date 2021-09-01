from typing import Any, Callable, Dict, Optional, List, Union

import nltk
import torch
import numpy as np

from torchmetrics import Metric


def normalize_sentence(sentence: str):
    return nltk.tokenize.word_tokenize(sentence.lower())


def rec_prec_f1(n_correct, n_true, n_pred) -> float:
    rec = n_correct / n_true if n_true.item() != 0 else 0
    prec = n_correct / n_pred if n_pred.item() != 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) != 0 else 0
    return rec, prec, f1


def d_f1(n_true, n_pred, n_correct) -> float:
    # 1/r + 1/p = 2/F1
    # dr / r^2 + dp / p^2 = 2dF1 /F1^2
    # dF1 = 1/2 F1^2 (dr/r^2 + dp/p^2)
    dr = b_stderr(n_true, n_correct)
    dp = b_stderr(n_pred, n_correct)

    r = n_correct / n_true
    p = n_correct / n_pred
    f1 = 2 * p * r / (p + r) if (p + r) != 0 else 0

    d_f1 = 0.5 * f1 ** 2 * (dr / r ** 2 + dp / p ** 2)
    return d_f1


def b_stderr(n_total, n_pos) -> float:
    return np.std(b_arr(n_total, n_pos)) / (n_total ** 0.5)


def b_arr(n_total, n_pos) -> float:
    out = np.zeros(int(n_total))
    out[: int(n_pos)] = 1.0
    return out


class DSTScore(Metric):
    def __init__(
        self,
        strict: bool=True,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.strict = strict

        self.add_state(
            "n_frames", torch.tensor(0., dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state(
            "n_true_acts", torch.tensor(0., dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state(
            "n_pred_acts", torch.tensor(0., dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state(
            "n_correct_acts", torch.tensor(0., dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state(
            "n_true_slots", torch.tensor(0., dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state(
            "n_pred_slots", torch.tensor(0., dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state(
            "n_correct_slots", torch.tensor(0., dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state(
            "n_true_request_slots", torch.tensor(0., dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state(
            "n_pred_request_slots", torch.tensor(0., dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state(
            "n_correct_request_slots", torch.tensor(0., dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state(
            "n_true_objects", torch.tensor(0., dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state(
            "n_pred_objects", torch.tensor(0., dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state(
            "n_correct_objects", torch.tensor(0., dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state(
            "n_correct_beliefs", torch.tensor(0., dtype=torch.float), dist_reduce_fx="sum"
        )

    def update(
        self,
        predictions: List[List[Dict[str, Any]]],
        ground_truths: List[List[Dict[str, Any]]]
    ):
        '''
            Code modified from evaluate_from_flat_list in baseline. Supports batching and individual sample computation.

            e.g. Each resulting sample has the following format:
                [
                    {
                        'act': 'INFORM:REFINE',
                        'slots': [
                            ['pattern', 'plain with stripes on side'],
                            ['customerReview', 'good'],
                            ['availableSizes', "['M', 'XL', 'XS']"],
                            ['type', 'sweater']
                        ],
                        'request_slots': [],
                        'objects': []
                    },
                    ...
                ]
        '''
        if isinstance(predictions, list) and len(predictions) == 0:
            pass
        else:
            if isinstance(predictions[0], dict):
                predictions = [predictions]
            if isinstance(ground_truths[0], dict):
                ground_truths = [ground_truths]
                
        for pred_turn, true_turn in zip(predictions, ground_truths):
            for frame_idx, true_frame in enumerate(true_turn):
                if frame_idx >= len(pred_turn):
                    pred_frame = dict()
                else:
                    pred_frame = pred_turn[frame_idx]
                
                self.n_frames += 1

                # Dialogue acts
                true_act = true_frame.get('act', None)
                pred_act = pred_frame.get('act', None)
                b_correct_act = (true_act == pred_act)

                self.n_correct_acts += b_correct_act
                self.n_true_acts += ("act" in true_frame)
                self.n_pred_acts += ("act" in pred_frame)

                # Slots
                true_frame_slots = {f"{k}={v}" for k, v in true_frame.get('slots', [])}
                pred_frame_slots = {f"{k}={v}" for k, v in pred_frame.get('slots', [])}

                self.n_true_slots += len(true_frame_slots)
                self.n_pred_slots += len(pred_frame_slots)
                if self.strict and not b_correct_act:
                    pass
                else:
                    self.n_correct_slots += len(true_frame_slots.intersection(pred_frame_slots))

                # Request slots
                true_frame_request_slots = {request for request in true_frame.get('request_slots', [])}
                pred_frame_request_slots = {request for request in pred_frame.get('request_slots', [])}

                self.n_true_request_slots += len(true_frame_request_slots)
                self.n_pred_request_slots += len(pred_frame_request_slots)
                if self.strict and not b_correct_act:
                    pass
                else:
                    self.n_correct_request_slots += len(
                        true_frame_request_slots.intersection(pred_frame_request_slots))

                # Objects
                true_frame_objects = {object_id for object_id in true_frame.get('objects', [])}
                pred_frame_objects = {object_id for object_id in pred_frame.get('objects', [])}

                self.n_true_objects += len(true_frame_objects)
                self.n_pred_objects += len(pred_frame_objects)

                if self.strict and not b_correct_act:
                    pass
                else:
                    self.n_correct_objects += len(true_frame_objects.intersection(pred_frame_objects))

                # Joint 
                self.n_correct_beliefs += (
                    b_correct_act
                    and (true_frame_slots == pred_frame_slots)
                    and (true_frame_request_slots == pred_frame_request_slots)
                    and (true_frame_objects == pred_frame_objects)
                )

    def compute(self) -> Dict:
        joint_accuracy = self.n_correct_beliefs / self.n_frames
        act_rec, act_prec, act_f1 = rec_prec_f1(
            n_correct=self.n_correct_acts,
            n_true=self.n_true_acts,
            n_pred=self.n_pred_acts
        )
        slot_rec, slot_prec, slot_f1 = rec_prec_f1(
            n_correct=self.n_correct_slots,
            n_true=self.n_true_slots,
            n_pred=self.n_pred_slots
        )
        request_slot_rec, request_slot_prec, request_slot_f1 = rec_prec_f1(
            n_correct=self.n_correct_request_slots,
            n_true=self.n_true_request_slots,
            n_pred=self.n_pred_request_slots
        )
        object_rec, object_prec, object_f1 = rec_prec_f1(
            n_correct=self.n_correct_objects,
            n_true=self.n_true_objects,
            n_pred=self.n_pred_objects
        )
        # Errors
        act_f1_stderr = d_f1(
            n_true=self.n_true_acts,
            n_pred=self.n_pred_acts,
            n_correct=self.n_correct_acts
        )
        slot_f1_stderr = d_f1(
            n_true=self.n_true_slots,
            n_pred=self.n_pred_slots,
            n_correct=self.n_correct_slots
        )
        request_slot_f1_stderr = d_f1(
            n_true=self.n_true_request_slots,
            n_pred=self.n_pred_request_slots,
            n_correct=self.n_correct_request_slots
        )
        object_f1_stderr = d_f1(
            n_true=self.n_true_objects,
            n_pred=self.n_pred_objects,
            n_correct=self.n_correct_objects
        )
        return {
            "joint_accuracy": joint_accuracy,
            "act_rec": act_rec,
            "act_prec": act_prec,
            "act_f1": act_f1,
            "act_f1_stderr": act_f1_stderr,
            "slot_rec": slot_rec,
            "slot_prec": slot_prec,
            "slot_f1": slot_f1,
            "slot_f1_stderr": slot_f1_stderr,
            "request_slot_rec": request_slot_rec,
            "request_slot_prec": request_slot_prec,
            "request_slot_f1": request_slot_f1,
            "request_slot_f1_stderr": request_slot_f1_stderr,
            "object_rec": object_rec,
            "object_prec": object_prec,
            "object_f1": object_f1,
            "object_f1_stderr": object_f1_stderr,
        }

    @property
    def is_differentiable(self) -> bool:
        return False


class DisambAccuracy(Metric):
    '''
        Metric tracker in torchmetrics format for disambiguation accuracy.
    '''
    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state(
            "n_correct", torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state(
            "n_total", torch.tensor(0, dtype=torch.float),
            dist_reduce_fx="sum"
        )

    def update(
        self,
        predicted: Union[List[List[Dict]], List[Dict]],
        labels: Union[List[int], int]
    ):
        if isinstance(predicted, list) and len(predicted) == 0:
            self.n_total += 1
            return
        else:
            if isinstance(predicted[0], dict):
                predicted = [predicted]
                labels = [labels]
        
        for pred_turn, label in zip(predicted, labels):
            for pred_frame in pred_turn:
                if label == -1:
                    pass
                else:
                    self.n_total += 1
                    pred = 1 if (pred_frame.get('act', None) == "INFORM:DISAMBIGUATE") else 0
                    self.n_correct += (pred == label)

    def compute(self):
        acc = self.n_correct / self.n_total if self.n_total.item() != 0 else 0
        return acc  


class BLEUScore(Metric):
    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.smoothing = nltk.translate.bleu_score.SmoothingFunction()

        self.add_state(
            "bleu", torch.tensor(0, dtype=torch.float).view(1), dist_reduce_fx="cat"
        )


    def update(
        self,
        predicted: Union[str, List[str]],
        labels: Union[str, List[str]]
    ):
        if isinstance(predicted, tuple):
            predicted = [predicted]
        if isinstance(labels, tuple):
            labels = [labels]
        device = self.bleu.device

        for pred, true in zip(predicted, labels):
            self.bleu = torch.cat([
                self.bleu, 
                torch.tensor(
                    nltk.translate.bleu_score.sentence_bleu(
                        [normalize_sentence(true[1])],
                        normalize_sentence(pred[1]),
                        smoothing_function=self.smoothing.method7,
                    ), device=device
                ).view(1)
            ])

    def compute(self):
        return \
            {
                "bleu_mean": self.bleu.mean(),
                "bleu_std": self.bleu.std() / (len(self.bleu) ** 0.5)
            }