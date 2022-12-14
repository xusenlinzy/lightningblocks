from typing import List

import numpy as np

from .utils import _prf_divide, _warn_prf


def _precision_recall_fscore(
    pred_sum,
    tp_sum,
    true_sum,
    warn_for=('precision', 'recall', 'f-score'),
    beta: float = 1.0,
    zero_division: str = 'warn',
):

    if beta < 0:
        raise ValueError('beta should be >=0 in the F-beta score')

    beta2 = beta ** 2
    precision = _prf_divide(numerator=tp_sum, denominator=pred_sum, metric='precision', modifier='predicted',
                           warn_for=warn_for, zero_division=zero_division)

    recall = _prf_divide(numerator=tp_sum, denominator=true_sum, metric='recall', modifier='true',
                         warn_for=warn_for, zero_division=zero_division)

    if zero_division == 'warn' and ('f-score',) == warn_for and (pred_sum[true_sum == 0] == 0).any():
        _warn_prf('true nor predicted', 'F-score is', len(true_sum))
    if np.isposinf(beta):
        f_score = recall
    else:
        denom = beta2 * precision + recall
        denom[denom == 0.0] = 1
        f_score = (1 + beta2) * precision * recall / denom

    precision = np.average(precision)
    recall = np.average(recall)
    f_score = np.average(f_score)
    return precision, recall, f_score


def extract_tp_actual_correct(y_true: List[set], y_pred: List[set]):
    entities_true = set()
    entities_pred = set()
    for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
        for d in y_t:
            entities_true.add((i, d))
        for d in y_p:
            entities_pred.add((i, d))

    tp_sum = np.array([len(entities_true & entities_pred)], dtype=np.int32)
    pred_sum = np.array([len(entities_pred)], dtype=np.int32)
    true_sum = np.array([len(entities_true)], dtype=np.int32)
    return pred_sum, tp_sum, true_sum
