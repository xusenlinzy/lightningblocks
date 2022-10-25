from lightningnlp.metrics.base import Metric
from lightningnlp.metrics.extraction.precision_recall_fscore import extract_tp_actual_correct
from lightningnlp.metrics.extraction.precision_recall_fscore import _precision_recall_fscore


class ExtractionScore(Metric):

    def __init__(self, average="micro"):
        self.average = average
        self.reset()

    def update(self, y_true, y_pred):
        pred_sum, tp_sum, true_sum = extract_tp_actual_correct(y_true, y_pred)
        self.pred_sum += pred_sum
        self.tp_sum += tp_sum
        self.true_sum += true_sum

    def value(self):
        p, r, f = _precision_recall_fscore(self.pred_sum, self.tp_sum, self.true_sum)
        return p.item(), r.item(), f.item()

    def name(self):
        return "extraction_score"

    def reset(self):
        self.pred_sum = 0
        self.tp_sum = 0
        self.true_sum = 0


if __name__ == "__main__":
    metric = ExtractionScore()
    p = [{("a", 1), ("a", 2), ("b", 3)}, {("a", 4), ("c", 5)}]
    t = [{("a", 1), ("a", 4), ("b", 3)}, {("a", 2), ("c", 5)}]
    metric.update(t, p)
    print(metric.value())
