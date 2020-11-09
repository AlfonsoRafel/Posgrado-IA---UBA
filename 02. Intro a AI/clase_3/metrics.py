import numpy as np

class Metric(object):
    def __call__(self, target, prediction):
        return NotImplemented


class MSE(Metric):
    def __call__(self, target, prediction):
        n = target.size
        return np.sum((target - prediction) ** 2) / n
    
class Precision(Metric):
    def __call__(self, target, prediction):
        true_pos_mask = (prediction == 1) & (target == 1)
        true_pos = true_pos_mask.sum()
        false_pos_mask = (prediction == 1) & (target == 0)
        false_pos = false_pos_mask.sum()
        return true_pos / (true_pos + false_pos)

class Recall(Metric):
    def __call__(self, target, prediction):
        true_pos_mask = (prediction == 1) & (target == 1)
        true_pos = true_pos_mask.sum()
        false_neg_mask = (prediction == 0) & (target == 1)
        false_neg = false_neg_mask.sum()
        return true_pos / (true_pos + false_neg)

class Accuracy(Metric):
    def __call__(self, target, prediction):
        true_pos_mask = (prediction == 1) & (target == 1)
        true_pos = true_pos_mask.sum()
        true_neg_mask = (prediction == 0) & (target == 0)
        true_neg = true_neg_mask.sum()

        false_neg_mask = (prediction == 0) & (target == 1)
        false_neg = false_neg_mask.sum()
        false_pos_mask = (prediction == 1) & (target == 0)
        false_pos = false_pos_mask.sum()
        return (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

class F1_score(Metric):
    def __call__(self, target, prediction):
        true_pos_mask = (prediction == 1) & (target == 1)
        true_pos = true_pos_mask.sum()
        false_pos_mask = (prediction == 1) & (target == 0)
        false_pos = false_pos_mask.sum()
        false_neg_mask = (prediction == 0) & (target == 1)
        false_neg = false_neg_mask.sum()

        return true_pos / (true_pos + 0.5 * (false_pos + false_neg))
