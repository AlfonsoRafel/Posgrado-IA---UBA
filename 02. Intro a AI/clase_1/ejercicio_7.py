import numpy as np

class BaseMetric(object):
    def __init__(self, **kwargs):
        self.prediction = kwargs.pop('prediction',None)
        self.truth = kwargs.pop('truth',None)
        self.predicted_rank = kwargs.pop('predicted_rank',None)
        self.truth_relevance = kwargs.pop('truth_relevance',None)
        self.query_ids = kwargs.pop('query_ids',None)
        self.k = kwargs.pop('k',None)
        print("Elemento creado")


class Precision(BaseMetric):
    def __call__(self):
        true_pos_mask = (self.prediction == 1) & (self.truth == 1)
        true_pos = true_pos_mask.sum()
        false_pos_mask = (self.prediction == 1) & (self.truth == 0)
        false_pos = false_pos_mask.sum()
        return true_pos / (true_pos + false_pos)
class Recall(BaseMetric):
    def __call__(self):
        true_pos_mask = (self.prediction == 1) & (self.truth == 1)
        true_pos = true_pos_mask.sum()
        false_neg_mask = (self.prediction == 0) & (self.truth == 1)
        false_neg = false_neg_mask.sum()
        return true_pos / (true_pos + false_neg)
class Accuracy(BaseMetric):
    def __call__(self):
        true_pos_mask = (self.prediction == 1) & (self.truth == 1)
        true_pos = true_pos_mask.sum()
        true_neg_mask = (self.prediction == 0) & (self.truth == 0)
        true_neg = true_neg_mask.sum()

        false_neg_mask = (self.prediction == 0) & (self.truth == 1)
        false_neg = false_neg_mask.sum()
        false_pos_mask = (self.prediction == 1) & (self.truth == 0)
        false_pos = false_pos_mask.sum()
        return (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
class QueryMeanPrecision(BaseMetric):
    def __call__(self):
        # Todas las entradas deben ser np.array para que la mascara no sea solo un bool
        true_relevance_mask = (self.truth_relevance == 1)
        filtered_query_id = self.query_ids[true_relevance_mask]
        filtered_true_relevance_count = np.bincount(filtered_query_id)
        # complete the count of queries with zeros in queries without true relevant documents
        unique_query_ids = np.unique(self.query_ids)
        non_zero_count_idxs = np.where(filtered_true_relevance_count > 0)
        true_relevance_count = np.zeros(unique_query_ids.max() + 1)
        true_relevance_count[non_zero_count_idxs] = filtered_true_relevance_count[non_zero_count_idxs]
        true_relevance_count_by_query = true_relevance_count[unique_query_ids]
        fetched_documents_count = np.bincount(self.query_ids)[unique_query_ids]
        precision_by_query = true_relevance_count_by_query/fetched_documents_count
        return np.mean(precision_by_query)
class QueryMeanPrecisionAtK(BaseMetric):
    def __call__(self):
        # Todas las entradas deben ser np.array para que la mascara no sea solo un bool
        true_relevance_mask = (self.truth_relevance == 1)
        filtered_query_id = self.query_ids[true_relevance_mask]
        filtered_true_relevance_count = np.bincount(filtered_query_id)
        # complete the count of queries with zeros in queries without true relevant documents
        unique_query_ids = np.unique(self.query_ids)
        non_zero_count_idxs = np.where(filtered_true_relevance_count > 0)
        true_relevance_count = np.zeros(unique_query_ids.max() + 1)
        true_relevance_count[non_zero_count_idxs] = filtered_true_relevance_count[non_zero_count_idxs]
        true_relevance_count_by_query = true_relevance_count[unique_query_ids]
        fetched_documents_count = np.bincount(self.query_ids)[unique_query_ids]
        precision_by_query = true_relevance_count_by_query/self.k
        return np.mean(precision_by_query)