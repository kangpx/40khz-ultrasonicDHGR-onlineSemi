import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


class ForestUnion:
    def __init__(self, forests=None, dtw=None):
        if forests is None:
            self.forests = []
        else:
            self.forests = forests
        self.dtw = dtw

    def __getitem__(self, item):
        return self.forests[item]

    def __len__(self):
        return len(self.forests)
    
    def append(self, forest):
        self.forests.append(forest)

    def reset_weight(self):
        for forest in self.forests:
            forest.reset_weight()

    def predict_proba(self, x_batch):
        proba_mat = np.concatenate([forest.predict_proba(self.dtw(x_batch, i) if self.dtw is not None else x_batch)[:, 1].reshape(1, -1) for i, forest in enumerate(self.forests)], 0)
        # proba_sum = np.sum(proba_mat, 0)
        # return proba_mat / proba_sum if (proba_sum > 0).any() else proba_mat  # proba_mat在森林未分裂时为全零，也即forest_union中的每一个森林对任意样本的预测值均为-1且负例置信度为1（正例置信度为0）
        return proba_mat  # proba_mat在森林未分裂时为全零，也即forest_union中的每一个森林对任意样本的预测值均为-1且负例置信度为1（正例置信度为0）

    def predict(self, x_batch):
        assert x_batch.ndim == 2, 'ERROR: input dimension error.'
        proba_mat = self.predict_proba(x_batch)
        return np.argmax(proba_mat, 0).reshape(-1)

    def binary_predict(self, x_batch):
        assert x_batch.ndim == 2
        return [forest.predict(self.dtw(x_batch, i) if self.dtw is not None else x_batch) for i, forest in enumerate(self.forests)]

    def score(self, x_batch, y_batch):
        assert x_batch.ndim == 2 and y_batch.ndim == 1, 'ERROR: input dimension error.'
        return accuracy_score(y_batch, self.predict(x_batch))

    def confusion_matrix(self, x_batch, y_batch, normalize=None):
        assert x_batch.ndim == 2 and y_batch.ndim == 1, 'ERROR: input dimension error.'
        return confusion_matrix(y_batch, self.predict(x_batch), normalize=normalize)
