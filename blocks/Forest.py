import numpy as np
from .IncNode import IncNode
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class Forest:
    def __init__(self, from_sklearn=False, n_estimators=50, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_leaf_nodes=50,
                 idxpol_cap=20, idxpol_pri=True, idxpol_pst=True, idxpol_mss=20, alpha=0.1, inc_source=None):
        self.from_sklearn = from_sklearn
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.idxpol_cap = idxpol_cap
        self.idxpol_pri = idxpol_pri
        self.idxpol_pst = idxpol_pst
        self.idxpol_mss = idxpol_mss
        self.alpha = alpha
        self.inc_source = inc_source
        self.estimators_ = None

    def __getitem__(self, item):
        return self.estimators_[item] if self.estimators_ else None

    def __len__(self):
        return len(self.estimators_) if self.estimators_ else None

    def reset_trees(self, scope):
        if self.estimators_ is None:
            self.estimators_ = [IncNode(depth=0, max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                                        idxpol_cap=self.idxpol_cap, idxpol_pri=self.idxpol_pri, idxpol_pst=self.idxpol_pst, idxpol_mss=self.idxpol_mss, alpha=self.alpha, inc_source=self.inc_source)
                                for _ in range(self.n_estimators)]
        for i_tree in scope:
            self.estimators_[i_tree] = IncNode(depth=0, max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                                               idxpol_cap=self.idxpol_cap, idxpol_pri=self.idxpol_pri, idxpol_pst=self.idxpol_pst, idxpol_mss=self.idxpol_mss, alpha=self.alpha, inc_source=self.inc_source)

    def get_expands(self):
        if self.estimators_ is None:
            return []
        else:
            return [tree for tree in self.estimators_ if tree.is_expand]

    def fit(self, x_batch, y_batch, idx_batch=None, attr_indices=None):
        assert x_batch.ndim == 2 and y_batch.ndim == 1, 'ERROR: input dimension error.'
        if self.from_sklearn:
            forest = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
                                            max_features=self.max_features, max_leaf_nodes=self.max_leaf_nodes)
            forest.fit(x_batch, y_batch)
            self.estimators_ = forest.estimators_.copy()
        else:
            if self.estimators_ is None:
                self.estimators_ = []
            for _ in range(self.n_estimators):
                tree = IncNode(depth=0, max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                               idxpol_cap=self.idxpol_cap, idxpol_pri=self.idxpol_pri, idxpol_pst=self.idxpol_pst, idxpol_mss=self.idxpol_mss, alpha=self.alpha, inc_source=self.inc_source)
                tree.fit(x_batch, y_batch, idx_batch, attr_indices)
                self.estimators_.append(tree)
        return self

    def partial_fertilize(self, x, y, idx, attr_indices, scope):
        if self.estimators_ is None:
            self.estimators_ = [IncNode(depth=0, max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                                idxpol_cap=self.idxpol_cap, idxpol_pri=self.idxpol_pri, idxpol_pst=self.idxpol_pst, idxpol_mss=self.idxpol_mss, alpha=self.alpha, inc_source=self.inc_source)
                                for _ in range(self.n_estimators)]
        return [self.estimators_[i_tree].fertilize(x=x, y=y, idx=idx, attr_indices=attr_indices) for i_tree in scope] if not self.from_sklearn else None

    def fertilize(self, x, y, idx, attr_indices):
        if self.estimators_ is None:
            self.estimators_ = [IncNode(depth=0, max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                                idxpol_cap=self.idxpol_cap, idxpol_pri=self.idxpol_pri, idxpol_pst=self.idxpol_pst, idxpol_mss=self.idxpol_mss, alpha=self.alpha, inc_source=self.inc_source)
                                for _ in range(self.n_estimators)]
        return [tree.fertilize(x=x, y=y, idx=idx, attr_indices=attr_indices) for tree in self.estimators_] if not self.from_sklearn else None

    def predict_treewise(self, x_batch):
        assert x_batch.ndim == 2, 'ERROR: input dimension error.'
        if self.estimators_ is None:
            return None
        else:
            if self.from_sklearn:
                votes_mat = np.concatenate([tree.predict(x_batch).reshape(1, -1) for tree in self.estimators_], 0)
            else:
                expands = self.get_expands()
                if expands:
                    votes_mat = np.concatenate([tree.predict(x_batch).reshape(1, -1) for tree in expands], 0)
                else:
                    votes_mat = np.array([])
            return votes_mat

    def predict_proba(self, x_batch):
        assert x_batch.ndim == 2, 'ERROR: input dimension error.'
        if self.estimators_ is None:
            return np.concatenate((np.zeros((x_batch.shape[0], 1)), np.ones((x_batch.shape[0], 1))), 1)
        else:
            if self.from_sklearn:
                votes_mat = np.concatenate([tree.predict(x_batch).reshape(1, -1) for tree in self.estimators_], 0)
            else:
                expands = self.get_expands()
                if expands:
                    votes_mat = np.concatenate([tree.predict(x_batch).reshape(1, -1) for tree in expands], 0)
                else:
                    votes_mat = np.array([])
            if votes_mat.size > 0:
                pos_mask = votes_mat > 0
                neg_mask = ~ pos_mask
                votes_mat_pos, votes_mat_neg = votes_mat.copy(), votes_mat.copy()
                votes_mat_pos[pos_mask], votes_mat_pos[neg_mask] = 1.0, 0.0
                votes_mat_neg[pos_mask], votes_mat_neg[neg_mask] = 0.0, 1.0
                votes_p = np.sum(votes_mat_pos, 0)
                votes_n = np.sum(votes_mat_neg, 0)
                proba_p = votes_p / (votes_p + votes_n)
                proba_n = votes_n / (votes_p + votes_n)
                return np.concatenate((proba_n.reshape(-1, 1), proba_p.reshape(-1, 1)), 1)
            else:
                return np.concatenate((np.zeros((x_batch.shape[0], 1)), np.ones((x_batch.shape[0], 1))), 1)

    def predict(self, x_batch):
        assert x_batch.ndim == 2, 'ERROR: input dimension error.'
        return np.array([1 if proba >= 0.5 else -1 for proba in self.predict_proba(x_batch)[:, 1]]).reshape(-1)

    def score_tree(self, x_batch, y_batch):
        acc_vec = [self.estimators_[i].score(x_batch, y_batch) for i in range(self.n_estimators)]
        return np.mean(acc_vec)

    def score(self, x_batch, y_batch):
        return accuracy_score(y_batch, self.predict(x_batch))
