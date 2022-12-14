from .Forest import *


class WeightedForest(Forest):
    def __init__(self, from_sklearn=False, n_estimators=50, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_leaf_nodes=50,
                 idxpol_cap=20, idxpol_pri=True, idxpol_pst=True, idxpol_mss=20, alpha=0.1, inc_source=None, pos_weight=1.0, neg_weight=1.0):
        super().__init__(from_sklearn=from_sklearn, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                         idxpol_cap=idxpol_cap, idxpol_pri=idxpol_pri, idxpol_pst=idxpol_pst, idxpol_mss=idxpol_mss, alpha=alpha, inc_source=inc_source)
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def reset_weight(self):
        self.pos_weight = 1.0
        self.neg_weight = 1.0

    def predict_proba(self, x_batch):
        assert x_batch.ndim == 2, 'ERROR: input dimension error.'
        if self.estimators_ is None:
            return np.concatenate((np.ones((x_batch.shape[0], 1)), np.zeros((x_batch.shape[0], 1))), 1)
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
                votes_p = np.sum(votes_mat_pos, 0) * self.pos_weight
                votes_n = np.sum(votes_mat_neg, 0) * self.neg_weight
                proba_p = votes_p / (votes_p + votes_n)
                proba_n = votes_n / (votes_p + votes_n)
                return np.concatenate((proba_n.reshape(-1, 1), proba_p.reshape(-1, 1)), 1)
            else:
                return np.concatenate((np.ones((x_batch.shape[0], 1)), np.zeros((x_batch.shape[0], 1))), 1)

