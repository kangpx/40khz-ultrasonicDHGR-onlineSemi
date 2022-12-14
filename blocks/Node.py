import numpy as np
import random
from sklearn.metrics import accuracy_score


class Node:
    def __init__(self, depth, max_depth, min_samples_split, min_samples_leaf, max_features):
        # 本征参数
        self.depth = depth
        self.leaf = True
        self.label = -1  # 默认预测为负例
        self.attr_idx = None  # 测试属性的索引
        self.attr_thr = None  # 测试属性的阈值
        self.l_node = None
        self.r_node = None
        # 训练参数
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        assert self.depth >= 0

    @staticmethod
    def get_gini_trial(x_batch, y_batch, indices_select):
        def get_gini(the_mask):
            y_lhs, y_rhs = y_batch[the_mask], y_batch[~the_mask]
            n_tot = (n_lhs := y_lhs.size) + (n_rhs := y_rhs.size)
            n_lhs_pos, n_lhs_neg = np.count_nonzero(y_lhs > 0), np.count_nonzero(y_lhs < 0)
            n_rhs_pos, n_rhs_neg = np.count_nonzero(y_rhs > 0), np.count_nonzero(y_rhs < 0)
            gini_l, gini_r = 1 - (n_lhs_pos / n_lhs) ** 2 - (n_lhs_neg / n_lhs) ** 2, 1 - (n_rhs_pos / n_rhs) ** 2 - (n_rhs_neg / n_rhs) ** 2
            return (n_lhs / n_tot) * gini_l + (n_rhs / n_tot) * gini_r

        attr_list = []
        gini_list = []
        thr_list = []
        for attr_idx in indices_select:
            x_slice = x_batch[:, attr_idx]
            rand_thr = random.uniform(np.min(x_slice), np.max(x_slice))
            mask = x_slice < rand_thr
            if np.all(mask) or not np.any(mask):
                continue
            attr_list.append(attr_idx)
            gini_list.append(get_gini(the_mask=mask))
            thr_list.append(rand_thr)
        return attr_list, gini_list, thr_list

    @property
    def is_expand(self):
        if self.depth == 0 and self.leaf is True and self.l_node is None and self.r_node is None:
            return False
        else:
            return True

    @property
    def n_leaves(self):
        if self.leaf:
            return 1
        else:
            n_leaves_l = self.l_node.n_leaves
            n_leaves_r = self.r_node.n_leaves
            return n_leaves_l + n_leaves_r

    @property
    def local_info(self):
        if self.leaf:
            return f'$ LAYER: {self.depth:<2}, LABEL: {self.label:<+2}'
        else:
            return f'$ LAYER: {self.depth:<2}, ATTR_IDX: {self.attr_idx:<5}, ATTR_THR: {self.attr_thr:<+10.4f}'

    @property
    def traverse_info(self):
        nodes = [(self, '*')]
        info = []
        while nodes:
            current_node, current_path = nodes.pop(0)
            info.append((current_node.local_info, current_path))
            if current_node.l_node:
                nodes.append((current_node.l_node, current_path + 'l'))
            if current_node.r_node:
                nodes.append((current_node.r_node, current_path + 'r'))
        return info

    def fit(self, x_batch, y_batch, idx_batch=None, attr_indices=None):
        assert (y_batch.ndim == 1) and (x_batch.ndim == 2), 'ERROR: input dimension error.'
        if attr_indices is None:
            attr_indices = list(range(1080))
        # 统计正负例数量和总样本数量
        n_tot = (n_pos := np.count_nonzero(y_batch > 0)) + (n_neg := np.count_nonzero(y_batch < 0))

        def set_leaf():  # 后两项是为IncNode中的grow做准备
            self.leaf, self.label = True, 1 if n_pos >= n_neg else -1
            return self.label

        # 检查是否满足分裂停止条件（all 1、all -1、max_depth、min_samples_split），若满足则返回
        if n_pos == 0 or n_neg == 0 or self.depth >= self.max_depth or n_tot < self.min_samples_split:
            return set_leaf()
        # 特征选择
        indices_select = random.sample(attr_indices, int(np.sqrt(len(attr_indices)))) if self.max_features == 'sqrt' else list(attr_indices).copy()
        # gini试验
        attr_list, gini_list, threshold_list = self.get_gini_trial(x_batch=x_batch, y_batch=y_batch, indices_select=indices_select)
        # 确定最优(属性，阈值)
        argmin_idx = np.argmin(gini_list)
        self.attr_idx, self.attr_thr = attr_list[argmin_idx], threshold_list[argmin_idx]
        # 检查是否满足分裂停止条件（min_samples_leaf），若满足则返回
        if not self.min_samples_leaf <= np.count_nonzero(mask := x_batch[:, self.attr_idx] < self.attr_thr) <= n_tot - self.min_samples_leaf:
            return set_leaf()
        # 已满足分裂条件，将当前节点置为非叶子节点
        self.leaf = False
        # 基于最优(属性，阈值)分裂样本集
        pack_l = (x_batch[mask], y_batch[mask], idx_batch[mask] if idx_batch is not None else None)
        pack_r = (x_batch[~mask], y_batch[~mask], idx_batch[~mask] if idx_batch is not None else None)
        # 将本节点最优属性从new_attr_indices中移除
        new_attr_indices = list(attr_indices).copy()
        new_attr_indices.remove(self.attr_idx)
        # 左节点生长
        self.l_node = Node(depth=self.depth + 1, max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                           min_samples_leaf=self.min_samples_leaf, max_features=self.max_features)
        self.l_node.fit(x_batch=pack_l[0], y_batch=pack_l[1], idx_batch=pack_l[2], attr_indices=new_attr_indices)
        # 右节点生长
        self.r_node = Node(depth=self.depth + 1, max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                           min_samples_leaf=self.min_samples_leaf, max_features=self.max_features)
        self.r_node.fit(x_batch=pack_r[0], y_batch=pack_r[1], idx_batch=pack_r[2], attr_indices=new_attr_indices)
        # 回缩
        if self.l_node.leaf and self.r_node.leaf and (self.l_node.label == self.r_node.label):
            self.leaf, self.label = True, self.l_node.label
            self.l_node, self.r_node = None, None
        return self

    def __uni_predict(self, x):
        assert x.ndim == 1, 'ERROR: input dimension error.'
        if self.leaf:
            return np.array(self.label).reshape(-1).astype('float64')
        else:
            return self.l_node.__uni_predict(x) if x[self.attr_idx] < self.attr_thr else self.r_node.__uni_predict(x)

    def predict(self, x_batch):
        assert x_batch.ndim == 2, 'ERROR: input dimension error.'
        pred = np.concatenate([self.__uni_predict(x) for x in x_batch], 0)
        return pred

    def score(self, x_batch, y_batch):
        assert (x_batch.ndim == 2) and (y_batch.ndim == 1), 'ERROR: input dimension error.'
        return accuracy_score(y_batch, self.predict(x_batch))
