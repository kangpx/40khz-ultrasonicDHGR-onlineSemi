import numpy as np
import random
from .Node import Node
from .IndexPool import IndexPool


class IncNode(Node):
    def __init__(self,
                 depth, max_depth,
                 min_samples_split, min_samples_leaf, max_features,
                 idxpol_cap, idxpol_pri, idxpol_pst, idxpol_mss, alpha,
                 inc_source):
        """
        :param idxpol_cap: int, index pool的capacity
        :param idxpol_pri: bool, index pool的priority
        :param idxpol_pst: bool, index pool的pre-set
        :param idxpol_mss: int, index pool的min_samples_split
        :param inc_source: 定义了get接口的数据源
        """
        # 默认为pre模式
        super().__init__(depth=depth, max_depth=max_depth,
                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features)
        self.idxpol = None
        self.idxpol_cap = idxpol_cap
        self.idxpol_pri = idxpol_pri
        self.idxpol_pst = idxpol_pst
        self.idxpol_mss = idxpol_mss
        self.alpha = alpha
        self.inc_source = inc_source

    def set_fit_params(self, **kwargs):
        for key, value in kwargs.items():
            self.__dict__[key] = value
        if self.l_node:
            self.l_node.set_fit_params(**kwargs)
        if self.r_node:
            self.r_node.set_fit_params(**kwargs)

    def clear_idxpol(self):
        self.idxpol = None
        if self.l_node:
            self.l_node.idxpol = None
        if self.r_node:
            self.r_node.idxpol = None

    def fit(self, x_batch, y_batch, idx_batch=None, attr_indices=None):
        if x_batch is None or y_batch is None:
            return self
        if attr_indices is None:
            attr_indices = list(range(1080))
        # 统计正负例数量和总样本数量
        n_tot = (n_pos := np.count_nonzero(y_batch > 0)) + (n_neg := np.count_nonzero(y_batch < 0))

        def set_leaf():
            self.leaf, self.label = True, 1 if n_pos >= n_neg else -1
            # 若self.idxpol_pst为真，则在将当前节点置为叶子节点的同时创建idxpol，并使用当前叶子节点中样本的索引和标签初始化idxpol
            if self.idxpol_pst and (self.idxpol is None):
                self.idxpol = IndexPool(capacity=self.idxpol_cap, priority=self.idxpol_pri, idx_batch=idx_batch, y_batch=y_batch)
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
        x_batch_l, y_batch_l, idx_batch_l = x_batch[mask], y_batch[mask], idx_batch[mask] if idx_batch is not None else None
        x_batch_r, y_batch_r, idx_batch_r = x_batch[~mask], y_batch[~mask], idx_batch[~mask] if idx_batch is not None else None
        # 将本节点最优属性从attr_indices中移除
        new_attr_indices = list(attr_indices).copy()
        new_attr_indices.remove(self.attr_idx)
        # 左节点生长
        self.l_node = IncNode(depth=self.depth + 1, max_depth=self.max_depth,
                              min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                              idxpol_cap=self.idxpol_cap, idxpol_pri=self.idxpol_pri, idxpol_pst=self.idxpol_pst,
                              alpha=self.alpha, idxpol_mss=self.idxpol_mss, inc_source=self.inc_source)
        self.l_node.fit(x_batch=x_batch_l, y_batch=y_batch_l, idx_batch=idx_batch_l, attr_indices=new_attr_indices)
        # 右节点生长
        self.r_node = IncNode(depth=self.depth + 1, max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                              min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                              idxpol_cap=self.idxpol_cap, idxpol_pri=self.idxpol_pri, idxpol_pst=self.idxpol_pst,
                              alpha=self.alpha, idxpol_mss=self.idxpol_mss, inc_source=self.inc_source)
        self.r_node.fit(x_batch=x_batch_r, y_batch=y_batch_r, idx_batch=idx_batch_r, attr_indices=new_attr_indices)
        # 回缩
        if self.l_node.leaf and self.r_node.leaf and (self.l_node.label == self.r_node.label):
            self.leaf, self.label = True, self.l_node.label
            self.l_node, self.r_node = None, None
        return self

    def fertilize(self, x, y, idx, attr_indices):
        if self.leaf:
            if self.idxpol is None:
                self.idxpol = IndexPool(capacity=self.idxpol_cap, priority=self.idxpol_pri, y_batch=y, idx_batch=idx)
            else:
                self.idxpol.push(y=y, idx=idx)
            if self.idxpol.occupation >= self.idxpol_mss and self.idxpol.gini > 1 - 1 / (1 + self.alpha) ** 2:
                x_batch, y_batch, idx_batch = self.inc_source.get(self.idxpol.idx_batch)
                if x_batch is not None:
                    self.leaf, self.label, self.idxpol = False, None, None
                    self.fit(x_batch=x_batch, y_batch=y_batch, idx_batch=idx_batch, attr_indices=attr_indices)
                    return True   # 若满足分裂条件则返回True
                else:
                    self.leaf, self.label, self.idxpol = True, None, None
                    return False
            else:
                return False  # 否则返回False
        else:
            new_attr_indices = list(attr_indices).copy()
            new_attr_indices.remove(self.attr_idx)
            if x.reshape(-1)[self.attr_idx] < self.attr_thr:
                return self.l_node.fertilize(x=x, y=y, idx=idx, attr_indices=new_attr_indices)
            else:
                return self.r_node.fertilize(x=x, y=y, idx=idx, attr_indices=new_attr_indices)
