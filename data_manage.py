import random
import json
import pickle
import os
import numpy as np


with open('./config.json', 'r') as f:
    config = json.load(f)
n_classes = len(config['classes'])


class ManualDataset:
    """
    基于指定pathfile格式的数据集类。
    """
    def __init__(self, pathfile_root=None, pathfile_name=None, transform=None):
        """
        :param fold: 当前折
        :param mode: 'pre', 'inc' or 'test'
        :param pathfile_root: pathfile所在的根目录
        """
        self.pathfile_root = pathfile_root
        self.pathfile_name = pathfile_name
        self.transform = transform
        with open(os.path.join(self.pathfile_root, self.pathfile_name), 'r') as f:  # 将指定路径下的pathfile读入
            self.item_paths = f.readlines()

    def __getitem__(self, item):
        with open(self.item_paths[item][:-1].split(',')[0], 'rb') as f:
            data = pickle.load(f)
        if self.transform is not None:
            return self.transform(data['X'], data['y'])
        else:
            return data['X'].reshape(1, -1), data['y'].reshape(-1)

    def __len__(self):
        return len(self.item_paths)


class ManualTemplateDataset:
    def __init__(self, mode=None, pathfile_root=None, fold=None):
        """
        :param fold: 当前折
        :param mode: 'pre', 'inc' or 'test'
        :param pathfile_root: pathfile所在的根目录
        """
        assert mode in ['pre', 'inc', 'test'], f'ERROR: "{mode}" not in ["pre", "inc", "test"].'
        self.mode = mode
        self.pathfile_root = pathfile_root
        with open(os.path.join(self.pathfile_root, f'fold_{fold}_{mode}.txt'), 'r') as file:  # 将指定路径下的pathfile读入
            self.item_paths = file.readlines()

    def __getitem__(self, item):
        path_list = self.item_paths[item][:-1].split(',')
        with open(path_list[0], 'rb') as f:
            data = pickle.load(f)
        x_data = data['X'].reshape(1, -1)
        y_data = data['y'].reshape(-1)
        aligned_x_data = []
        for path in path_list[1:]:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                aligned_x_data.append(data['X'].reshape(1, -1))
        return x_data, aligned_x_data, y_data

    def __len__(self):
        return len(self.item_paths)


class ManualSampler:
    """
    采样器类，实现对样本集（dataset）中样本的索引（index）的管理和访问，可选重采样（resample）、重排（shuffle）等功能。
    iterable，调用__iter__返回一个样本索引的迭代器。
    """
    def __init__(self, dataset=None, resample=False, ges_idx=None, shuffle=False, random_state=None):
        self.dataset = dataset
        self.resample = resample
        self.ges_idx = ges_idx
        self.shuffle = shuffle
        self.random_state = random_state
        n_samples = len(self.dataset)
        n_samples_per_ges = n_samples // n_classes
        self.indices = list(range(n_samples))
        if self.resample:
            self.indices += list(range(ges_idx * n_samples_per_ges, (ges_idx + 1) * n_samples_per_ges)) * 6
        if self.shuffle:
            random.seed(self.random_state)
            self.indices = random.sample(self.indices, len(self.indices))
            random.seed()

    def __getitem__(self, item):
        new_sampler = ManualSampler(dataset=self.dataset, resample=self.resample, ges_idx=self.ges_idx, shuffle=self.shuffle, random_state=self.random_state)
        new_sampler.indices = self.indices[item]
        return new_sampler

    def __iter__(self):
        return iter(self.indices.copy())

    def __len__(self):
        return len(self.indices)


class ManualStreamLoader:
    """
    基于以上所实现采样器的流式样本加载器。
    iterable，调用__iter__返回一个样本迭代器。
    """
    def __init__(self, dataset, binary=False, resample=False, shuffle=False, random_state=None, ges_idx=None):
        self.dataset = dataset
        self.binary = binary
        self.resample = resample
        self.shuffle = shuffle
        self.random_state = random_state
        self.ges_idx = ges_idx
        self.sampler = ManualSampler(dataset=dataset, resample=resample, ges_idx=ges_idx, shuffle=shuffle, random_state=random_state)

    def __getitem__(self, item):
        new_loader = ManualStreamLoader(dataset=self.dataset, binary=self.binary, resample=self.resample, shuffle=self.shuffle, random_state=self.random_state)
        new_loader.sampler = self.sampler[item]
        return new_loader

    def __iter__(self):
        for dataset_idx in self.sampler:
            x_data, y_data = self.dataset[dataset_idx]
            if self.binary:
                y_data = np.array([1]) if y_data == self.ges_idx else np.array([-1])
            yield x_data, y_data  # 返回一个生成器迭代器

    def __len__(self):
        return len(self.sampler)


class ManualWholeLoader:
    """
        基于以上所实现采样器的样本全集加载器。
    """
    def __init__(self, dataset, binary=False, resample=False, shuffle=False, random_state=None, ges_idx=None):
        self.dataset = dataset
        self.binary = binary
        self.resample = resample
        self.shuffle = shuffle
        self.random_state = random_state
        self.ges_idx = ges_idx
        self.sampler = ManualSampler(dataset=dataset, resample=resample, ges_idx=ges_idx, shuffle=shuffle, random_state=random_state)

    def __getitem__(self, item):
        new_loader = ManualStreamLoader(dataset=self.dataset, binary=self.binary, resample=self.resample, shuffle=self.shuffle, random_state=self.random_state)
        new_loader.sampler = self.sampler[item]
        return new_loader

    def get(self):
        x_batch = np.concatenate([self.dataset[idx][0] for idx in self.sampler], 0)
        if self.binary:
            y_batch = np.concatenate([np.array([1]) if self.dataset[idx][1] == self.ges_idx else np.array([-1]) for idx in self.sampler], 0)
        else:
            y_batch = np.concatenate([np.array(self.dataset[idx][1]).reshape(-1) for idx in self.sampler], 0)
        return x_batch, y_batch, np.array(range(y_batch.size))

    def __len__(self):
        return len(self.sampler)
