import os
import random
import json


with open('./config.json', 'r') as f:
    config = json.load(f)

dataset_root = config['dataset_root']
pathfile_root = config['pathfile_root']
subjects = config['subjects']
n_subjects = len(subjects)
n_classes = len(config['classes'])
inc_ratio = config['inc_ratio']
rand_state = config['pathfile_random_state']


def get_subject_path(subject):
    """
    :param subject: 某一手势采集者
    :return: 该手势采集者所有的手势样本，样本按照（手势类别编号，手势样本编号）排序
    """
    assert subject in subjects, "ERROR: target subject is not in specified subjects."
    targ_item_paths = []
    for cls in os.listdir(name_dir := os.path.join(dataset_root, f'subject_{subjects.index(subject)}')):  # 当前手势采集者的目录
        for local_path in os.listdir(cls_dir := os.path.join(name_dir, cls)):  # 当前手势采集者的当前手势类别的目录
            targ_item_paths.append(os.path.join(cls_dir, local_path))  # 当前手势采集者的当前手势类别的当前样本的路径
    targ_item_paths = sorted(targ_item_paths, key=lambda x: (x.split('/')[-2], x.split('/')[-1]))  # 使得(手势类别编号, 手势样本编号)较小者排序靠前
    return targ_item_paths


def split_path(path_list, random_state):
    """
    将给定的所有path分割成两份
    :param path_list:
    :param random_state:
    :return:
    """
    path_list_s = []
    path_list_t = []
    for i_class in range(n_classes):
        path_i_class = [path for path in path_list if path.split('/')[-2][-1] == f'{i_class}']
        if random_state:
            random.seed(random_state + i_class)
        else:
            random.seed(None)
        path_list_s += random.sample(path_i_class, int(inc_ratio * len(path_i_class)))
        path_list_t += [path for path in path_i_class if path not in path_list_s]
        random.seed()
    return sorted(path_list_s, key=lambda x: (x.split('/')[-2], x.split('/')[-1])), sorted(path_list_t, key=lambda x: (x.split('/')[-2], x.split('/')[-1]))


def get_cross_validation_path(user, random_state=None):
    pre_paths = []
    for subject in [name for name in subjects if name != user]:
        pre_paths += get_subject_path(subject)
    user_paths = get_subject_path(user)
    inc_paths, test_paths = split_path(user_paths, random_state)
    with open(os.path.join(pathfile_root, f'fold_{subjects.index(user)}_pre.txt'), 'w') as file:
        for path in pre_paths:
            file.write(f'{path}\n')
    with open(os.path.join(pathfile_root, f'fold_{subjects.index(user)}_inc.txt'), 'w') as file:
        for path in inc_paths:
            file.write(f'{path}\n')
    with open(os.path.join(pathfile_root, f'fold_{subjects.index(user)}_test.txt'), 'w') as file:
        for path in test_paths:
            file.write(f'{path}\n')


if __name__ == '__main__':
    for user in subjects:
        get_cross_validation_path(user, random_state=rand_state)
