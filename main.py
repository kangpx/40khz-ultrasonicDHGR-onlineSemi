import math
import multiprocessing as mp
from tqdm import tqdm
from matplotlib import pyplot as plt
from data_manage import *
from blocks.FeatureBuffer import *
from blocks.WeightedForest import *
from blocks.ForestUnion import *


# 加载config文件
with open('./config.json', 'r') as f:
    config = json.load(f)

# 全局变量
n_classes = len(config['classes'])
subjects = config['subjects']
n_subjects = len(config['subjects'])
dataset_root = config['dataset_root']
pathfile_root = config['pathfile_root']
ref_forest_params = config['ref_forest_params']
inc_forest_params = config['inc_forest_params']
feabuf_params = config['feabuf_params']
max_gini_params = config['max_gini_params']
tree_batch_size = config['tree_batch_size']
multiprocess = config['multiprocess']
n_cores = config['n_cores']
en_probe = config['en_probe']


class Probe:
    def __init__(self, test_dataset):
        self.test_dataset = test_dataset
        self.x_test, self.y_test, self.idx_test = ManualWholeLoader(dataset=self.test_dataset).get()
        self.ref_scores = []
        self.inc_scores = []
        self.mix_scores = []

    def _mix_score(self, ref, inc, mix_ratio):
        proba_ref = ref.predict_proba(self.x_test)
        proba_inc = inc.predict_proba(self.x_test)
        proba_mix = (1.0 - mix_ratio) * proba_ref + mix_ratio * proba_inc
        pred = np.argmax(proba_mix, 0).reshape(-1)
        return accuracy_score(self.y_test, pred)
    
    def reset(self):
        self.ref_scores = []
        self.inc_scores = []
        self.mix_scores = []

    @property
    def final_scores(self):
        return self.ref_scores[-1], self.inc_scores[-1], self.mix_scores[-1]

    def __call__(self, ref, inc, mix_ratio):
        self.ref_scores.append(ref.score(self.x_test, self.y_test))
        self.inc_scores.append(inc.score(self.x_test, self.y_test))
        self.mix_scores.append(self._mix_score(ref, inc, mix_ratio))
        

class Classifier:
    def __init__(self):
        self.feabuf = ClassWiseFeatureBuffer(n_classes=n_classes, pos_class=None, block_capacity=feabuf_params['block_capacity'])
        self.ref_union = None
        self.pseudo_pos_cnt = None
        self.pseudo_neg_cnt = None
        self.inc_union = None
        self.n_tree_batches = math.ceil(inc_forest_params["n_estimators"] / tree_batch_size)
        self.tree_batch_address = [range(i * tree_batch_size, (i + 1) * tree_batch_size if (i + 1) * tree_batch_size <= inc_forest_params["n_estimators"] else inc_forest_params["n_estimators"]) for i in range(self.n_tree_batches)]

    def get_ref_union(self, pre_dataset):
        self.ref_union = ForestUnion()
        for i in tqdm(range(n_classes), desc='# Offline Training: '):
            x_pre, y_pre, _, = ManualWholeLoader(dataset=pre_dataset, binary=True, resample=False, ges_idx=i).get()
            forest = WeightedForest(**ref_forest_params)
            self.ref_union.append(forest.fit(x_pre, y_pre))
        return self.ref_union

    def get_inc_union(self, inc_dataset, probe):
        current_batch = 0
        volume_counts = 0
        self.feabuf.clear()
        self.pseudo_pos_cnt = np.zeros(len(self.ref_union))
        self.pseudo_neg_cnt = np.zeros(len(self.ref_union))
        self.inc_union = ForestUnion()
        for _ in range(n_classes):
            self.inc_union.append(WeightedForest(**inc_forest_params, inc_source=self.feabuf))
        inc_loader = ManualStreamLoader(shuffle=True, dataset=inc_dataset, random_state=None)
        for idx, (x, _) in tqdm(enumerate(inc_loader), desc='# Incremental Training: '):
            if idx % max_gini_params['period'] == 0:
                if probe is not None:
                    probe(ref=self.ref_union, inc=self.inc_union, mix_ratio=min(0.16 * current_batch, 0.8))
                if self.feabuf.occupation >= feabuf_params['threshold']:
                    for _ in range(max_gini_params['volume']):
                        for i, forest in enumerate(self.inc_union):
                            self.feabuf.pos_class = i
                            forest.partial_fertilize(*self.feabuf.sample(), attr_indices=range(1080), scope=self.tree_batch_address[current_batch])
                        if (volume_counts := volume_counts + 1) >= max_gini_params['quota']:
                            volume_counts = 0
                            current_batch = (current_batch + 1) % self.n_tree_batches
                            for forest in self.inc_union:
                                forest.reset_trees(scope=self.tree_batch_address[current_batch])
                            break
            self.feabuf.push(x, self.ref_union.predict(x), np.array([idx]))
            y_pseudo_classwise = np.array(self.ref_union.binary_predict(x)).reshape(-1)
            self.pseudo_pos_cnt[y_pseudo_classwise > 0] += 1
            self.pseudo_neg_cnt[y_pseudo_classwise < 0] += 1
            for i, (pos_cnt_i, neg_cnt_i) in enumerate(zip(self.pseudo_pos_cnt, self.pseudo_neg_cnt)):
                self.ref_union[i].pos_weight = 0.125 * (neg_cnt_i + 2) / (pos_cnt_i + 2)
        if probe is not None:
            probe(ref=self.ref_union, inc=self.inc_union, mix_ratio=0.8)
        return self.inc_union


def show_curve(probe_list):
    plt.figure(figsize=(2.6 * 7, 6))
    for i, subject in enumerate(subjects):
        # 
        ax = plt.subplot(3, 7, 1 + i)
        ref_curve = probe_list[i].ref_scores
        if i > 0:
            plt.yticks([0.8, 0.85, 0.90, 0.95, 1.0], [], fontsize=12)
        else:
            plt.yticks([0.8, 0.85, 0.90, 0.95, 1.0], [0.8, 0.85, 0.90, 0.95, 1.0], fontsize=13)

        plt.title(f'{subjects[i]}', fontsize=17)
        line1, = plt.plot(range(0, len(ref_curve)), ref_curve, marker='o', markersize=3)
        plt.hlines(probe_list[i].ref_scores[0], 0, 400, linestyles='--', colors='grey', linewidth=2)
        plt.xticks([0, 10, 20, 30, 40], [])
        ax.tick_params(bottom=False, top=False, left=False, right=False)
        plt.xlim(0, 40)
        plt.ylim(0.8, 1)
        plt.grid()
        
        ax = plt.subplot(3, 7, 8 + i)
        inc_curve = probe_list[i].inc_scores
        if i > 0:
            plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [], fontsize=15)
        else:
            plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=13)
            plt.ylabel('Accuracy', fontsize=17)
        line2, = plt.plot(range(0, len(inc_curve)), inc_curve, marker='o', markersize=3, color='firebrick')
        plt.hlines(probe_list[i].ref_scores[0], 0, 400, linestyles='--', colors='grey', linewidth=2)
        plt.xticks([0, 100, 200, 300, 400], [])
        ax.tick_params(bottom=False, top=False, left=False, right=False)
        plt.xlim(0, 40)
        plt.ylim(0.125, 1)
        plt.grid()
        
        ax = plt.subplot(3, 7, 15 + i)
        mix_curve = probe_list[i].mix_scores
        if i > 0:
            plt.yticks([0.8, 0.85, 0.90, 0.95, 1.0], [], fontsize=12)
        else:
            plt.yticks([0.8, 0.85, 0.90, 0.95, 1.0], [0.8, 0.85, 0.90, 0.95, 1.0], fontsize=13)
        if i == 3:
            plt.xlabel('# Incremental Samples', fontsize=17)
        line3, = plt.plot(range(0, len(mix_curve)), mix_curve, marker='o', markersize=3, color='green')
        line4, = plt.plot(range(0, len(ref_curve)), ref_curve, linestyle='-.', linewidth=2)
        line5, = plt.plot(range(0, len(inc_curve)), inc_curve, color='firebrick', linestyle='-.', linewidth=2)
        line6 = plt.hlines(probe_list[i].ref_scores[0], 0, 40, linestyles='--', colors='grey', linewidth=2)
        plt.xticks([0, 10, 20, 30, 40], [0, 100, 200, 300, 400], fontsize=13, rotation=0, ha='center')
        ax.tick_params(bottom=False, top=False, left=False, right=False)
        plt.xlim(0, 40)
        plt.ylim(0.8, 1)
        plt.grid()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.2, wspace=0.15)
    plt.show()


def fold_test(fold):
    pre_dataset = ManualDataset(pathfile_root=pathfile_root, pathfile_name=f'fold_{fold}_pre.txt')
    inc_dataset = ManualDataset(pathfile_root=pathfile_root, pathfile_name=f'fold_{fold}_inc.txt')
    test_dataset = ManualDataset(pathfile_root=pathfile_root, pathfile_name=f'fold_{fold}_test.txt')
    probe = Probe(test_dataset) if en_probe else None
    classifier = Classifier()
    classifier.get_ref_union(pre_dataset)
    classifier.get_inc_union(inc_dataset, probe)
    return probe


def main():
    if multiprocess:
        pool = mp.Pool(processes=n_cores)
        result_list = [pool.apply_async(fold_test, args=(fold, )) for fold in range(n_subjects)]
        pool.close()
        pool.join()
        probe_list = [result.get() for result in result_list]
    else:
        probe_list = [fold_test(fold) for fold in range(n_subjects)]
    show_curve(probe_list)
    print("# Final Scores:")
    for subject, probe in zip(subjects, probe_list):
        print(f'{subject: <4}:: ref:{probe.final_scores[0]: .3f}, inc:{probe.final_scores[1]: .3f}, mix:{probe.final_scores[2]: .3f}')
    print(f'mean:: ref: {np.mean([probe.final_scores[0] for probe in probe_list]): .3f}, inc: {np.mean([probe.final_scores[1] for probe in probe_list]): .3f}, mix: {np.mean([probe.final_scores[2] for probe in probe_list]): .3f}')


if __name__ == '__main__':
    main()
