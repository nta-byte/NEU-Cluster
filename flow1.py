"""
1. We consider public dataset is splitted into 2 sets (train and test set).

- step 1 : We'll train our system with original train set.

- step 2: after that, We extract feature from test set and cluster them by optimal number cluster algorithm.

- step 3: train and valid test set with new labels retrieved from step 2 --> check accuracy.
"""

import os
import pickle
from train_first import train_function
from libs.utils.yaml_config import init
from training.config import update_config, config
from create_pretext_pytorch import extract_feature
from cluster_run import clustering
from libs.relabeling import get_relabeling


def main():
    """- step 1 : We'll train our system with original train set."""
    args, logging = init("experiments/neu-cls/flow1_resnet18.yaml")
    update_config(config, args)

    # train_function(args, config, step=1)

    """- step 2: after that, We extract feature from test set and cluster them by optimal number cluster algorithm."""
    # if os.path.exists(args.fc1_dir):
    #     print()
    # else:
    extract_feature(args, logging)
    with open(args.fc1_path, 'rb') as f:
        data = pickle.load(f)
    print('start clustering')
    opt_clst = clustering(args, logging, data)

    # relabel data
    print('start relabeling data')
    relabeling = get_relabeling(args)(args, data)
    relabeling.load_state()
    relabeling.process_relabel()

    """- step 3: train and valid test set with new labels retrieved from step 2 --> check accuracy."""
    opt_clst = list(set(opt_clst))
    print('best cluster:', opt_clst)
    for clusters in opt_clst:
        # clusters=10
        print('clusters', clusters)
        # with open(os.path.join(args.relabel_dir, str(clusters) + '_new_le.pkl'), 'rb') as f:
        #     new_le = pickle.load(f)

        config.DATASET.NUM_CLASSES = int(clusters)
        config.DATASET.LE_PATH = os.path.join(args.relabel_dir, str(clusters) + '_new_le.pkl')
        config.DATASET.TRAIN_LIST = os.path.join(args.relabel_dir, str(clusters) + '_train.pkl')
        config.DATASET.VAL_LIST = os.path.join(args.relabel_dir, str(clusters) + '_test.pkl')
        config.MODEL.PRETRAINED = False
        config.TRAIN.FINETUNE = args.pretrained_path
        config.TRAIN.BEGIN_EPOCH = 0
        config.TRAIN.END_EPOCH = 100
        train_function(args, config, step=3)


if __name__ == '__main__':
    main()