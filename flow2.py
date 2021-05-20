"""
2. We'll used the whole dataset. consider the dataset have 2*k classes.

- step 1: make new labels for the datatset by merging 2 classes into 1 randomly -> we got k classes.

- step 2: train classifier with k classes.

- step 3: extract feature and cluster the datatset by optimal number cluster algorithm.

- step 4: train and valid with new labels retrieved from step 3 --> check accuracy.
"""

import os
import pickle
import torch

from train_first import train_function, train_function2
from libs.utils.yaml_config import init
from training.config import update_config, config
from create_pretext_pytorch import extract_feature
from cluster_run import clustering
from libs.relabeling import get_relabeling

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    args, logging = init("experiments/cifar10/flow2_resnet18_vae.yaml")
    update_config(config, args)
    """
    + step 1: make new labels for the datatset by merging 2 classes into 1 randomly -> we got k classes.
    + step 2: train classifier with k classes."""
    # train
    args.cluster_dataset = 'train'
    # args.pretrained_path = train_function2(args, config)

    """- step 3: extract feature and cluster the datatset by optimal number cluster algorithm."""
    args.cluster_dataset = 'test'
    extract_feature(args, logging, class_merging=True)
    with open(args.fc1_path, 'rb') as f:
        data = pickle.load(f)
    logging.info('start clustering')
    opt_clst = clustering(args, logging, data)
    # logging.info(f'Optimal number of clusters: {opt_clst}')
    opt_clst = list(set(opt_clst))
    # relabel data
    logging.info('start relabeling data')
    relabeling = get_relabeling(args)(args, data)
    relabeling.load_state()
    relabeling.process_relabel()
    del relabeling

    """- step 4: train and valid with new labels retrieved from step 3 --> check accuracy."""
    # train on trainset and validate on test set
    for clusters in opt_clst:
        # clusters = 10
        config.DATASET.NUM_CLASSES = int(clusters)
        config.DATASET.LE_PATH = os.path.join(args.relabel_dir, str(clusters) + '_new_le.pkl')
        config.DATASET.TRAIN_LIST = os.path.join(args.relabel_dir, str(clusters) + '_train.pkl')
        config.DATASET.VAL_LIST = os.path.join(args.relabel_dir, str(clusters) + '_test.pkl')
        config.MODEL.PRETRAINED = False
        config.TRAIN.FINETUNE = ''
        config.TRAIN.OPTIMIZER = 'sgd'
        # config.TRAIN.BEGIN_EPOCH = 0
        # config.TRAIN.END_EPOCH = 20
        train_function(args, config, step=3)


if __name__ == '__main__':
    main()
