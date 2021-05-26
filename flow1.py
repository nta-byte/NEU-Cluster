"""
1. We consider public dataset is splitted into 2 sets (train and test set).

- step 1 : We'll train our system with original train set.

- step 2: after that, We extract feature from test set and cluster them by optimal number cluster algorithm.

- step 3: train and valid test set with new labels retrieved from step 2 --> check accuracy.
"""

import os
import pickle
from time import time
from train_first import train_function
from libs.utils.yaml_config import init
from training.config import update_config, config
from create_pretext_pytorch import extract_feature
from cluster_run import clustering
from libs.relabeling import get_relabeling

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    startinit = time()
    args, logging = init("experiments/cifar10/flow1_resnet18_vae.yaml")
    update_config(config, args)
    doneinit = time()
    logging.info(f"<============> Init time: {round(doneinit - startinit, 2)}")

    """- step 1 : We'll train our system with original train set."""
    args.cluster_dataset = 'train_test'
    args.pretrained_path = train_function(args, config, step=1)
    done_firsttrain = time()
    logging.info(f"<============> First training time: {round(done_firsttrain - doneinit, 2)}")

    """- step 2: after that, We extract feature from test set and cluster them by optimal number cluster algorithm."""
    args.cluster_dataset = 'test'
    extract_feature(args, logging)
    done_extract = time()
    logging.info(f"<============> Feature extraction time: {done_extract - done_firsttrain}")
    with open(args.fc1_path, 'rb') as f:
        data = pickle.load(f)
    print('start clustering')
    opt_clst = clustering(args, logging, data, org_eval=True)
    done_clustering = time()
    logging.info(f"<============> Clustering time: {round(done_clustering - done_extract, 2)}")

    # relabel data
    print('start relabeling data')
    relabeling = get_relabeling(args)(args, data)
    relabeling.load_state()
    relabeling.process_relabel()
    done_relabel = time()
    logging.info(f"<============> Relabeling time: {round(done_relabel - done_clustering, 2)}")

    """- step 3: train and valid test set with new labels retrieved from step 2 --> check accuracy."""
    opt_clst = list(set(opt_clst))
    print('best cluster:', opt_clst)
    for clusters in opt_clst:
        # clusters=10
        print('clusters', clusters)

        config.DATASET.NUM_CLASSES = int(clusters)
        config.DATASET.LE_PATH = os.path.join(args.relabel_dir, str(clusters) + '_new_le.pkl')
        config.DATASET.TRAIN_LIST = os.path.join(args.relabel_dir, str(clusters) + '_train.pkl')
        config.DATASET.VAL_LIST = os.path.join(args.relabel_dir, str(clusters) + '_test.pkl')
        config.MODEL.PRETRAINED = True
        # config.TRAIN.FINETUNE = args.pretrained_path
        # config.TRAIN.BEGIN_EPOCH = 0
        # config.TRAIN.END_EPOCH = 100
        train_function(args, config, step=3)
    done_lasttrain = time()
    logging.info(f"<============> Relabeling time: {round(done_lasttrain - done_relabel, 2)}")


if __name__ == '__main__':
    main()
