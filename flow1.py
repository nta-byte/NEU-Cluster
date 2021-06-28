"""
1. We consider public dataset is splitted into 2 sets (train and test set).

- step 1 : We'll train our system with original train set.

- step 2: after that, We extract feature from test set and cluster them by optimal number cluster algorithm.

- step 3: train and valid test set with new labels retrieved from step 2 --> check accuracy.
"""

import os
import pickle
from time import time
import argparse
from libs.utils.yaml_config import init_v2
from train_first import train_function
from libs.utils.yaml_config import init
# from training.config import update_config, config
from create_pretext_pytorch import extract_feature
from cluster_run import clustering
from libs.relabeling import get_relabeling

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    startinit = time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='experiments/cifar10/flow1_resnet18_v2.yaml')
    args = parser.parse_args()
    cfg, logging = init_v2(args.filename)
    doneinit = time()
    logging.info(f"<============> Init time: {round(doneinit - startinit, 2)} seconds")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['master_model_params'].GPUS)
    """- step 1 : We'll train our system with original train set."""
    # cfg['master_model_params'].TEST.pretrained_path = train_function(cfg, step=1, dataset_part=cfg['1st_train_params']['dataset_part'])
    done_firsttrain = time()
    logging.info(f"<============> First training time: {round(done_firsttrain - doneinit, 2)} seconds")

    """- step 2: after that, We extract feature from test set and cluster them by optimal number cluster algorithm."""
    extract_feature(cfg, logging, class_merging=False)
    done_extract = time()
    logging.info(f"<============> Feature extraction time: {round(done_extract - done_firsttrain, 2)} seconds")
    with open(cfg['pretext_params']['fc1_path'], 'rb') as f:
        data = pickle.load(f)
    print('start clustering')
    opt_clst = clustering(cfg, logging, data, org_eval=True)
    done_clustering = time()
    logging.info(f"<============> Clustering time: {round(done_clustering - done_extract, 2)} seconds")

    # relabel data
    print('start relabeling data')
    relabeling = get_relabeling(cfg)(cfg, data)
    relabeling.load_state()
    relabeling.process_relabel()
    done_relabel = time()
    logging.info(f"<============> Relabeling time: {round(done_relabel - done_clustering, 2)} seconds")

    """- step 3: train and valid test set with new labels retrieved from step 2 --> check accuracy."""
    opt_clst = list(set(opt_clst))
    print('best cluster:', opt_clst)
    for clusters in opt_clst:
        # clusters=10
        print('clusters', clusters)

        cfg['master_model_params'].DATASET.NUM_CLASSES = int(clusters)
        cfg['master_model_params'].DATASET.LE_PATH = os.path.join(cfg['relabel_params']['relabel_dir'],
                                                                  str(clusters) + '_new_le.pkl')
        cfg['master_model_params'].DATASET.TRAIN_LIST = os.path.join(cfg['relabel_params']['relabel_dir'],
                                                                     str(clusters) + '_train.pkl')
        cfg['master_model_params'].DATASET.VAL_LIST = os.path.join(cfg['relabel_params']['relabel_dir'],
                                                                   str(clusters) + '_test.pkl')
        cfg['master_model_params'].MODEL.PRETRAINED = True
        cfg['master_model_params'].TRAIN.FINETUNE = ''
        # config.TRAIN.FINETUNE = args.pretrained_path
        # config.TRAIN.BEGIN_EPOCH = 0
        # config.TRAIN.END_EPOCH = 100
        train_function(cfg, step=3,
                       dataset_part=cfg['relabel_params']['dataset_part'])
    done_lasttrain = time()
    logging.info(f"<============> End training time: {round(done_lasttrain - done_relabel, 2)} seconds")


if __name__ == '__main__':
    main()
