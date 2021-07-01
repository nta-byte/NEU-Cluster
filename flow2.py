"""
2. We'll used the whole dataset. consider the dataset have 2*k classes.

- step 1: make new labels for the datatset by merging 2 classes into 1 randomly -> we got k classes.

- step 2: train classifier with k classes.

- step 3: extract feature and cluster the datatset by optimal number cluster algorithm.

- step 4: train and valid with new labels retrieved from step 3 --> check accuracy.
"""

import os
import pickle
from time import time
import argparse

from train import train_function, train_function2
from libs.utils.yaml_config import init_v2
# from training.config import update_config, config
from libs.pretext.create_pretext_pytorch import extract_feature
from libs.clustering.cluster_run import clustering
from libs.relabeling import get_relabeling


def main():
    startinit = time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='experiments/cifar10/flow2_resnet18_v2.yaml')
    args = parser.parse_args()
    cfg, logging = init_v2(args.filename)
    print(cfg['master_model_params'].GPUS)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['master_model_params'].GPUS)
    doneinit = time()
    logging.info(f"<============> Init time: {round(doneinit - startinit, 2)} seconds")
    """
    + step 1: make new labels for the datatset by merging 2 classes into 1 randomly -> we got k classes.
    + step 2: train classifier with k classes."""
    cfg['master_model_params'].TEST.pretrained_path = train_function2(cfg,
                                                                      dataset_part=cfg['1st_train_params']['dataset_part'])
    done_firsttrain = time()
    logging.info(f"<============> First training time: {round(done_firsttrain - doneinit, 2)} seconds")

    """- step 3: extract feature and cluster the datatset by optimal number cluster algorithm."""
    extract_feature(cfg, logging, class_merging=True)
    done_extract = time()
    logging.info(f"<============> Feature extraction time: {round(done_extract - done_firsttrain, 2)} seconds")
    with open(cfg['pretext_params']['fc1_path'], 'rb') as f:
        data = pickle.load(f)
    logging.info('start clustering')
    opt_clst = clustering(cfg, logging, data, org_eval=True)
    done_clustering = time()
    logging.info(f"<============> Clustering time: {round(done_clustering - done_extract, 2)} seconds")
    # logging.info(f'Optimal number of clusters: {opt_clst}')
    opt_clst = list(set(opt_clst))
    # relabel data
    logging.info('start relabeling data')
    relabeling = get_relabeling(cfg)(cfg, data)
    relabeling.load_state()
    relabeling.process_relabel()
    del relabeling
    done_relabel = time()
    logging.info(f"<============> Relabeling time: {round(done_relabel - done_clustering, 2)} seconds")

    """- step 4: train and valid with new labels retrieved from step 3 --> check accuracy."""
    # train on trainset and validate on test set
    for clusters in opt_clst:
        # clusters = 10
        cfg['master_model_params'].DATASET.NUM_CLASSES = int(clusters)
        cfg['master_model_params'].DATASET.LE_PATH = os.path.join(cfg['relabel_params']['relabel_dir'],
                                                                  str(clusters) + '_new_le.pkl')
        cfg['master_model_params'].DATASET.TRAIN_LIST = os.path.join(cfg['relabel_params']['relabel_dir'],
                                                                     str(clusters) + '_train.pkl')
        cfg['master_model_params'].DATASET.VAL_LIST = os.path.join(cfg['relabel_params']['relabel_dir'],
                                                                   str(clusters) + '_test.pkl')
        cfg['master_model_params'].MODEL.PRETRAINED = True
        cfg['master_model_params'].TRAIN.FINETUNE = ''
        # cfg['master_model_params'].TRAIN.OPTIMIZER = 'sgd'
        # cfg['master_model_params'].TRAIN.LR = 0.01
        # cfg['master_model_params'].TRAIN.BEGIN_EPOCH = 0
        # cfg['master_model_params'].TRAIN.END_EPOCH = 20
        train_function(cfg, step=3, dataset_part=cfg['relabel_params']['dataset_part'])
    done_lasttrain = time()
    logging.info(f"<============> Total Running time: {round(done_lasttrain - startinit, 2)} seconds")


if __name__ == '__main__':
    main()
