"""
Step1 ) Train - 50,000 w/ noise is used for first Training -> retrieve trained model. ==> M1
Step2 ) Test - 10,000 w/ original gt labels is used for clustering w/ extracting features
    extract features using M1
    clustering => y_gt means original gt labels in  acc = (y_pred_ == y_gt).sum() / len(y_gt)
    get clustering evaluation metrics
==================================> step1 and step2 are the same as before.

Step2-1) Relabeling
    relabeling Train - 50,000 :  w/ noise =>>   w/ clustered labels
Step3 ) Step2-1( Train - 50,000 w/ clustered labels ) is used for second training  ==> M2
    Transfer learning based on M1
Step4 ) Test - 10,000 w/ original gt labels is used for clustering w/ extracting features
    extract features using M2
    clustering => y_gt means original gt labels in  acc = (y_pred_ == y_gt).sum() / len(y_gt)
    get clustering evaluation metrics
"""

import os
import pickle
import torch
from time import time

from train_first import train_function, train_function3
from libs.utils.yaml_config import init
from training.config import update_config, config
from create_pretext_pytorch import extract_feature_flow4
from cluster_run import clustering_flow4
from libs.relabeling.cifar10 import Relabel_flow4

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    startinit = time()
    args, logging = init("experiments/cifar10/flow4_resnet50_32.yaml")
    update_config(config, args)
    doneinit = time()
    logging.info(f"<============> Init time: {round(doneinit - startinit, 2)} seconds")
    """
    Step1 ) Train - 50,000 w/ noise is used for first Training -> retrieve trained model. ==> M1.
    """
    args.cluster_dataset = 'train'
    args.pretrained_path = train_function3(args, config)
    done_firsttrain = time()
    logging.info(f"<============> First training time: {round(done_firsttrain - doneinit, 2)} seconds")

    """Step2 ) Test - 10,000 w/ original gt labels is used for clustering w/ extracting features"""
    # args.cluster_dataset = 'test'
    extract_feature_flow4(args, config, logging, active_data='test')
    done_extract = time()
    logging.info(f"<============> Feature extraction time: {round(done_extract - done_firsttrain, 2)} seconds")
    # fc1_test_file =
    fc1_test_file = os.path.join(args.fc1_dir, f'{args.model}_fc1_test_features_std.pickle')
    with open(fc1_test_file, 'rb') as f:
        data_test = pickle.load(f)
    logging.info('start clustering')
    clustering_flow4(args, logging, data_test, org_eval=True, active_data='test')
    done_clustering = time()
    logging.info(f"<============> Clustering time: {round(done_clustering - done_extract, 2)} seconds")

    """Step2-1) Relabeling"""
    args.cluster_dataset = 'train'
    extract_feature_flow4(args, config, logging, active_data='train')
    done_extract = time()
    logging.info(f"<============> Feature extraction time: {round(done_extract - done_firsttrain, 2)} seconds")
    # fc1_train_file = f'{args.model}_fc1_train_features_std.pickle'
    fc1_train_file = os.path.join(args.fc1_dir, f'{args.model}_fc1_train_features_std.pickle')
    with open(fc1_train_file, 'rb') as f:
        data_train = pickle.load(f)
    logging.info('start clustering')
    opt_clst = clustering_flow4(args, logging, data_train, org_eval=True, active_data='train')
    done_clustering = time()
    logging.info(f"<============> Clustering time: {round(done_clustering - done_extract, 2)} seconds")
    opt_clst = list(set(opt_clst))

    logging.info('start relabeling data')
    relabeling = Relabel_flow4(args, data_train)
    relabeling.load_state()
    relabeling.process_relabel()
    done_relabel = time()
    logging.info(f"<============> Relabeling time: {round(done_relabel - done_clustering, 2)} seconds")

    """Step3 ) Step2-1( Train - 50,000 w/ clustered labels ) is used for second training  ==> M2"""
    # train on trainset and validate on test set
    for clusters in opt_clst:
        # clusters = 10
        config.DATASET.NUM_CLASSES = int(clusters)
        config.DATASET.LE_PATH = os.path.join(args.relabel_dir, str(clusters) + '_new_le.pkl')
        config.DATASET.TRAIN_LIST = os.path.join(args.relabel_dir, str(clusters) + '_train.pkl')
        config.DATASET.VAL_LIST = os.path.join(args.relabel_dir, str(clusters) + '_test.pkl')
        config.MODEL.PRETRAINED = True
        config.TRAIN.FINETUNE = ''
        args.pretrained_path = train_function(args, config, step=3)
        # args.pretrained_path = "output_save_flow4/cifar10_RESNET18_pretrain_reduce_dim_NONE_kmeanNinit2_pytorch_histeq_whitten/relabel/output_training/train_10_cluster/resnet18-best.pth"
        """Step4 ) Test - 10,000 w/ original gt labels is used for clustering w/ extracting features"""
        args.fc1_dir = args.relabel_dir
        extract_feature_flow4(args, config, logging, active_data='test')
        fc1_test_file = os.path.join(args.fc1_dir, f'{args.model}_fc1_test_features_std.pickle')
        with open(fc1_test_file, 'rb') as f:
            data_test = pickle.load(f)
        logging.info('start clustering')
        clustering_flow4(args, logging, data_test, org_eval=True, active_data='test')
        done_clustering = time()
        logging.info(f"<============> Clustering time: {round(done_clustering - done_extract, 2)} seconds")
    done_lasttrain = time()
    logging.info(f"<============> End Training time: {round(done_lasttrain - done_relabel, 2)} seconds")


if __name__ == '__main__':
    main()
