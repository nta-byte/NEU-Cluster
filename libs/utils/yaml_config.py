import yaml
import os
import argparse

import logging


def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg


def init(config_path):
    parser = argparse.ArgumentParser()
    config = yaml_config_hook(config_path)

    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    dataset = config['dataset']
    config_dataset = yaml_config_hook("config/dataset/{}.yaml".format(dataset))
    for k, v in config_dataset.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    framework = 'pytorch' if args.framework == 'pytorch' else 'tf'
    use_histeq = "histeq" if args.use_histeq else "nohisteq"
    pca_whitten = "whitten" if args.pca_whitten else "nowhitten"
    pretrained = "pretrain" if args.pretrained_path else "transfer"
    fc1_file_name = f'{args.model}_{pretrained}_fc1_features_std_{framework}_{use_histeq}.pickle'
    args.fc1_dir = os.path.join(args.fc1_dir, fc1_file_name)
    output_dir_name = f"{dataset}_{args.model}_{pretrained}_PCA{args.pca_component}_kmeanNinit{args.kmeans_n_init}_{framework}_{use_histeq}_{pca_whitten}"
    args.save_dir = os.path.join(args.save_dir, output_dir_name)

    args.kmeans_k_cache_path = os.path.join(args.save_dir, args.kmeans_k_cache_path)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # Init logging
    logname = os.path.join(args.save_dir, 'log_train.txt')
    handlers = [logging.FileHandler(logname), logging.StreamHandler()]
    logging.basicConfig(
        handlers=handlers,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(args)
    return args, logging
