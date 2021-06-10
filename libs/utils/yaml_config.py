import yaml
import os
import argparse
import json
import logging
from pygments import highlight, lexers, formatters
from training.config import update_config_from_yaml_config, config, update_config_from_file


def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        print(cfg.get("defaults", []))
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg


def yaml_config_hook_v2(config_file):
    with open(config_file, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    return config


def init(config_path):
    parser = argparse.ArgumentParser()
    config = yaml_config_hook(config_path)

    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    dataset = config['dataset']

    args = parser.parse_args()
    framework = 'pytorch' if args.framework == 'pytorch' else 'tf'
    use_histeq = "histeq" if args.use_histeq else "nohisteq"
    pca_whitten = "whitten" if args.pca_whitten else "nowhitten"
    if args.reduce_dimension == 'pca':
        reduce_dimension = f"reduce_dim_PCA{args.pca_component}"
    else:
        reduce_dimension = f"reduce_dim_{args.reduce_dimension.upper()}"
        if args.reduce_dimension == 'vae':
            with open(args.vae_cfg, 'r') as file:
                try:
                    vaeconfig = yaml.safe_load(file)
                except yaml.YAMLError as exc:
                    print(exc)
            reduce_dimension += f'_latent_dim_{vaeconfig["model_params"]["latent_dim"]}'
    pretrained = "pretrain" if args.pretrained_path else "transfer"
    output_dir_name = f"{dataset}_{args.model}_{pretrained}_{reduce_dimension}_kmeanNinit{args.kmeans_n_init}_{framework}_{use_histeq}_{pca_whitten}"
    args.save_dir = os.path.join(args.save_dir, output_dir_name)

    fc1_file_name = f'{args.model}_{pretrained}_fc1_features_std_{framework}_{use_histeq}.pickle'
    args.fc1_dir = os.path.join(args.save_dir, args.fc1_dir)
    args.fc1_path = os.path.join(args.fc1_dir, fc1_file_name)
    if args.reduce_dimension == 'vae':
        args.fc1_path_vae = os.path.join(args.fc1_dir,
                                         f'{args.model}_{pretrained}_fc1_features_std_{framework}_{use_histeq}_vae_traindata.pickle')
    args.save_img_1st_step_dir = os.path.join(args.save_dir, args.save_img_1st_step_dir)
    args.save_first_train = os.path.join(args.save_dir, args.save_first_train)
    args.le_path = os.path.join(args.fc1_dir, args.le_path)
    args.label_transform_path = os.path.join(args.fc1_dir, args.label_transform_path)

    args.kmeans_k_cache_path = os.path.join(args.save_dir, args.kmeans_k_cache_path)
    args.relabel_dir = os.path.join(args.save_dir, args.relabel_dir)
    args.last_train_dir = os.path.join(args.relabel_dir, args.last_train_dir)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.relabel_dir):
        os.makedirs(args.relabel_dir)
    if not os.path.exists(args.last_train_dir):
        os.makedirs(args.last_train_dir)
    if not os.path.exists(args.fc1_dir):
        os.makedirs(args.fc1_dir)
    if not os.path.exists(args.save_img_1st_step_dir):
        os.makedirs(args.save_img_1st_step_dir)
    if not os.path.exists(args.save_first_train):
        os.makedirs(args.save_first_train)

    # Init logging
    logname = os.path.join(args.save_dir, 'log_train.txt')
    handlers = [logging.FileHandler(logname), logging.StreamHandler()]
    logging.basicConfig(
        handlers=handlers,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(args)
    return args, logging


def init_v2(config_path):
    # parser = argparse.ArgumentParser()
    cfg = yaml_config_hook_v2(config_path)
    # print(cfg)
    cfg['master_model_params'] = update_config_from_yaml_config(config, cfg['master_model_params'])
    # print(cfg['master_model_params'])
    # for k, v in config.items():
    #     parser.add_argument(f"--{k}", default=v, type=type(v))
    dataset = cfg['master_model_params'].DATASET.DATASET
    model = cfg['master_model_params'].MODEL.NAME.upper()

    # args = parser.parse_args()
    framework = 'pytorch' if cfg['general']['framework'] == 'pytorch' else 'tf'
    use_histeq = "histeq" if cfg['master_model_params'].DATASET.use_histeq else "nohisteq"
    # pca_whitten = "whitten" if args.pca_whitten else "nowhitten"
    reduce_dimension = f"reduce_dim_{cfg['reduce_dimension_params']['type'].upper()}"
    if cfg['reduce_dimension_params']['type'] == 'umap':
        reduce_dimension += f"_dims_{cfg['reduce_dimension_params']['umap_params']['dims']}"
    elif cfg['reduce_dimension_params']['type'] == 'pca':
        reduce_dimension += f"_dims_{cfg['reduce_dimension_params']['pca_params']['dims']}_pca_whitten{cfg['reduce_dimension_params']['pca_params']['whitten']}"
    elif cfg['reduce_dimension_params']['type'] == 'vae':
        reduce_dimension += f"_dims_{cfg['reduce_dimension_params']['vae_params']['model_params']['latent_dim']}"

    # pretrained = "pretrain" if args.pretrained_path else "transfer"
    output_dir_name = f"{dataset}_{model}_{reduce_dimension}_kmeanNinit{cfg['clustering_params']['kmean']['n_init']}_{framework}_{use_histeq}"
    cfg['general']['save_dir'] = os.path.join(cfg['general']['save_dir'], output_dir_name)

    fc1_file_name = f'fc1_features_std.pickle'
    cfg['pretext_params']['fc1_dir'] = os.path.join(cfg['general']['save_dir'], cfg['pretext_params']['fc1_dir'])
    cfg['pretext_params']['fc1_path'] = os.path.join(cfg['pretext_params']['fc1_dir'], fc1_file_name)
    cfg['pretext_params']['le_path'] = os.path.join(cfg['pretext_params']['fc1_dir'], cfg['pretext_params']['le_path'])
    cfg['pretext_params']['label_transform_path'] = os.path.join(cfg['pretext_params']['fc1_dir'],
                                                                 cfg['pretext_params']['label_transform_path'])

    if cfg['reduce_dimension_params']['type'] == 'vae':
        fc1_traindata_vae_file_name = f'fc1_features_std_vae_traindata.pickle'
        cfg['pretext_params']['fc1_path_traindata_vae'] = os.path.join(cfg['fc1_dir'], fc1_traindata_vae_file_name)

    cfg['general']['save_cluster_visualization'] = os.path.join(cfg['general']['save_dir'],
                                                                cfg['general']['save_cluster_visualization'])
    cfg['general']['first_train_dir'] = os.path.join(cfg['general']['save_dir'], cfg['general']['first_train_dir'])

    cfg['clustering_params']['kmean']['cache_path'] = os.path.join(cfg['general']['save_dir'],
                                                                   cfg['clustering_params']['kmean']['cache_path'])
    cfg['relabel_params']['relabel_dir'] = os.path.join(cfg['general']['save_dir'],
                                                        cfg['relabel_params']['relabel_dir'])
    cfg['general']['last_train_dir'] = os.path.join(cfg['general']['save_dir'],
                                                    cfg['general']['last_train_dir'])

    if not os.path.exists(cfg['general']['save_dir']):
        os.makedirs(cfg['general']['save_dir'])
    if not os.path.exists(cfg['relabel_params']['relabel_dir']):
        os.makedirs(cfg['relabel_params']['relabel_dir'])
    if not os.path.exists(cfg['general']['last_train_dir']):
        os.makedirs(cfg['general']['last_train_dir'])
    if not os.path.exists(cfg['pretext_params']['fc1_dir']):
        os.makedirs(cfg['pretext_params']['fc1_dir'])
    if not os.path.exists(cfg['general']['save_cluster_visualization']):
        os.makedirs(cfg['general']['save_cluster_visualization'])
    if not os.path.exists(cfg['general']['first_train_dir']):
        os.makedirs(cfg['general']['first_train_dir'])

    # Init logging
    logname = os.path.join(cfg['general']['save_dir'], 'log_train.txt')
    handlers = [logging.FileHandler(logname), logging.StreamHandler()]
    logging.basicConfig(
        handlers=handlers,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    # logging.info(args)
    json_object = json.dumps(cfg, indent=4)
    colorful_json = highlight(json_object, lexers.JsonLexer(), formatters.TerminalFormatter())
    logging.info(colorful_json)
    return cfg, logging


def new_init(config_path):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='../../experiments/cifar10/flow2_resnet18_v2.yaml')
    args = parser.parse_args()
    config, logging = init_v2(args.filename)
    # update_config(config, args)
