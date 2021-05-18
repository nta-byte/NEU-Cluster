import torch
import numpy as np
import os
import os.path as osp
import yaml
import argparse
from libs.clustering.VAE.data_loader import VAEDataset
from libs.clustering.VAE.models.models import VAE
from libs.pretext.utils import load_state


def vae_reduce_dimension(config, data=None, dev=None):
    if dev is None:
        dev = torch.device('cuda:{}'.format(0))

    n_genes = config['model_params']['n_genes']
    if data is not None:
        dataset = VAEDataset(datain=data)
    else:
        dataset = VAEDataset(path_datain=config['exp_params']['train_data_path'])
    vaeloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=config['trainer_params']['batch_size'],
                                            shuffle=False)
    vae_model = VAE(n_genes, latent_dim=config['model_params']['latent_dim']).to(dev)
    vae_model = load_state(config['infer']['weight_path'], vae_model)
    vae_model.eval()
    output = []
    with torch.no_grad():
        for idx, batch in enumerate(vaeloader):
            if idx >= len(vaeloader):
                break
            data = batch[0].to(dev)
            mu, lvar = vae_model.encode(data.view(-1, n_genes))
            z = vae_model.reparam(mu, lvar)
            out = z.cpu().detach().numpy()
            output.append(out)
    output = np.concatenate(output, axis=0)
    if len(output.shape) > 2:
        output = output.reshape((output.shape[0], output.shape[1]))
    x = output
    return x


# if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    # parser.add_argument('--config', '-c',
    #                     dest="filename",
    #                     metavar='FILE',
    #                     help='path to the config file',
    #                     default='experiments/simplevae.yaml')
    #
    # args = parser.parse_args()
    # with open(args.filename, 'r') as file:
    #     try:
    #         config = yaml.safe_load(file)
    #     except yaml.YAMLError as exc:
    #         print(exc)
    #
    # print(config)
    # x = reduce_dimension(config)
