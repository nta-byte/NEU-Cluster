import torch as t
from torch.nn import functional as F

from torch import distributions as dist
from torch import nn


class VAE(nn.Module):
    def __init__(self, n_genes, latent_dim, hidden_dims=None, num_classes=10):
        super(VAE, self).__init__()

        self.in_channels = n_genes
        in_channels = n_genes
        modules = []
        self.hidden_dims = hidden_dims if hidden_dims else [256]
        # else:

        # Build Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.ReLU()
                )
                # nn.Linear(in_channels, h_dim)
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        # self.layersize = [512,256,128,64, 32]
        self.latentsize = latent_dim

        self.fc2zmu = nn.Linear(self.hidden_dims[-1], self.latentsize)
        self.fc2zlvar = nn.Linear(self.hidden_dims[-1], self.latentsize)

        # Build Decoder
        modules = []

        self.hidden_dims.reverse()
        in_channels = self.latentsize

        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    # nn.Conv2d(in_channels, out_channels=h_dim,
                    #           kernel_size=3, stride=2, padding=1)
                    # nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                )

            )
            in_channels = h_dim

        self.decoder = nn.Sequential(*modules)
        self.fc10 = nn.Linear(self.hidden_dims[-1], self.in_channels)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.latentsize, num_classes)

    def forward_tail(self, x):
        x = self.fc(x)
        return x

    def encode(self, x):
        h2 = self.encoder(x)
        z_mu = self.fc2zmu(h2)
        z_lvar = self.fc2zlvar(h2)
        return z_mu, z_lvar

    def reparam(self, mu, lvar):
        std = t.exp(0.5 * lvar)
        eps = t.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # h4 = nn.functional.relu(self.fcz2(z))
        # h5 = nn.functional.relu(self.fc21(h4))
        z = self.decoder(z)
        #        x_hat = t.exp(self.fc10(h5))
        x_hat = self.fc10(z)
        return x_hat

    def forward(self, x, **kwargs):
        mu, lvar = self.encode(x.view(-1, self.in_channels))
        z = self.reparam(mu, lvar)

        return {'x_hat': self.decode(z), 'mu': mu, 'lvar': lvar, 'clspred': self.forward_tail(z), 'x': x}

    def loss_function(self, pred, **kwargs):
        # x_hat, x, mu, lvar, clspred, labels
        x_hat = pred['x_hat']
        x = pred['x']
        mu = pred['mu']
        lvar = pred['lvar']
        clspred = pred['clspred']
        labels = kwargs['labels']
        criterion = nn.CrossEntropyLoss()
        clsloss = criterion(clspred, labels)
        # print(x_hat, x)
        bceloss = F.binary_cross_entropy(x_hat.sigmoid(), x, reduction='mean')
        recons_loss_l2 = F.mse_loss(x_hat, x)
        recons_loss_l1 = F.l1_loss(x_hat, x)
        # print(x_hat.shape, x.shape,  x.view(-1, n_genes).shape, lvar.shape)
        # bce = nn.functional.mse_loss(x_hat, x.view(-1, n_genes), reduction='sum')
        kld = -0.5 * t.sum(1 + lvar - mu.pow(2) - lvar.exp())
        return kld * .2 + clsloss * .5 + bceloss * .3  # +recons_loss_l1


class SWAE(nn.Module):

    def __init__(self,
                 in_channels,
                 latent_dim,
                 hidden_dims=None,
                 num_classes=10,
                 reg_weight=100,
                 wasserstein_deg=2.,
                 num_projections=50,
                 projection_dist='normal',
                 **kwargs):
        super(SWAE, self).__init__()

        self.reg_weight = reg_weight
        self.p = wasserstein_deg
        self.num_projections = num_projections
        self.proj_dist = projection_dist
        self.in_channels = in_channels
        # in_channels = n_genes
        modules = []
        self.hidden_dims = hidden_dims if hidden_dims else [1024, 512,256,128]
        # else:

        # Build Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.ReLU()
                )
                # nn.Linear(in_channels, h_dim)
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.latent_dim = latent_dim

        self.fc2zmu = nn.Linear(self.hidden_dims[-1], self.latent_dim)

        # Build Decoder
        modules = []
        # self.decode_hd_dims = self.hidden_dims.copy()
        self.hidden_dims.reverse()
        in_channels = self.latent_dim

        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    # nn.Conv2d(in_channels, out_channels=h_dim,
                    #           kernel_size=3, stride=2, padding=1)
                    # nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                )

            )
            in_channels = h_dim

        self.decoder = nn.Sequential(*modules)
        self.fc10 = nn.Linear(self.hidden_dims[-1], self.in_channels)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.latent_dim, num_classes)

    def forward_tail(self, x):
        x = self.fc(x)
        return x

    def encode(self, x):
        h2 = self.encoder(x)
        z_mu = self.fc2zmu(h2)
        # z_lvar = self.fc2zlvar(h2)
        return z_mu

    def get_random_projections(self, latent_dim: int, num_samples: int) -> t.Tensor:
        """
        Returns random samples from latent distribution's (Gaussian)
        unit sphere for projecting the encoded samples and the
        distribution samples.

        :param latent_dim: (Int) Dimensionality of the latent space (D)
        :param num_samples: (Int) Number of samples required (S)
        :return: Random projections from the latent unit sphere
        """
        if self.proj_dist == 'normal':
            rand_samples = t.randn(num_samples, latent_dim)
        elif self.proj_dist == 'cauchy':
            rand_samples = dist.Cauchy(t.tensor([0.0]),
                                       t.tensor([1.0])).sample((num_samples, latent_dim)).squeeze()
        else:
            raise ValueError('Unknown projection distribution.')

        rand_proj = rand_samples / rand_samples.norm(dim=1).view(-1, 1)
        return rand_proj  # [S x D]

    def compute_swd(self,
                    z: t.Tensor,
                    p: float,
                    reg_weight: float) -> t.Tensor:
        """
        Computes the Sliced Wasserstein Distance (SWD) - which consists of
        randomly projecting the encoded and prior vectors and computing
        their Wasserstein distance along those projections.

        :param z: Latent samples # [N  x D]
        :param p: Value for the p^th Wasserstein distance
        :param reg_weight:
        :return:
        """
        prior_z = t.randn_like(z)  # [N x D]
        device = z.device

        proj_matrix = self.get_random_projections(self.latent_dim,
                                                  num_samples=self.num_projections).transpose(0, 1).to(device)

        latent_projections = z.matmul(proj_matrix)  # [N x S]
        prior_projections = prior_z.matmul(proj_matrix)  # [N x S]

        # The Wasserstein distance is computed by sorting the two projections
        # across the batches and computing their element-wise l2 distance
        w_dist = t.sort(latent_projections.t(), dim=1)[0] - \
                 t.sort(prior_projections.t(), dim=1)[0]
        w_dist = w_dist.pow(p)
        return reg_weight * w_dist.mean()

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        z = args[2]

        batch_size = input.size(0)
        bias_corr = batch_size * (batch_size - 1)
        reg_weight = self.reg_weight / bias_corr

        recons_loss_l2 = F.mse_loss(recons, input)
        recons_loss_l1 = F.l1_loss(recons, input)

        swd_loss = self.compute_swd(z, self.p, reg_weight)

        loss = recons_loss_l2 + recons_loss_l1 + swd_loss
        return loss

    def decode(self, z):
        z = self.decoder(z)
        x_hat = self.fc10(z)
        return x_hat

    def forward(self, x, **kwargs):
        e = self.encode(x.view(-1, self.in_channels))
        # z = self.reparam(mu, lvar)

        # return self.decode(e), self.forward_tail(e), e

        return [self.decode(e), x, e]


if __name__ == '__main__':
    m = SWAE(512, 8)
    print(m)
