import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn import init
import argparse
import os
from sklearn.model_selection import train_test_split
import glob
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from torchsummary import summary


class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv1d(16, 16, 8, 2, padding=3)
        self.conv3 = nn.Conv1d(16, 32, 8, 2, padding=3)
        self.conv4 = nn.Conv1d(32, 32, 8, 2, padding=3)
        self.fc1 = nn.Linear(32 * 21, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc21 = nn.Linear(16, z_dim)
        self.fc22 = nn.Linear(16, z_dim)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(32)
        self.bn5 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.dropout(x, 0.3)
        x = self.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.dropout(x, 0.3)
        x = self.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.dropout(x, 0.3)
        x = self.relu(self.conv4(x))
        x = self.bn4(x)
        x = F.dropout(x, 0.3)
        x = x.view(-1, 672)
        x = self.relu(self.fc1(x))
        x = self.bn5(x)
        x = F.dropout(x, 0.5)
        x = self.relu(self.fc2(x))
        z_loc = self.fc21(x)
        z_scale = self.fc22(x)
        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, 672)
        self.conv1 = nn.ConvTranspose1d(32, 32, 8, 2, padding=3)
        self.conv2 = nn.ConvTranspose1d(32, 32, 8, 2, padding=3)
        self.conv3 = nn.ConvTranspose1d(32, 16, 8, 2, padding=3)
        self.conv4 = nn.ConvTranspose1d(16, 16, 8, 2, padding=3)
        self.conv5 = nn.ConvTranspose1d(16, 1, 7, 1, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()

    def forward(self, z):
        z = self.relu(self.fc1(z))
        # z = F.dropout(z, 0.3)
        z = z.view(-1, 32, 21)
        z = self.relu(self.conv1(z))
        z = self.bn1(z)
        # z = F.dropout(z, 0.3)
        z = self.relu(self.conv2(z))
        z = self.bn2(z)
        # z = F.dropout(z, 0.3)
        z = self.relu(self.conv3(z))
        z = self.bn3(z)
        # z = F.dropout(z, 0.3)
        z = self.relu(self.conv4(z))
        z = self.bn4(z)
        # z = F.dropout(z, 0.3)
        z = self.conv5(z)
        recon = torch.sigmoid(z)
        return recon


class VAE(nn.Module):
    def __init__(self, z_dim=2):
        super(VAE, self).__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
        self.cuda()
        self.z_dim = z_dim

    def reparameterize(self, z_loc, z_scale):
        std = z_scale.mul(0.5).exp_()
        epsilon = torch.randn(*z_loc.size()).to(device)
        z = z_loc + std * epsilon
        return z


device = torch.device("cuda:0")
batch_size = 32

train_ds = TensorDataset(X_train, y_train)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_ds = TensorDataset(X_test, y_test)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

vae = VAE()

summary(vae.encoder, (1, 336))
summary(vae.decoder, (1, 2))

optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)


# optimizer = torch.optim.RMSprop(vae.parameters(), lr=0.001, alpha=0.9)

def loss_fn(recon_x, x, z_loc, z_scale):
    BCE = F.mse_loss(recon_x, x, size_average=False) * 100
    KLD = -0.5 * torch.sum(1 + z_scale - z_loc.pow(2) - z_scale.exp())
    return BCE + KLD


for epoch in range(1000):
    for x, _ in train_dl:
        x = x.cuda()
        z_loc, z_scale = vae.encoder(x)
        z = vae.reparameterize(z_loc, z_scale)
        recon = vae.decoder(z)
        loss = loss_fn(recon, x, z_loc, z_scale)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    vae.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(test_dl):
            x = x.cuda()
            z_loc, z_scale = vae.encoder(x)
            z = vae.reparameterize(z_loc, z_scale)
            recon = vae.decoder(z)
            test_loss = loss_fn(recon, x, z_loc, z_scale)
    normalizer_test = len(test_dl.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    if epoch == 0:
        loss_test_history = total_epoch_loss_test.item()
        patience = 0
    else:
        loss_test_history = np.append(loss_test_history, total_epoch_loss_test.item())

    if total_epoch_loss_test.item() < 0.000001 + np.min(loss_test_history):
        patience = 0
        torch.save(vae.decoder.state_dict(), "/home/ragan/pytorch_cnn/best_decoder_model.pt")
        torch.save(vae.encoder.state_dict(), "/home/ragan/pytorch_cnn/best_encoder_model.pt")
    else:
        patience += 1

    print(epoch, patience, total_epoch_loss_test.item(), np.min(loss_test_history))

    if patience == 32:
        break

# This is just to visualize the outputs for myself
X_enc, _ = vae.encoder(X_test)
recon = vae.decoder(X_enc)
X_enc = X_enc.cpu().detach().numpy()
y_enc = y_test.cpu().detach().numpy()
# y_enc = np.array(([np.argmax(l) for l in y_enc]))

for i in range(7):
    plt.scatter(X_enc[y_enc == i][:, 0], X_enc[y_enc == i][:, 1], label=f"{i}")
plt.legend()
plt.show()

X_cpu = X_test.cpu()
X_numpy = X_cpu.detach().numpy()

recon_cpu = recon.cpu()
recon_numpy = recon_cpu.detach().numpy()


def key_event(e):
    global curr_pos, con

    if e.key == "right":
        curr_pos = curr_pos + 1
    elif e.key == "left":
        curr_pos = curr_pos - 1

    else:
        return

    curr_pos = curr_pos % len(X_numpy)
    ax.cla()
    ax.set_title(curr_pos)
    ax.plot(X_numpy[curr_pos, 0], label="original")
    plt.plot(recon_numpy[curr_pos, 0], label="decoded")
    fig.canvas.draw()


curr_pos = 0
fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', key_event)
ax = fig.add_subplot(111)
ax.plot(X_numpy[curr_pos, 0], label="original")
ax.set_title(f"{curr_pos}")
plt.plot(recon_numpy[curr_pos, 0], label="decoded")
plt.show()
