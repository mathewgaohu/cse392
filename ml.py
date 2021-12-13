from time import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation

import torch
import torch.nn as nn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom

device = 'cuda' if torch.cuda.is_available() else 'cpu'

X = torch.tensor(np.load("X.npy"), dtype=torch.float)
Y = torch.tensor(np.load("Y.npy"), dtype=torch.float)

test_split = 10000
print(X.shape)
print(Y.shape)

ndim_tot = 50
ndim_x = 6
ndim_y = 40
ndim_z = 0


def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, 512), nn.ReLU(),
                         nn.Linear(512, c_out))


nodes = [InputNode(ndim_tot, name='input')]

for k in range(8):
    nodes.append(Node(nodes[-1],
                      GLOWCouplingBlock,
                      {'subnet_constructor': subnet_fc, 'clamp': 2.0},
                      name=F'coupling_{k}'))
    nodes.append(Node(nodes[-1],
                      PermuteRandom,
                      {'seed': k},
                      name=F'permute_{k}'))

nodes.append(OutputNode(nodes[-1], name='output'))

model = ReversibleGraphNet(nodes, verbose=False)

# Training parameters
n_epochs = 200
n_its_per_epoch = 10
batch_size = 1600

lr = 1e-3
l2_reg = 2e-5

y_noise_scale = 0.
zeros_noise_scale = 0.

# relative weighting of losses:
lambd_predict = 1.
lambd_latent = 1.
lambd_rev = 5.

pad_x = torch.zeros(batch_size, ndim_tot - ndim_x)
pad_yz = torch.zeros(batch_size, ndim_tot - ndim_y - ndim_z)

trainable_parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_parameters, lr=lr, betas=(0.8, 0.9),
                             eps=1e-6, weight_decay=l2_reg)


def fit(input, target):
    return torch.mean((input - target) ** 2)


loss_backward = fit
loss_latent = None
loss_fit = fit

test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X[:test_split], Y[:test_split]),
    batch_size=batch_size, shuffle=True, drop_last=True)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X[test_split:], Y[test_split:]),
    batch_size=batch_size, shuffle=True, drop_last=True)


def train(i_epoch=0):
    model.train()

    l_tot = 0
    batch_idx = 0

    # loss_factor = min(1., 2. * 0.002**(1. - (float(i_epoch) / n_epochs)))
    loss_factor = 1

    for x, y in train_loader:
        batch_idx += 1
        if batch_idx > n_its_per_epoch:
            break

        x, y = x.to(device), y.to(device)

        y_clean = y.clone()
        pad_x = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                ndim_x, device=device)
        pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                 ndim_y - ndim_z, device=device)

        y += y_noise_scale * torch.randn(batch_size, ndim_y, dtype=torch.float, device=device)

        x, y = (torch.cat((x, pad_x), dim=1),
                torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz, y),
                          dim=1))

        optimizer.zero_grad()

        # Forward step:

        output, _ = model(x)

        # Shorten output, and remove gradients wrt y, for latent loss
        y_short = torch.cat((y[:, :ndim_z], y[:, -ndim_y:]), dim=1)

        l = lambd_predict * loss_fit(output[:, ndim_z:], y[:, ndim_z:])

        l_tot += l.data.item()

        l.backward()

        # Backward step:
        pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                 ndim_y - ndim_z, device=device)
        y = y_clean + y_noise_scale * torch.randn(batch_size, ndim_y, device=device)

        orig_z_perturbed = (output.data[:, :ndim_z] + y_noise_scale *
                            torch.randn(batch_size, ndim_z, device=device))
        y_rev = torch.cat((orig_z_perturbed, pad_yz,
                           y), dim=1)
        y_rev_rand = torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz,
                                y), dim=1)

        output_rev, _ = model(y_rev, rev=True)
        output_rev_rand, _ = model(y_rev_rand, rev=True)

        l_rev = (
                lambd_rev
                * loss_factor
                * loss_backward(output_rev_rand[:, :ndim_x],
                                x[:, :ndim_x])
        )

        l_rev += lambd_predict * loss_fit(output_rev, x)

        l_tot += l_rev.data.item()
        l_rev.backward()

        for p in model.parameters():
            if p.requires_grad:
                p.grad.data.clamp_(-15.00, 15.00)

        optimizer.step()

    return l_tot / batch_idx


# train.
x_true = torch.tensor(np.load("xtrue.npy").reshape(1, -1), dtype=torch.float)
x_true = torch.cat((x_true, torch.zeros(1, ndim_tot - ndim_x)), dim=1)
x_true = x_true.to(device)

y_true = torch.tensor(np.load("ytrue.npy").reshape(1, -1), dtype=torch.float)
y_true = torch.cat([torch.randn(1, ndim_z), zeros_noise_scale * torch.zeros(1, ndim_tot - ndim_y - ndim_z), y_true],
                   dim=1)
y_true = y_true.to(device)

trace_x = []
trace_y = []
trace_l = []
try:
    t_start = time()
    for i_epoch in tqdm(range(n_epochs), ascii=True, ncols=80):
        trace_l.append(train(i_epoch))

        # plot fwd and bwd
        rev_x = model(y_true, rev=True)[0].cpu().data.numpy()[0][:ndim_x]
        pred_y = model(x_true)[0].to(device).data.numpy()[0][-ndim_y:]

        trace_x.append(rev_x)
        trace_y.append(pred_y)

    np.save("trace_l.npy", np.array(trace_l))
    np.save("trace_x.npy", np.array(trace_x))
    np.save("trace_y.npy", np.array(trace_y))


except KeyboardInterrupt:
    pass
finally:
    print(f"\n\nTraining took {(time() - t_start) / 60:.2f} minutes\n")

# %%
x_true_numpy = np.load("xtrue.npy")
y_true_numpy = np.load("ytrue.npy")
trace_x = np.load("trace_x.npy")
trace_y = np.load("trace_y.npy")
trace_l = np.load("trace_l.npy")

fig = plt.figure(figsize=(12, 8))
grid = plt.GridSpec(4, 3, wspace=0.5, hspace=0.5)
ax0 = plt.subplot(grid[:, 0])
axes = [plt.subplot(grid[i, 1:]) for i in range(4)]

ttl = ax0.text(0.5, 1.05, "epoch = {:3d}".format(0),
               size=plt.rcParams["axes.titlesize"],
               ha="center", transform=ax0.transAxes, )

zz = np.linspace(0, 1, 6)
plt.sca(ax0)
plt.plot(x_true_numpy, zz, "k", label="true")
L0, = plt.plot([], [], label="trained")
ax0.set_ylim(0, 1)
ax0.set_xlim(-0.01, 0.4)
ax0.yaxis.set_inverted(True)
point_s = ax0.plot(0,
                   0.2,
                   marker='*',
                   linestyle='',
                   markersize=8,
                   label="source")
point_r = ax0.plot(np.zeros(4),
                   np.array([0.4, 0.6, 0.8, 1.]),
                   marker='^',
                   linestyle='',
                   markersize=8,
                   label="receiver")
plt.legend(loc="upper right")

observation_times = np.linspace(0.2, 2., 10)
data = y_true_numpy.reshape(-1, 4)
LL = []
for i, ax in enumerate(axes):
    plt.sca(ax)
    plt.plot(observation_times, data[:, i], 'k-', label='true')
    L, = plt.plot([], [], label="trained")
    LL.append(L)
    ax.set_xlim(0, 2)
    ax.set_ylim(data.min(), data.max())
    plt.legend(loc="upper left")


def update(frame):
    L0.set_data(trace_x[frame, :], zz)
    ttl.set_text("epoch = {:3d}, Lx+Ly={:8.4f}".format(frame, trace_l[frame]))

    data = trace_y[frame, :].reshape(-1, 4)
    for i, ax in enumerate(axes):
        LL[i].set_data(observation_times, data[:, i])

    return L0, ttl, *LL


ani = FuncAnimation(fig, update, frames=np.arange(trace_x.shape[0]), blit=True)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
ani.save('ML.mp4', writer=writer)
plt.show()
