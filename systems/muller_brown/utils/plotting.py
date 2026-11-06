import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from utils.potentials import *

plt.rcParams["font.family"] = "Arial"
plt.rcParams['text.usetex'] = False

def compute_grid(eval_fn, limits, step):
    x = torch.arange(limits[0][0], limits[0][1], step, dtype=torch.float32)
    y = torch.arange(limits[1][0], limits[1][1], step, dtype=torch.float32)
    Y, X = torch.meshgrid(y, x, indexing='ij')  # Match NumPy's default behavior

    # Flatten the grid for vectorized evaluation (if possible)
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    # Evaluate function on each point (eval_fn should accept a list or 1D tensor)
    Z = torch.tensor([eval_fn(p) for p in points], dtype=torch.float32)
    Z = Z.view(X.shape)  # Reshape to match the grid

    return X, Y, Z


def plot_contour_map(ax, X, Y, Z, title, limits=None, levels=None, cbar=True, colorlab=None, contours=True, cmap="viridis", fs=8):
    if contours == True:
        ax.contour(X.numpy(), Y.numpy(), Z.numpy(), levels=levels, colors="white", alpha=0.7, linewidths=0.2)
    ct = ax.contourf(X.numpy(), Y.numpy(), Z.numpy(), levels=levels, cmap=cmap)
    if cbar == True:
        cb = plt.colorbar(ct)
        cb.ax.set_title(colorlab, pad=5, fontsize=fs)
        cb.ax.yaxis.label.set_size(fs)
        cb.ax.tick_params(length=0, labelsize=2*fs/3)
        cb.outline.set_visible(False)
    ax.set_xticks([]); ax.set_xlabel(r"$x$", fontsize=fs)
    ax.set_yticks([]); ax.set_ylabel(r"$y$", fontsize=fs)
    # Remove all x and y axis ticks
    ax.set_frame_on(False)

    ax.tick_params(axis='both', pad=2, labelsize=fs)
    if limits != None:
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
    ax.set_title(title, fontweight='bold', fontsize=fs)


def plot_loss(al, log=True):
    fig, ax = plt.subplots(figsize=(7, 3), dpi=300)
    ax.plot(range(len(al.logger["train_loss"])), al.logger["train_loss"])
    ax.set_title("Training loss")
    ax.set_xlabel("epoch")
    if log == True:
        ax.set_yscale('log')
        ax.set_ylabel("log loss")  
    else:
        ax.set_ylabel("loss")    
    return fig



def plot_al_iteration(train_iter, al_logger, sim_logger, n_hidden, D0, batch_size, lvls, limits, save_dir, titlelabel="", show_legend=True, save_fig=True):
    # load time steps between training itereations
    train_step = al_logger["train_steps"][train_iter]
    train_step_prev = al_logger["train_steps"][train_iter-1]
    data = al_logger["train_data"][:len(D0) + (train_iter)*batch_size]
    newdata = data[-batch_size:]
    traj = sim_logger["traj"].detach().numpy()

    # load model iterate
    model_iter = MLPPotential(hidden_dim=n_hidden, n_hidden_layers=1, activation=nn.Tanh, device=torch.device("cpu"))
    model_iter.load_state_dict(torch.load(save_dir + f"model_{train_step}.pt"))

    # evaluate model at points
    model_iter.eval()
    X, Y, V_pred = compute_grid(model_iter.potential_energy, limits, 0.05)

    fig, ax = plt.subplots(figsize=(3.2, 2.5), dpi=600)
    plot_contour_map(ax, X, Y, V_pred, title=f'{titlelabel} iter. {train_iter}', levels=lvls, colorlab="V(x,y)", limits=limits, fs=7)
    ax.plot(traj[train_step_prev:train_step,0], traj[train_step_prev:train_step,1], 'o-', markersize=0.2, linewidth=0.25, color='#440154FF', alpha=0.25, label="trajectory")
    ax.plot(data[:,0].cpu(), data[:,1].cpu(), 'o', markersize=0.5, color="red", alpha=1, label="train data")
    ax.plot(newdata[:,0].cpu(), newdata[:,1].cpu(), 'o', markersize=0.5, color="cyan", alpha=1, label="current data")
    # Remove all x and y axis ticks
    ax.set_xlim(limits[0][0], limits[0][1]-0.1)
    ax.set_ylim(limits[1][0], limits[1][1]-0.1)
    if show_legend == True:
        ax.legend(fontsize='x-small', loc='lower right')
    if save_fig == True:
        fig.savefig(save_dir + f"plot_iter{train_iter}.png")


def al_iteration_animation(train_iter, al_logger, sim_logger, n_hidden, D0, batch_size, lvls, limits, save_dir, titlelabel="", show_legend=True, save_fig=True):
    # load time steps between training iterations
    train_step = al_logger["train_steps"][train_iter]
    train_step_prev = al_logger["train_steps"][train_iter-1]
    data = al_logger["train_data"][:len(D0) + (train_iter)*batch_size]
    newdata = data[-batch_size:]
    traj = sim_logger["traj"].detach().numpy()

    # load model iterate
    model_iter = MLPPotential(hidden_dim=n_hidden, n_hidden_layers=1, activation=nn.Tanh, device=torch.device("cpu"))
    model_iter.load_state_dict(torch.load(save_dir + f"model_{train_step}.pt"))

    # evaluate model at points
    model_iter.eval()
    X, Y, V_pred = compute_grid(model_iter.potential_energy, limits, 0.05)

    fig, ax = plt.subplots(figsize=(3.2, 2.5), dpi=600)
    plot_contour_map(ax, X, Y, V_pred, title=f'{titlelabel} iter. {train_iter}', levels=lvls, colorlab=r"$V(x,y)$", limits=limits, cbar=True, contours=True, fs=5)
    ax.plot(traj[train_step_prev:train_step,0], traj[train_step_prev:train_step,1], 'o-', markersize=0.2, linewidth=0.25, color='#440154FF', alpha=0.25, label="trajectory")
    ax.plot(data[:,0].cpu(), data[:,1].cpu(), 'o', markersize=0.5, color="red", alpha=1, label="train data")
    ax.plot(newdata[:,0].cpu(), newdata[:,1].cpu(), 'o', markersize=0.5, color="cyan", alpha=1, label="current data")
    ax.set_xlim(limits[0][0], limits[0][1]-0.1)
    ax.set_ylim(limits[1][0], limits[1][1]-0.1)
    if show_legend == True:
        ax.legend(fontsize='x-small', loc='lower right')
    if save_fig == True:
        fig.savefig(save_dir + f"animation_snapshot_{train_iter}.png")



def plot_rmse(rmse_e, rmse_f, log=True):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
    fig.tight_layout()
    ax[0].plot(range(len(rmse_e)), rmse_e)
    ax[0].set_title("Test error in energies")
    ax[0].set_xlabel("time step")
    ax[0].set_ylabel("RMSE")

    ax[1].plot(range(len(rmse_f)), rmse_f)
    ax[1].set_title("Test error in forces")
    ax[1].set_xlabel("time step")
    ax[1].set_ylabel("RMSE")

    if log == True:
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")

    return fig



def plot_rmse_over_trials(rmse_e_list, rmse_f_list, col_list, lab_list, tsteps=None, fs=15, log=True):
    Nsim = len(rmse_e_list)
    # compute stats
    e_med = [np.median(rmse_e,axis=0) for rmse_e in rmse_e_list]
    e_25 = [np.percentile(rmse_e, 25, axis=0) for rmse_e in rmse_e_list]
    e_75 = [np.percentile(rmse_e, 75, axis=0) for rmse_e in rmse_e_list]
    f_med = [np.median(rmse_f,axis=0) for rmse_f in rmse_f_list]
    f_25 = [np.percentile(rmse_f, 25, axis=0) for rmse_f in rmse_f_list]
    f_75 = [np.percentile(rmse_f, 75, axis=0) for rmse_f in rmse_f_list]

    rmse = {
        "e_med": e_med,
        "e_25": e_25,
        "e_75": e_75,
        "f_med": f_med,
        "f_25": f_25,
        "f_75": f_75,
    }

    if tsteps is None: # plot all time steps
        T = len(e_med[0])
        tsteps = [list(range(T)) for m in range(Nsim)]
        xlab = "time step"
    elif type(tsteps[0]) == int: # plot only selected time steps
        tsteps = [tsteps for m in range(Nsim)]

    # create figure
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 3.5), dpi=600)
    fig.subplots_adjust(hspace=0.4, wspace=0.2, top=0.9, bottom=0.1, left=0.1, right=0.9)

    # plot energies
    for m in range(Nsim):
        ts = tsteps[m]
        ax[0].plot(range(len(ts)), e_med[m][ts], "o-", markersize=3.5, linewidth=2.5, color=col_list[m], label=lab_list[m])
        ax[0].fill_between(range(len(ts)), e_25[m][ts], e_75[m][ts], color=col_list[m], edgecolor='none', alpha=0.15)
    ax[0].set_title("Test error in energies", fontsize=fs+2)
    ax[0].set_xlabel("training iteration", fontsize=fs)
    ax[0].set_xlim(0,16); ax[0].set_ylim(10**(-1.2), 10**(2))
    ax[0].tick_params(axis='both', pad=2, labelsize=fs-1)
    if log == True:
        ax[0].set_yscale("log")
        ax[0].set_ylabel("log RMSE", fontsize=fs)
    else:
        ax[0].set_ylabel("RMSE", fontsize=fs)

    # plot forces
    for m in range(Nsim):
        ts = tsteps[m]
        ax[1].plot(range(len(ts)), f_med[m][ts], "o-", markersize=3.5, linewidth=2.5, color=col_list[m], label=lab_list[m])
        ax[1].fill_between(range(len(ts)), f_25[m][ts], f_75[m][ts], color=col_list[m], edgecolor='none', alpha=0.15)
    ax[1].set_title("Test error in forces", fontsize=fs+2)
    ax[1].set_xlabel("training iteration", fontsize=fs)
    ax[1].set_xlim(0,16); ax[1].set_ylim(10**(-1), 10**(1))
    ax[1].tick_params(axis='both', pad=2, labelsize=fs-1)
    ax[1].legend(fontsize=fs-2)
    if log == True:
        ax[1].set_yscale("log")
        ax[1].set_ylabel("log RMSE", fontsize=fs)
    else:
        ax[1].set_ylabel("RMSE", fontsize=fs)


    return fig, rmse



def plot_trigger(al_logger, trigger, ttl, ylab):
    trigvals = al_logger["trigger"].detach().numpy()
    T = len(trigvals)
    trainpts = al_logger["train_steps"][1:]
    fig, ax = plt.subplots(figsize=(5, 2), dpi=300)
    ax.plot(trigvals, linewidth=0.75, label="stop criterion")
    ax.hlines(y=trigger.thresh, xmin=0, xmax=T, color="orange",  linewidth=0.75, label="threshold")
    ax.plot(trainpts, trigger.thresh*np.ones(len(trainpts)), "o", markersize=2, color="red", label="train steps")
    ax.set_title(ttl)
    ax.set_ylabel(ylab)
    ax.set_xlabel("time step")
    ax.legend(fontsize="x-small", loc="upper right")
    return fig



def column_boxplots(ax, A, title="", colors=None, labels=None, gap=0.8, width=0.6, showfliers=False):
    """Paper-style log-scale boxplots for each column of A (N x d)."""
    N, d = A.shape
    positions = np.arange(d) * (1 + gap)

    bp = ax.boxplot(
        [A[:, j] for j in range(d)],
        positions=positions,
        widths=width,
        vert=True,
        patch_artist=True,
        showfliers=showfliers,
        boxprops=dict(edgecolor="black", linewidth=1),
        whiskerprops=dict(color="black", linewidth=1),
        capprops=dict(color="black", linewidth=1),
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="gray", markeredgecolor="none")
    )

    # Apply colors
    if colors is None:
        colors = ["#DDDDDD"] * d
    if len(colors) != d:
        raise ValueError("colors must have length equal to number of columns")

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    # set axis labels
    if labels is None:
        labels = np.arange(1, d + 1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    ax.set_xlabel("Sampler")
    ax.set_ylabel("Value")
    ax.set_title(title)

    # set gridlines
    ax.set_yscale('log')
    # Major ticks at powers of 10
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
    # Minor ticks at 2–9 between each power of 10
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1)
    )
    # Grid styling
    ax.grid(which="major", linestyle="dotted", color="gray", linewidth=0.5, alpha=0.4)

    return ax
