import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
from dataloader.augmentations import MixUp
import torch


def forward(model, data, criterion, device):
    yields = data['yield'].to(device)
    s2_bands = data['s2_bands'].to(device)
    clim_bands = data['clim_bands'].to(device)
    soil = data['soil_data'].to(device)
    year = data['year'].to(device)
    prediction = model(s2_bands, clim_bands, soil, year).reshape(yields.shape)
    loss = criterion(yields, prediction)
    return loss, prediction


def evaluate(model, dataloader, device, criterion):
    losses = []
    model.eval()
    with torch.no_grad():
        val_iter = iter(dataloader)
        for i in range(len(dataloader)):
            data = next(val_iter)
            loss, _ = forward(model, data, criterion, device)
            losses.append(loss.item())
    return np.mean(losses)


def train_model(model,
                train_dataloader,
                val_dataloader,
                device,
                optimizer,
                criterion,
                scheduler,
                cfg_data,
                cfg_model):
    losses_train, losses_train_mean = [], []
    losses_val = []
    best_val_loss = 1e6
    p_mixup = train_dataloader.dataset.p_mup

    tr_it = iter(train_dataloader)
    progress_bar = tqdm(range(cfg_model['train_params']['n_iters']))

    for i in progress_bar:
        try:
            data = next(tr_it)
        except StopIteration:
            tr_it = iter(train_dataloader)
            data = next(tr_it)

        if p_mixup > 0 and np.random.rand() < p_mixup:
            data = MixUp(cfg_data)(data)

        model.train()
        torch.set_grad_enabled(True)
        loss, _ = forward(model, data, criterion, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_train.append(loss.item())
        losses_train_mean.append(np.mean(losses_train[-1:-10:-1]))
        progress_bar.set_description(f'loss: {loss.item():.5f}, avg loss: {np.mean(losses_train):.5f}')

        if i % cfg_model['train_params']['n_iters_eval'] == 0:
            loss_val = evaluate(model, val_dataloader, device, criterion)
            losses_val.append(loss_val)
            progress_bar.set_description(f'val_loss: {loss_val:.5f}')

        if scheduler:
            if scheduler.__class__.__name__ == 'ReduceLROnPlateau' and losses_val:
                scheduler.step(loss_val)
            else:
                scheduler.step()

        clear_output(True)
        if cfg_model['train_params']['plot_mode']:
            train_monitor(losses_train, losses_train_mean, losses_val)

        if cfg_model['train_params']['save_best_val'] and loss_val < best_val_loss:
            best_val_loss = loss_val
            checkpoint_path = cfg_model['train_params']['checkpoint_path']
            torch.save(model.state_dict(),
                       f'{checkpoint_path}/{model.__class__.__name__}_{loss_val:.3f}')


def train_monitor(losses_train, losses_train_mean, losses_val):

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    iters = np.arange(len(losses_train))
    n_vals = len(losses_val)
    step = int(len(losses_train) / n_vals)
    val_steps = np.linspace(step, step * n_vals, n_vals)
    for i in range(2):
        ax[i].plot(iters, losses_train, linewidth=1.5, alpha=0.6,
                   c='tab:blue', label='train loss')
        ax[i].plot(iters, losses_train_mean, linewidth=2, alpha=1,
                   c='tab:blue', label='avg10 train loss')
        ax[i].plot(val_steps, losses_val, linewidth=2, alpha=1,
                   c='tab:red', label='val loss')
        ax[i].set_ylabel('MSE loss')
        ax[i].set_xlabel('Iteration')
        ax[i].legend()
        ax[i].grid()          
    ax[0].set_ylim([0.8*np.min(losses_train[-1:-100:-1]), 1.2*np.max(losses_train[-1:-100:-1])])    
    ax[1].set_yscale('log')    
    plt.show()