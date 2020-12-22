import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
from dataloader.augmentations import MixUp
import torch


def forward(model, data, criterion, device):
    model_name = model.__class__.__name__
    yields = data['yield'].to(device)
    if model_name in ['S2BandModel', 'BandsCNN', 'HybridBandsModel', 'AutoS2BandNet', 'AutoS2BandNetShallow']:
        s2_bands = data['s2_bands'].to(device)
        prediction = model(s2_bands).reshape(yields.shape)
    elif model_name in ['ClimNet']:
        clim_bands = data['clim_bands'].to(device)
        prediction = model(clim_bands).reshape(yields.shape)
    elif model_name in ['S2CilmNet']:
        s2_bands = data['s2_bands'].to(device)
        clim_bands = data['clim_bands'].to(device)
        prediction = model(s2_bands, clim_bands).reshape(yields.shape)
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


def train_epoch(model, dataloader, device, optimizer, criterion, cfg_data):
    p_mixup = dataloader.dataset.p_mup
    losses = []
    progress_bar = tqdm(range(len(dataloader)))
    dataloader_iter = iter(dataloader)
    for _ in progress_bar:
        data = next(dataloader_iter)

        if p_mixup > 0 and np.random.rand() < p_mixup:
            data = MixUp(cfg_data)(data)

        model.train()
        torch.set_grad_enabled(True)
        loss, _ = forward(model, data, criterion, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        progress_bar.set_description(f'loss: {loss.item()}, avg loss: {np.mean(losses)}')

    return losses


def train_model(model, train_dataloader, val_dataloader, device, optimizer, criterion, cfg_data, cfg_model):
    losses_train, losses_train_mean = [], []
    losses_val, losses_val_mean = [], []
    best_val_loss = 1e6
    n_epochs = cfg_model['train_params']['n_epochs']

    progress_bar = tqdm(range(n_epochs))
    for _ in progress_bar:
        epoch_loss_train = train_epoch(model, train_dataloader, device, optimizer, criterion, cfg_data)
        epoch_loss_val = evaluate(model, val_dataloader, device, criterion)

        train_loss = np.mean(epoch_loss_train)
        losses_train.append(train_loss)
        losses_train_mean.append(np.mean(losses_train))
        losses_val.append(epoch_loss_val)
        losses_val_mean.append(np.mean(losses_val))
        progress_bar.set_description(f'train loss: {train_loss:.4f}, val loss: {epoch_loss_val:.4f}')
        clear_output(True)
        if cfg_model['train_params']['plot_mode']:
            train_monitor(losses_train, losses_train_mean, losses_val, losses_val_mean)
        
        if cfg_model['train_params']['save_best_val'] and epoch_loss_val < best_val_loss:
          best_val_loss = epoch_loss_val
          checkpoint_path = cfg_model['train_params']['checkpoint_path']
          torch.save(model.state_dict(),
            f'{checkpoint_path}/{model.__class__.__name__}_{epoch_loss_val:.3f}')


def train_monitor(osses_train, losses_train_mean, losses_val, losses_val_mean):
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))

    iters = np.arange(len(losses_train))
    ax[0].plot(iters, losses_train, linewidth=1.5, alpha=0.6,
               c='tab:blue', label='train loss')
    ax[0].plot(iters, losses_val, linewidth=1.5, alpha=0.6,
               c='tab:red', label='validation loss')
    ax[0].plot(iters, losses_train_mean, linewidth=2, alpha=1,
               c='tab:blue', label='mean train loss')
    ax[0].plot(iters, losses_val_mean, linewidth=2, alpha=1,
               c='tab:red', label='mean validation loss')

    ax[1].plot(iters, losses_train, linewidth=1.5, alpha=0.6,
               c='tab:blue', label='train loss')
    ax[1].plot(iters, losses_val, linewidth=1.5, alpha=0.6,
               c='tab:red', label='validation loss')
    ax[1].plot(iters, losses_train_mean, linewidth=2, alpha=1,
               c='tab:blue', label='mean train loss')
    ax[1].plot(iters, losses_val_mean, linewidth=2, alpha=1,
               c='tab:red', label='mean validation loss')
    ax[1].set_yscale('log')

    for i in [0, 1]:
        ax[i].set_ylabel('MSE loss')
        ax[i].set_xlabel('Iteration')
        ax[i].legend()
        ax[i].grid()

    plt.show()
