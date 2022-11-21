import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_2d_error(predict, truth, xlim=None, ylim=None, name='', unit=''):
    assert predict.shape == truth.shape
    assert (predict.ndim == 1 or predict.ndim == 2)

    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)

    if predict.ndim == 1:
        predict = np.vstack([np.arange(predict.shape[0]), predict]).T
        truth = np.vstack([np.arange(truth.shape[0]), truth]).T

    #     t = np.arange(predict.shape[0])
    #     plt.scatter(predict[:, 0], predict[:, 1], c=t, cmap='Blues', label='predict')
    #     plt.scatter(truth[:, 0], truth[:, 1], c=t, cmap='Reds', label='truth')

    plt.scatter(predict[:, 0], predict[:, 1], color='b', label='predict')
    plt.scatter(truth[:, 0], truth[:, 1], color='orange', label='truth')

    # t = np.arange(predict.shape[0])
    # plt.scatter(predict[:, 0], predict[:, 1], c=t, cmap='Blues', label='predict')
    # plt.scatter(truth[:, 0], truth[:, 1], c=t, cmap='Reds', label='truth')

    for i in range(predict.shape[0]):
        plt.plot(
            (predict[i][0], truth[i][0]),
            (predict[i][1], truth[i][1]),
            color='r',
            linestyle='dashed',
            label='error' if i == 0 else None
        )

    mean_error = np.mean(np.abs(truth - predict))
    plt.title(f'{name} Mean Error: {str(np.round(mean_error, 6))} {unit}')
    plt.legend()
    plt.tight_layout()


def timeseries(datas, labels, xlabel=None, ylabel=None, title=None):
    plt.style.use('seaborn-darkgrid')
    plt.title(title)
    plt.xlabel(xlabel, fontdict={'fontsize': 12})
    plt.ylabel(ylabel, fontdict={'fontsize': 12})
    for i in range(len(datas)):
        plt.plot(datas[i], label=labels[i])

    plt.legend()
    plt.tight_layout()


if __name__ == '__main__':
    # line = np.linspace(-10, 10, 100)
    # plot_2d_error(np.sin(line), np.cos(line), unit='[m]')
    # plt.show()
    v0 = pd.read_csv('logs/Plots/SelfSupVsPrivileged/VAE.csv')['Value'][:1000]
    v1 = pd.read_csv('logs/Plots/SelfSupVsPrivileged/Focus.csv')['Value'][:1000]
    timeseries(datas=[v0, v1], labels=['VAE', 'Focus'], xlabel='Train Steps', ylabel='Success Rate')
