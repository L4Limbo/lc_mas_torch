import os
import matplotlib.pyplot as plt
import numpy as np
import json


def plot_train_vis(path):
    input_path = './plot_data/%s/train_vis.json' % path
    data = json.load(open(input_path))

    iterIds = data['iterIds']
    for key in data.keys():
        if (key != 'rouges' and key != 'rounds' and key != 'iterIds'):
            # Data for plotting

            fig, ax = plt.subplots()
            ax.plot(data['iterIds'], data[key])
            ax.set(xlabel='Iterations', ylabel=key, title='')

            ax.grid()

            fig.savefig("plots/%s/train/%s.png" % (path, key))


def plot_abot_vis(path):
    input_path = './plot_data/%s/abot_vis.json' % path
    data = json.load(open(input_path))

    iterIds = data['iterIds']
    for key in data.keys():
        if (key != 'rouges' and key != 'rounds' and key != 'iterIds'):
            # Data for plotting
            if len(data[key]) == 0:
                continue
            fig, ax = plt.subplots()
            ax.plot(data['iterIds'], data[key])
            ax.set(xlabel='Iterations', ylabel=key, title='')

            ax.grid()

            fig.savefig("plots/%s/abot/%s.png" % (path, key))


def plot_qbot_vis(path):
    input_path = './plot_data/%s/qbot_vis.json' % path
    data = json.load(open(input_path))

    iterIds = data['iterIds']
    for key in data.keys():
        if (key != 'rouges' and key != 'rounds' and key != 'iterIds'):
            # Data for plotting
            if len(data[key]) == 0:
                continue
            fig, ax = plt.subplots()
            ax.plot(data['iterIds'], data[key])
            ax.set(xlabel='Iterations', ylabel=key, title='')
            ax.grid()

            fig.savefig("plots/%s/qbot/%s.png" % (path, key))


def create_plots(path):
    os.makedirs('./plots/%s/train/' % path,  exist_ok=True)
    os.makedirs('./plots/%s/abot/' % path,  exist_ok=True)
    os.makedirs('./plots/%s/qbot/' % path,  exist_ok=True)

    plot_train_vis(path)
    plot_qbot_vis(path)
    plot_abot_vis(path)


if __name__ == '__main__':
    create_plots('json_26-12-2022_22-27-49')
