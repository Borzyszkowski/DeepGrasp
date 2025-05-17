""" Creates evaluation plots """

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sn

from sklearn.metrics import (classification_report, cohen_kappa_score,
                             confusion_matrix, matthews_corrcoef)


def create_confusion_matrix(truth, pred, labels):
    """ Creates confusion matrix """
    report = classification_report(truth, pred, target_names=labels, digits=3, zero_division=0, output_dict=True)
    cf_matrix = confusion_matrix(truth, pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix),
                         index=[i for i in labels],
                         columns=[i for i in labels])

    yticks = df_cm.index
    # keptticks = yticks[::int(len(yticks) / 10)]
    # yticks = ['' for y in yticks]
    # yticks[::int(len(yticks) / 10)] = keptticks

    xticks = df_cm.columns
    # keptticks = xticks[::int(len(xticks) / 10)]
    # xticks = ['' for y in xticks]
    # xticks[::int(len(xticks) / 10)] = keptticks
    annot = False if len(labels) > 10 else True

    accuracy = report['accuracy']
    ck_score = cohen_kappa_score(truth, pred)
    mc_coef = matthews_corrcoef(truth, pred)

    plt.figure(figsize=(15, 15))
    plt.suptitle(f'Confusion Matrix', fontsize=20)
    plt.figtext(.15, .94, f"Accuracy: {'{:.2%}'.format(accuracy)}", fontsize=15)
    plt.figtext(.15, .92, f"Cohen’s Kappa score: {round(ck_score, 4)}", fontsize=15)
    plt.figtext(.15, .9, f"Matthew’s correlation coefficient: {round(mc_coef, 4)}", fontsize=15)

    plot = sn.heatmap(df_cm, linewidth=0, yticklabels=yticks, xticklabels=xticks, annot=annot, fmt='.2%').get_figure()
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.subplots_adjust(left=0.15)
    return plot


def create_class_report(truth, pred, labels):
    """ Creates classification report """
    report = classification_report(truth, pred, target_names=labels, digits=3, zero_division=0, output_dict=True)

    # add support (number of classes) to the report
    class_report = {}
    for key, value in report.items():
        if isinstance(value, dict) and isinstance(value['support'], int):
            new_key = key + f" ({value['support']})"
            class_report[new_key] = value
        else:
            class_report[key] = value

    accuracy = class_report['accuracy']
    class_report.pop("accuracy", None)
    ck_score = cohen_kappa_score(truth, pred)
    mc_coef = matthews_corrcoef(truth, pred)

    plt.figure(figsize=(15, 15))
    plt.suptitle(f'Classification Report', fontsize=20)
    plt.figtext(.15, .94, f"Accuracy: {'{:.2%}'.format(accuracy)}", fontsize=15)
    plt.figtext(.15, .92, f"Cohen’s Kappa score: {round(ck_score, 4)}", fontsize=15)
    plt.figtext(.15, .9, f"Matthew’s correlation coefficient: {round(mc_coef, 4)}", fontsize=15)

    df_cr = pd.DataFrame(class_report)
    plot = sn.heatmap(df_cr.iloc[:-1, :].T, linewidth=0, annot=True, fmt='.2%').get_figure()
    plt.yticks(rotation=0)
    plt.subplots_adjust(left=0.15)
    return plot


def plot_class_samples(sequences, out_path, plot_name='class_samples'):
    """ Plot a bar with number of samples for each class """
    plot_dict = {}
    for seq_name in sequences:
        plot_dict[seq_name] = len(sequences[seq_name])

    names = list(plot_dict.keys())
    values = list(plot_dict.values())
    values, names = zip(*sorted(zip(values, names)))

    logging.info(f"Minimal number of samples: {min(zip(values, names))}")
    logging.info(f"Maximal number of samples: {max(zip(values, names))}")

    plt.figure(figsize=(30, 30))
    plt.xticks(rotation='vertical', fontsize=15)
    plt.yticks(fontsize=15)
    plt.suptitle('Histogram of number of samples for each class', fontsize=25)
    plt.xlabel("Classes", fontsize=20)
    plt.ylabel("Number of samples", fontsize=20)
    plt.figtext(.15, .9, f"Minimal number of samples: {min(zip(values, names))}", fontsize=20)
    plt.figtext(.15, .92, f"Maximal number of samples: {max(zip(values, names))}", fontsize=20)

    plt.bar(range(len(plot_dict)), values, tick_label=names)

    # if the number of values is greater than 60 remove the details as they won't be visible
    if len(names) > 60:
        plt.xticks([])
    else:
        xlocs, xlabs = plt.xticks()
        for i, v in enumerate(values):
            plt.text(xlocs[i], v + 0.05, str(v), ha="center", fontsize=15)

    plt.savefig(os.path.join(out_path, plot_name))
    # plt.show()


def plot_length_histogram(seq_lengths, out_path, plot_name='length_histogram'):
    """ Plot a bar with length (duration) of each sequence """
    names = list(seq_lengths.keys())
    values = list(seq_lengths.values())
    values, names = zip(*sorted(zip(values, names)))

    logging.info(f"Minimal length of a sequence: {min(zip(values, names))}")
    logging.info(f"Maximal length of a sequence: {max(zip(values, names))}")

    plt.figure(figsize=(30, 30))
    plt.xticks(rotation='vertical', fontsize=15)
    plt.yticks(fontsize=15)
    plt.suptitle('Histogram of length (duration) of sequences', fontsize=25)
    plt.xlabel("Sequences", fontsize=20)
    plt.ylabel("Length (number of frames)", fontsize=20)
    plt.figtext(.15, .9, f"Minimal length: {min(zip(values, names))}", fontsize=20)
    plt.figtext(.15, .92, f"Maximal length: {max(zip(values, names))}", fontsize=20)

    plt.bar(range(len(seq_lengths)), values, tick_label=names)

    # if the number of values is greater than 60 remove the details as they won't be visible
    if len(names) > 60:
        plt.xticks([])
    else:
        xlocs, xlabs = plt.xticks()
        for i, v in enumerate(values):
            plt.text(xlocs[i], v + 0.05, str(v), ha="center", fontsize=15)

    plt.savefig(os.path.join(out_path, plot_name))
    # plt.show()


def create_prediction_figure(truth, pred, joint_name, batch_idx=0):
    """ Visualizes predictions and ground truth values across the sequence """
    fig_name = f'{joint_name}_predictions'

    # take only a single element from the batch
    y_values_true = truth[batch_idx]
    y_values_pred = pred[batch_idx]
    x_values = range(80)

    # generate the plot
    plot = plt.figure(figsize=(15, 15))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True)
    plt.suptitle(f'Plot of the {joint_name} truth / predicted values', fontsize=20)
    plt.xlabel(f"frames", fontsize=20)
    plt.ylabel(f"mean distance", fontsize=20)

    plt.plot(x_values, y_values_true, color='maroon', marker='o', label="ground truth")
    plt.plot(x_values, y_values_pred, color='midnightblue', marker='o', label="prediction")
    plt.legend()

    plt.close()
    # plt.show()
    return plot, fig_name


def create_distance_figure(distance, joint_name, batch_idx=0):
    """ Visualizes distance across the sequence """
    fig_name = f'{joint_name}_distance'

    # take only a single element from the batch
    y_values = distance[batch_idx]
    x_values = range(80)

    # generate the plot
    plot = plt.figure(figsize=(15, 15))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True)
    plt.suptitle(f'Plot of the {joint_name} distance', fontsize=20)
    plt.xlabel(f"frames", fontsize=20)
    plt.ylabel(f"mean distance", fontsize=20)
    plt.plot(x_values, y_values, color='maroon', marker='o')

    plt.close()
    # plt.show()
    return plot, fig_name
