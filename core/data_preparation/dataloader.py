"""" Load preprocessed data """

import argparse
import glob
import logging
import numpy as np
import os
import torch

from torch.utils import data


class LoadData(data.Dataset):
    def __init__(self, dataset_dir, data_split='train', only_params=False):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.split_name = data_split
        self.only_params = only_params
        self.sequence_paths = glob.glob(os.path.join(dataset_dir, data_split, '*'))
        self.target_classes = self.load_targets()

    def load(self, datasets):
        """ Load an actual file """
        loaded = {}
        for d in datasets:
            k = os.path.basename(d).split('_')[0]
            loaded[k] = torch.load(d, weights_only=True)
        return loaded

    def load_sample(self, idx):
        """ Load a single data sample, including its data and label """
        pickle_data = glob.glob(self.sequence_paths[idx] + '/*.pt')
        label_data = glob.glob(self.sequence_paths[idx] + '/*.npy')
        if len(label_data) == 0 or len(pickle_data) == 0:
            raise ValueError('Error when loading features/labels!')
        else:
            labels = np.load(label_data[0])
            loaded_data = self.load(pickle_data)
        return loaded_data, labels

    def load_targets(self):
        """ Load target classes from the file to make sure that all indexes of labels are correct """
        targets = glob.glob(self.dataset_dir + '/*.npy')
        target_labels = []
        if len(targets) == 0:
            raise ValueError('Error when loading target classes! Make sure that the dataset path is correct!')
        for target in targets:
            target_labels.append(np.load(target).tolist())
        return target_labels

    def __len__(self):
        """ Return number of samples in the entire dataset """
        return len(self.sequence_paths)

    def __getitem__(self, idx):
        """ Get single item from the dataset and return it as a dict with features and labels """
        data_item, labels = self.load_sample(idx)
        labels_idx = []

        # for multilabel and multitasking learning one item can have multiple labels
        for list_idx in range(len(labels)):
            if labels[list_idx] in self.target_classes[list_idx]:
                labels_idx.append(self.target_classes[list_idx].index(str(labels[list_idx])))
            else:
                raise ValueError(f'Unknown label {labels[list_idx]}!')

        label_idx_tensor = torch.LongTensor([labels_idx])
        data_sample = {'features': data_item, 'labels': label_idx_tensor}
        return data_sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data-Loader')
    parser.add_argument('--data-dir', required=True, type=str, help='The path to the preprocessed data')
    parser.add_argument('--split-name', required=True, choices=['train', 'test', 'val'], type=str,
                        help='Name of data split')
    args = parser.parse_args()
    data_dir = args.data_dir
    split_name = args.split_name

    logging.info(f'Loading data from: {data_dir} for {split_name}')
    dataset = LoadData(data_dir, split_name)
    logging.info(f'Data loading finished!')
