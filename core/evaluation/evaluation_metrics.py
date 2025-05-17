""" Evaluation metrics for the machine learning algorithms """

import numpy as np

from core.evaluation.evaluation_plots import create_distance_figure


def calc_distance_obj(ground_truth, prediction):
    """ Computes the Euclidean Distance formula between the three points in the space """
    translation_gt = ground_truth[:, :, :3].detach().cpu().numpy()
    translation_pred = prediction[:, :, :3].detach().cpu().numpy()
    distance = translation_distance(translation_gt, translation_pred, 'global')
    return distance


def calc_distance_body(ground_truth, prediction, neptune=None):
    """ Computes the Euclidean Distance formula between the three points in the space """
    body_localization = []
    for i in range(0, ground_truth.shape[2]):
        translation_gt = ground_truth[:, :, i].detach().cpu().numpy()
        translation_pred = prediction[:, :, i].detach().cpu().numpy()
        distance = translation_distance(translation_gt, translation_pred, 'global', neptune)
        body_localization.append(distance)
    body_localization = np.mean(body_localization)
    return body_localization


def translation_distance(translation_gt, translation_pred, joint_name, neptune=None):
    """ Computes the Euclidean Distance between the true and predicted translation """
    x = translation_gt[:, :, 0] - translation_pred[:, :, 0]
    y = translation_gt[:, :, 1] - translation_pred[:, :, 1]
    z = translation_gt[:, :, 2] - translation_pred[:, :, 2]
    distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if neptune:
        figure, name = create_distance_figure(distance, joint_name, batch_idx=0)
        neptune[f"{name}"].log(figure)
    distance = np.mean(distance)
    return distance


def calc_rotation_diff(ground_truth, prediction):
    """ Calculates the distance between two rotations in degree
        source: https://github.com/ethnhe/PVN3D/blob/master/pvn3d/lib/utils/evaluation_utils.py
    Arguments
        rotation_gt: numpy array with shape (3, 3) containing the ground truth rotation matrix
        rotation_pred: numpy array with shape (3, 3) containing the predicted rotation matrix
    Returns
        the rotation distance in degree
    """
    rotation_gt = ground_truth[:, :, 3:].detach().cpu().numpy()
    rotation_pred = prediction[:, :, 3:].detach().cpu().numpy()

    distances = []
    for i in range(rotation_pred.shape[0]):
        for j in range(rotation_pred.shape[1]):
            rotation_diff = np.dot(rotation_pred[i, j], rotation_gt[i, j].T)
            angular_distance = np.rad2deg(rotation_diff)
            distance = abs(angular_distance)
            distances.append(distance)
    distances = np.mean(np.array([distances]))

    return distances
