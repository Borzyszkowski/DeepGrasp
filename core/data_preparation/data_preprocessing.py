"""" Preprocess the dataset for deep learning experiments """

import glob
import logging
import numpy as np
import os
import pickle
import re
import smplx
import torch

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from core.evaluation.evaluation_plots import (plot_class_samples,
                                              plot_length_histogram)
from core.mesh_visualization.visualizers.visualize_grab import vis_sequence
from core.mesh_visualization.mesh_viewer import Mesh, MeshViewer
from core.mesh_visualization.object_model import ObjectModel
from core.mesh_visualization.visual_utils import colors, euler
from graphs.graph_operations import load_graph_edges, get_arm_indexes, get_arm_and_hand_indexes
from tools.utils import (INTENTS, OBJECTS, OBJECTS_SIZES, SUBJECTS, to_cpu,
                         makepath, np2torch, params2torch, parse_npz, prepare_params)


class DataSet:
    """ Representation of DataSet, grouped by sequences, objects and subjects """
    def __init__(self, cfg):
        self.cfg = cfg
        self.grab_path = cfg.grab_path
        self.all_seqs = glob.glob(self.grab_path + '/*/*.npz')
        self.out_path = cfg.out_path
        makepath(self.out_path)

        logging.info('Starting data preprocessing!')

        # remove some data samples as requested
        self.keep_intents, self.remove_objects, self.remove_subjects = self.remove_sequences(cfg)

        # to be filled during the processing
        self.selected_seqs = []
        self.target_object_labels = []
        self.target_intent_labels = []
        self.seq_lengths = {}
        self.obj_based_seqs = {}
        self.sbj_based_seqs = {}
        self.intent_based_seqs = {}
        self.obj_intent_based_seqs = {}

        # choose the desired configuration of sequences:
        if self.cfg.mode == 0:
            desired_sequences = self.obj_based_seqs
        elif self.cfg.mode == 1:
            desired_sequences = self.intent_based_seqs
        elif self.cfg.mode == 2:
            desired_sequences = self.obj_intent_based_seqs
        else:
            raise ValueError(f'Wrong configuration (mode) specified: {self.cfg.mode}')

        # group, mask, and sort sequences based on objects, subjects, and intents
        self.filter_sequences()

        # plot a number of samples for each class
        plot_class_samples(sequences=desired_sequences, out_path=self.out_path, plot_name='class_samples')

        # split data into train, test, val subsets
        splits = self.split_data(sequences=desired_sequences)

        # process the data
        for split_name, split_sequences in splits.items():
            logging.info(f'Processing data for the {split_name} split')
            self.data_preprocessing(cfg, split_name, split_sequences)
            logging.info(f'Processing for the {split_name} split finished!')

        # save the results and plot a length histogram
        np.save(os.path.join(self.out_path, 'target_object_labels.npy'), self.target_object_labels)
        np.save(os.path.join(self.out_path, 'target_intent_labels.npy'), self.target_intent_labels)
        plot_length_histogram(seq_lengths=self.seq_lengths, out_path=self.out_path, plot_name='sequence_lengths')
        logging.info(f'Subsampled sequences to the fixed length: {self.cfg.subsample_sequences}')
        logging.info(f"Data preprocessed and exported to the location: {self.out_path}")

    def data_preprocessing(self, cfg, split_name, split_sequences, visualize=False):
        """ Processes sequences of data for each subject, object and intent """
        for sequence in tqdm(split_sequences):

            # get details of the sequence
            intent_name = sequence['intent_name']
            object_name = sequence['object_name']
            subject_id = sequence['subject_id']
            sequence = sequence['sequence_name']
            short_name = subject_id + "_" + os.path.splitext(os.path.basename(sequence))[0]

            # define label(s) depending on the desired config mode
            if self.cfg.mode == 0:
                label = [object_name]
            elif self.cfg.mode == 1:
                label = [intent_name]
            elif self.cfg.mode == 2:
                label = [object_name, intent_name]

            # add label(s) to the target labels lists
            if object_name in label and object_name not in self.target_object_labels:
                self.target_object_labels.append(object_name)
            elif intent_name in label and intent_name not in self.target_intent_labels:
                self.target_intent_labels.append(intent_name)
            logging.info(f'Sequence: {sequence}, Label(s): {label}')

            # visualize the sequence before preprocessing
            if visualize:
                self.visualize_sequence(cfg, sequence)

            # parse numpy files with sequences and filter contact frames
            frame_names = []
            seq_data = parse_npz(sequence)
            frame_mask = self.filter_contact_frames(seq_data)
            verts_mapping = pickle.load(open('core/mesh_visualization/MANO_SMPLX_vertex_ids.pkl', 'rb'))

            # total selected frames
            T = frame_mask.sum()
            self.seq_lengths[short_name] = T
            if T < 1:
                logging.info(f'Frame mask does not select any frame for sequence: {sequence}')
                continue  # if no frame is selected continue to the next sequence

            # creates a dictionary of parameters and applies the mask
            obj_data = prepare_params(seq_data.object.params, frame_mask)
            body_data = prepare_params(seq_data.body.params, frame_mask)
            rhand_data = prepare_params(seq_data.rhand.params, frame_mask)
            lhand_data = prepare_params(seq_data.lhand.params, frame_mask)
            body_contact = seq_data.contact.body[frame_mask]

            # reshape fullpose data
            body_data['fullpose'] = body_data['fullpose'].reshape(body_data['fullpose'].shape[0], 55, 3)
            rhand_data['fullpose'] = rhand_data['fullpose'].reshape(rhand_data['fullpose'].shape[0], 15, 3)
            lhand_data['fullpose'] = lhand_data['fullpose'].reshape(lhand_data['fullpose'].shape[0], 15, 3)

            # compute vertices and store them in the corresponding dictionaries
            self.compute_vertices(T, obj_data, body_data, rhand_data, lhand_data, seq_data, body_contact, verts_mapping)

            # save contact data for the subject, including whole body and both hands separately
            if cfg.save_contact:
                body_data['contact'] = body_contact if cfg.save_body_verts else []
                rhand_data['contact'], lhand_data['contact'] = [], []

                # extract contact for right and left hands
                for frame in range(T):
                    rhand_data['contact'].append(np.take(body_contact[frame], verts_mapping['right_hand']))
                    lhand_data['contact'].append(np.take(body_contact[frame], verts_mapping['left_hand']))
                rhand_data['contact'] = np.array(rhand_data['contact'])
                lhand_data['contact'] = np.array(lhand_data['contact'])
            frame_names.extend(['%s_%s' % (sequence.split('.')[0], fId) for fId in np.arange(T)])

            # compute data for the arms
            rarm_data, larm_data = self.generate_arm_data(body_data, rhand_data, lhand_data)

            # save the results
            out_data = [body_data, rhand_data, rarm_data, lhand_data, larm_data, obj_data]
            out_data_name = ['body_data', 'rhand_data', 'rarm_data', 'lhand_data', 'larm_data', 'obj_data']
            for idx, data in enumerate(out_data):
                out_dir = subject_id + '_' + os.path.basename(os.path.normpath(sequence)).split('.', 1)[0]
                data = np2torch(item=data)
                data_name = out_data_name[idx]
                outfname = makepath(os.path.join(self.out_path, split_name, out_dir, '%s.pt' % data_name), isfile=True)
                torch.save(data, outfname)
            np.savez(os.path.join(self.out_path, split_name, out_dir, 'frame_names.npz'), frame_names=frame_names)
            np.save(os.path.join(self.out_path, split_name, out_dir, 'label.npy'), label)

    def compute_vertices(self, T, obj_data, body_data, rhand_data, lhand_data, seq_data, body_contact, verts_mapping):
        """ Computes vertices for the body parts and objects and stores them in the corresponding dictionaries """

        # compute vertices for the object if requested
        if self.cfg.save_object_verts:
            mesh_path = os.path.join(self.grab_path, '..', seq_data.object.object_mesh)
            obj_mesh = Mesh(filename=mesh_path)
            obj_vtemp = np.array(obj_mesh.vertices)
            obj_m = ObjectModel(v_template=obj_vtemp, batch_size=T)
            obj_parms = params2torch(obj_data)
            verts_obj = to_cpu(obj_m(**obj_parms).vertices)
            obj_data['verts'] = verts_obj

        # compute vertices for the whole body if requested
        if self.cfg.save_body_verts:
            sbj_mesh = os.path.join(self.grab_path, '..', seq_data.body.vtemp)
            sbj_vtemp = np.array(Mesh(filename=sbj_mesh).vertices)
            sbj_m = smplx.create(model_path=self.cfg.model_path,
                                 model_type='smplx',
                                 gender=seq_data.gender,
                                 num_pca_comps=seq_data.n_comps,
                                 v_template=sbj_vtemp,
                                 batch_size=T)
            sbj_parms = params2torch(body_data)
            verts_sbj = to_cpu(sbj_m(**sbj_parms).vertices)
            joints_sbj = to_cpu(sbj_m(**sbj_parms).joints)
            body_data['verts'] = verts_sbj
            body_data['joints'] = joints_sbj
            assert body_data['joints'].shape == (self.cfg.subsample_sequences, 127, 3), "wrong shape of joints"

            wrist_loc, arm_loc, body_loc = self.compute_reference_body_data(joints_sbj)
            body_data['wrist_loc'] = wrist_loc
            body_data['arm_loc'] = arm_loc
            body_data['body_loc'] = body_loc

        # compute vertices for the right hand if requested
        if self.cfg.save_rhand_verts:
            rh_mesh = os.path.join(self.grab_path, '..', seq_data.rhand.vtemp)
            rh_vtemp = np.array(Mesh(filename=rh_mesh).vertices)
            sbj_m = smplx.create(model_path=self.cfg.model_path,
                                 model_type='mano',
                                 is_rhand=True,
                                 v_template=rh_vtemp,
                                 num_pca_comps=seq_data.n_comps,
                                 flat_hand_mean=True,
                                 batch_size=T)
            rh_parms = params2torch(rhand_data)
            verts_sbj = to_cpu(sbj_m(**rh_parms).vertices)
            joints_sbj = to_cpu(sbj_m(**rh_parms).joints)
            rhand_data['verts'] = verts_sbj
            rhand_data['joints_21'] = joints_sbj
            assert rhand_data['joints_21'].shape == (self.cfg.subsample_sequences, 21, 3), "wrong shape of joints"
            rhand_data['joints_15'] = np.take(body_data['joints'], list(range(40, 55)), axis=1)
            assert rhand_data['joints_15'].shape == (self.cfg.subsample_sequences, 15, 3), "wrong shape of joints"

            # compute bone lengths (edges)
            rhand_data['edges_21'] = self.compute_edge_lengths(rhand_data['joints_21'], "GRAPH_JOINTS21_HAND_R")
            rhand_data['edges_15'] = self.compute_edge_lengths(rhand_data['joints_15'], "GRAPH_JOINTS15_HAND_R")

        # compute vertices for the left hand if requested
        if self.cfg.save_lhand_verts:
            lh_mesh = os.path.join(self.grab_path, '..', seq_data.lhand.vtemp)
            lh_vtemp = np.array(Mesh(filename=lh_mesh).vertices)
            sbj_m = smplx.create(model_path=self.cfg.model_path,
                                 model_type='mano',
                                 is_rhand=False,
                                 v_template=lh_vtemp,
                                 num_pca_comps=seq_data.n_comps,
                                 flat_hand_mean=True,
                                 batch_size=T)
            lh_parms = params2torch(lhand_data)
            verts_sbj = to_cpu(sbj_m(**lh_parms).vertices)
            joints_sbj = to_cpu(sbj_m(**lh_parms).joints)
            lhand_data['verts'] = verts_sbj
            lhand_data['joints_21'] = joints_sbj
            assert lhand_data['joints_21'].shape == (self.cfg.subsample_sequences, 21, 3), "wrong shape of joints"
            lhand_data['joints_15'] = np.take(body_data['joints'], list(range(25, 40)), axis=1)
            assert lhand_data['joints_15'].shape == (self.cfg.subsample_sequences, 15, 3), "wrong shape of joints"

            # compute bone lengths (edges) and take right hand graph as input since it is symmetric
            lhand_data['edges_21'] = self.compute_edge_lengths(lhand_data['joints_21'], "GRAPH_JOINTS21_HAND_R")
            lhand_data['edges_15'] = self.compute_edge_lengths(lhand_data['joints_15'], "GRAPH_JOINTS15_HAND_R")

        # visualization is performed only with vertices for object and subject
        if self.cfg.save_object_verts and (self.cfg.save_body_verts or self.cfg.save_rhand_verts or self.cfg.save_lhand_verts):
            self.visualize_preprocessed(T, obj_mesh, verts_obj, sbj_m, verts_sbj, body_contact, verts_mapping)

    def generate_arm_data(self, body_data, rhand_data, lhand_data):
        """ Computes the arm data """
        arm_attributes = ['fullpose', 'joints_15', 'joints_21', 'contact']
        rarm_data = {key: rhand_data[key] for key in rhand_data if key in arm_attributes}
        larm_data = {key: lhand_data[key] for key in lhand_data if key in arm_attributes}

        # compute arm fullpose
        r_arm_and_hand_indexes, l_arm_and_hand_indexes = get_arm_and_hand_indexes()
        rarm_data['fullpose'] = np.take(body_data['fullpose'], r_arm_and_hand_indexes, axis=1)
        larm_data['fullpose'] = np.take(body_data['fullpose'], l_arm_and_hand_indexes, axis=1)

        # enabling body vertices allows to compute 15 joints and edges
        if self.cfg.save_body_verts:
            # compute arm 15 joints
            rarm_data['joints_15'] = np.take(body_data['joints'], r_arm_and_hand_indexes, axis=1)
            larm_data['joints_15'] = np.take(body_data['joints'], l_arm_and_hand_indexes, axis=1)

            # compute 15 bone lengths (edges) and take right hand graph as input since it is symmetric
            rarm_data['edges_15'] = self.compute_edge_lengths(rarm_data['joints_15'], "GRAPH_JOINTS15_ARM_R")
            larm_data['edges_15'] = self.compute_edge_lengths(larm_data['joints_15'], "GRAPH_JOINTS15_ARM_R")

            # add neck as 0 and add edges as another feature with the joints
            rarm_data['edges_15'] = np.concatenate((rarm_data['edges_15'], np.array([0])), axis=0)
            rarm_data['edges_15'] = np.repeat(rarm_data['edges_15'][np.newaxis, :], 80, axis=0).reshape(80, 20, 1)
            larm_data['edges_15'] = np.concatenate((larm_data['edges_15'], np.array([0])), axis=0)
            larm_data['edges_15'] = np.repeat(larm_data['edges_15'][np.newaxis, :], 80, axis=0).reshape(80, 20, 1)

        # enabling vertices allows to compute 21 joints and edges
        if self.cfg.save_lhand_verts and self.cfg.save_rhand_verts:
            # compute arm 21 joints
            r_arm_indexes, l_arm_indexes = get_arm_indexes()
            r_arm = np.take(body_data['joints'], r_arm_indexes, axis=1)
            l_arm = np.take(body_data['joints'], l_arm_indexes, axis=1)
            rarm_data['joints_21'] = np.concatenate((rarm_data['joints_21'], r_arm), axis=1)
            larm_data['joints_21'] = np.concatenate((larm_data['joints_21'], l_arm), axis=1)

            # compute 21 bone lengths (edges) and take right hand graph as input since it is symmetric
            rarm_data['edges_21'] = self.compute_edge_lengths(rarm_data['joints_21'], "GRAPH_JOINTS21_ARM_R")
            larm_data['edges_21'] = self.compute_edge_lengths(larm_data['joints_21'], "GRAPH_JOINTS21_ARM_R")

            # add neck as 0 and add edges as another feature with the joints
            rarm_data['edges_21'] = np.concatenate((rarm_data['edges_21'], np.array([0])), axis=0)
            rarm_data['edges_21'] = np.repeat(rarm_data['edges_21'][np.newaxis, :], 80, axis=0).reshape(80, 26, 1)
            larm_data['edges_21'] = np.concatenate((larm_data['edges_21'], np.array([0])), axis=0)
            larm_data['edges_21'] = np.repeat(larm_data['edges_21'][np.newaxis, :], 80, axis=0).reshape(80, 26, 1)

        return rarm_data, larm_data

    def compute_edge_lengths(self, joint_coordinates, graph_name):
        """ Computes lengths of the graph edges (bones) based on the given joint coordinates """
        joint_coordinates = joint_coordinates[0, :, :]
        graph_edges = load_graph_edges(graph_name).numpy()
        bone_lengths = []
        for i in range(0, graph_edges.shape[1], 2):
            # get index of the source and target node in the graph
            source = graph_edges[0][i]
            target = graph_edges[1][i]
            # compute the Euclidean distance between the target and source joint
            bone_len = np.linalg.norm(joint_coordinates[source] - joint_coordinates[target])
            bone_lengths.append(bone_len)
        return np.array(bone_lengths)

    def compute_reference_body_data(self, body_joints):
        """ Computes the reference body joint and computes relative location of the other joints """
        # Consider only SMPL-X joints without the head keypoints etc. (55 joints instead of 127)
        body_joints = body_joints[:, 0:55, :].tolist()

        # Set neck (idx 12) as a reference joint (0,0,0) and compute the distance to all body parts
        new_body_joints = []
        for sample in body_joints:
            tmp = []
            original_neck = sample[12]
            for joint in sample:
                new_joint = np.subtract(joint, original_neck)
                tmp.append(new_joint)
            new_body_joints.append(tmp)
        new_body_joints = np.array(new_body_joints)

        # Select only the wrist joints
        wrist_joints = new_body_joints[:, 20:22, :]

        # Select only the arm joints
        r_arm_indexes, l_arm_indexes = get_arm_and_hand_indexes()
        arm_joints = np.take(new_body_joints, r_arm_indexes + list(set(l_arm_indexes) - set(r_arm_indexes)), axis=1)
        return wrist_joints, arm_joints, new_body_joints

    def remove_sequences(self, cfg):
        """ Removes intents, objects, and subjects when requested by the user """
        assert isinstance(cfg.keep_intents, list) and len(cfg.keep_intents) >= 1
        assert all(intnt in INTENTS for intnt in cfg.keep_intents), 'Wrong intent to keep!'
        logging.info(f'Specified intents to keep: {cfg.keep_intents}')

        if not cfg.remove_objects:
            logging.info('No remove_objects were given: --> processing all object classes!')
            remove_obj = []
        else:
            assert isinstance(cfg.remove_objects, list)
            assert all(obj in OBJECTS for obj in cfg.remove_objects), 'Wrong object to remove!'
            remove_obj = cfg.remove_objects
            logging.info(f'Specified objects to remove: {cfg.remove_objects}')

        if not cfg.remove_subjects:
            logging.info('No remove_subjects were given: --> processing all subject classes!')
            remove_sbj = []
        else:
            assert isinstance(cfg.remove_subjects, list)
            assert all(obj in SUBJECTS for obj in cfg.remove_subjects), 'Wrong subject to remove!'
            remove_sbj = cfg.remove_subjects
            logging.info(f'Specified subjects to remove: {cfg.remove_subjects}')

        return cfg.keep_intents, remove_obj, remove_sbj

    def filter_sequences(self):
        """ Processes sequences of data: filters the intents and groups motion based on objects and subjects """
        for sequence in self.all_seqs:
            subject_id = sequence.split('/')[-2]
            basename = os.path.basename(sequence)
            object_name = basename.split('_')[0]

            # consolidate object names such that the same objects of various shapes are of the same class
            if not self.cfg.distinguish_shapes:
                replace_list = ['small', 'medium', 'large']
                object_name = re.sub(r'|'.join(map(re.escape, replace_list)), '', object_name)
                valid_obj = OBJECTS
            else:
                valid_obj = OBJECTS_SIZES

            # retrieve the motion intent
            if not any(intnt in basename for intnt in INTENTS[:3]):
                intent_name = 'use'
            else:
                intent_list = [intnt for intnt in INTENTS[:3] if intnt in basename]
                assert len(intent_list) == 1, 'Wrong intent in the dataset!'
                intent_name = intent_list[0]

            assert object_name in valid_obj, f'Wrong object_name detected: {object_name}!'
            assert subject_id in SUBJECTS, f'Wrong subject_id detected: {subject_id}!'
            assert intent_name in INTENTS, f'Wrong intent_name detected: {intent_name}!'

            # filter data based on the motion intent
            if 'all' in self.keep_intents:
                pass
            elif intent_name not in self.keep_intents:
                continue

            sequence = {"sequence_name": sequence,
                        "subject_id": subject_id,
                        "object_name": object_name,
                        "intent_name": intent_name}

            # filter data based on the objects and subject
            if object_name not in self.remove_objects and subject_id not in self.remove_subjects:
                self.selected_seqs.append(sequence)

                # group motion sequences based on objects
                if object_name not in self.obj_based_seqs:
                    self.obj_based_seqs[object_name] = [sequence]
                else:
                    self.obj_based_seqs[object_name].append(sequence)

                # group motion sequences based on intents
                if intent_name not in self.intent_based_seqs:
                    self.intent_based_seqs[intent_name] = [sequence]
                else:
                    self.intent_based_seqs[intent_name].append(sequence)

                # group motion sequences based on objects and intents together
                obj_intent_name = object_name + '_' + intent_name
                if obj_intent_name not in self.obj_intent_based_seqs:
                    self.obj_intent_based_seqs[obj_intent_name] = [sequence]
                else:
                    self.obj_intent_based_seqs[obj_intent_name].append(sequence)

                # group motion sequences based on subjects
                if subject_id not in self.sbj_based_seqs:
                    self.sbj_based_seqs[subject_id] = [sequence]
                else:
                    self.sbj_based_seqs[subject_id].append(sequence)

        logging.info(f'Total sequences: {len(self.all_seqs)}')
        logging.info(f'Selected sequences: {len(self.selected_seqs)}')

        logging.info(f'Total number of objects: {len(self.obj_based_seqs) + len(self.remove_objects)}')
        logging.info(f'Number of selected objects: {len(self.obj_based_seqs)}')

        logging.info(f'Total number of subjects: {len(self.sbj_based_seqs) + len(self.remove_subjects)}')
        logging.info(f'Number of selected subjects: {len(self.sbj_based_seqs)}')

    def subsample_sequences(self, frame_mask, expected_length):
        """ Subsamples sequences to a specified length. Considers only the prefiltered frames """
        # Create an indexed mask
        mask_indexed = [(idx, val) for idx, val in enumerate(frame_mask)]

        # Create a temporal mask with only selected True frames
        temp_mask = np.array([elem for elem in mask_indexed if elem[1]])

        # Subsample prefiltered frames
        assert expected_length <= len(temp_mask), f'Expected length is greater than the sequence: {len(temp_mask)}!'
        temp_mask = temp_mask[np.round_(np.linspace(1, len(temp_mask)-1, num=expected_length)).astype(int)]

        # Add subsampled frames to the final mask
        subsampled_mask = [False for _ in range(len(frame_mask))]
        for elem in range(len(temp_mask)):
            index = temp_mask[elem][0]
            subsampled_mask[index] = True
        return np.array(subsampled_mask)

    def subsample_sequences_fixed_timestamp(self, frame_mask, expected_length):
        """ Subsamples sequences to a specified length with fixed timestamp. Considers only the prefiltered frames """
        # Create an indexed mask
        mask_indexed = [(idx, val) for idx, val in enumerate(frame_mask)]

        # Create a temporal mask with only selected True frames
        temp_mask = np.array([elem for elem in mask_indexed if elem[1]])

        # Subsample prefiltered frames with a fixed timestamp (the maximum lengths are hardcoded)
        if self.cfg.only_contact:
            max_len = 94
            timestamp = max_len // expected_length
        elif self.cfg.only_prehension:
            max_len = 85
            timestamp = max_len // expected_length
        else:
            max_len = 491
            timestamp = max_len // expected_length
        assert timestamp > 0, 'Length for subsampling greater than the sequence!'

        # Subsample prefiltered frames
        temp_mask = [temp_mask[i] for i in range(0, expected_length * timestamp, timestamp)]
        assert len(temp_mask) == expected_length, 'Length of the sequence is incorrect!'

        # Add subsampled frames to the final mask
        subsampled_mask = [False for _ in range(len(frame_mask))]
        for elem in range(len(temp_mask)):
            index = temp_mask[elem][0]
            subsampled_mask[index] = True
        return np.array(subsampled_mask)

    def filter_contact_frames(self, seq_data, fixed_timestamp=False):
        """ Filters the frames and keeps only these that contain contact between subject and object """
        assert not (self.cfg.only_contact and self.cfg.only_prehension), 'Contact and prehension were selected at once!'
        if self.cfg.only_contact:
            frame_mask = (seq_data['contact']['object'] > 0).any(axis=1)
        elif self.cfg.only_prehension:
            frame_mask = self.filter_prehension_phase(seq_data)
        else:
            frame_mask = (seq_data['contact']['object'] > -1).any(axis=1)

        if self.cfg.subsample_sequences:
            assert isinstance(self.cfg.subsample_sequences, int), 'Length for subsampling is not an integer!'
            assert self.cfg.subsample_sequences < len(frame_mask), 'Length for subsampling greater than the sequence!'
            if fixed_timestamp:
                frame_mask = self.subsample_sequences_fixed_timestamp(frame_mask, self.cfg.subsample_sequences)
            else:
                frame_mask = self.subsample_sequences(frame_mask, self.cfg.subsample_sequences)
        return frame_mask

    def filter_prehension_phase(self, seq_data, crop_preparation=False):
        """ Filters the frames and keeps only these that contain prehension phase """
        frame_mask = (seq_data['contact']['object'] > 0).any(axis=1).tolist()

        # Get index of the first contact frame
        first_contact_idx = frame_mask.index(True)

        # Get frames that contain the prehension phase only
        frame_mask = [False if x > first_contact_idx else True for x in range(len(frame_mask))]

        # Crops the preparation phase (the "T" pose)
        if crop_preparation:
            frame_mask = [False if idx < 60 else frame_mask[idx] for idx in range(len(frame_mask))]

        return np.array(frame_mask)

    def split_data(self, sequences, train_size=0.8, test_size=0.1, val_size=0.1):
        """ Splits data into train/test/val sets """
        assert train_size + test_size + val_size == 1, 'Wrong train/test/val ratio!'
        logging.info(f'Splitting data into train/test/val with ratio {train_size}/{test_size}/{val_size}')

        splits = {'train': [], 'test': [], 'val': []}
        leftovers = []
        for seq_name in sequences:
            seq_len = len(sequences[seq_name])
            if seq_len < 3:
                leftovers.extend(sequences[seq_name])
                logging.info(f'{seq_name} has too few samples to split the data: {seq_len}')
                continue
            train, test = train_test_split(sequences[seq_name], test_size=0.1)
            train, val = train_test_split(train, test_size=0.1)
            splits['train'].extend(train)
            splits['test'].extend(test)
            splits['val'].extend(val)

        logging.info(f'{len(leftovers)} sequences had too few samples to split them! They are added to the training set')
        splits['train'] += leftovers
        assert len(splits["train"]) + len(splits["test"]) + len(splits["val"]) == len(self.selected_seqs)
        logging.info(f'Splitted: {len(splits["train"])} train - {len(splits["test"])} test - {len(splits["val"])} val')
        return splits

    def visualize_sequence(self, cfg, sequence):
        """ Visualize the sequence in MeshViewer """
        logging.info('Visualizing input data!')
        mv = MeshViewer(offscreen=False)
        cfg.compute_joints = False

        # set the camera pose
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = euler([80, -15, 0], 'xzx')
        camera_pose[:3, 3] = np.array([-.5, -4., 1.5])
        mv.update_camera_pose(camera_pose)

        vis_sequence(cfg, sequence, mv)
        mv.close_viewer()

    def visualize_preprocessed(self, T, obj_mesh, verts_obj, sbj_m, verts_sbj, contact, verts_map, offscreen=False):
        """ Visualize the preprocessed data in MeshViewer and render it to mp4 or gif """
        # set offscreen=True to enable rendering to mp4, otherwise recordings will be saved in gif
        logging.info('Visualizing preprocessed data!')
        if self.cfg.save_lhand_verts and self.cfg.save_rhand_verts:
            logging.info('Visualizing can be performed only for a single hand or for the whole body')

        mv = MeshViewer(offscreen=offscreen)
        recording_folder = self.cfg.out_path + '/recordings'
        makepath(recording_folder)
        if not offscreen:
            mv.start_gif_recording()

        # set the camera pose
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = euler([80, -15, 0], 'xzx')
        camera_pose[:3, 3] = np.array([-.5, -4., 1.5])
        mv.update_camera_pose(camera_pose)

        for frame in range(T):
            s_mesh = Mesh(vertices=verts_sbj[frame], faces=sbj_m.faces, vc=colors['pink'], smooth=True)
            s_mesh_wf = Mesh(vertices=verts_sbj[frame], faces=sbj_m.faces, vc=colors['grey'], wireframe=True)

            if self.cfg.save_lhand_verts:
                vertex_ids = contact[frame][verts_map['left_hand']] > 0
            elif self.cfg.save_rhand_verts:
                vertex_ids = contact[frame][verts_map['right_hand']] > 0
            elif self.cfg.save_body_verts:
                vertex_ids = contact[frame] > 0

            s_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=vertex_ids)
            o_mesh = Mesh(vertices=verts_obj[frame], faces=obj_mesh.faces, vc=colors['yellow'])
            mv.set_static_meshes([s_mesh, s_mesh_wf, o_mesh])
            if offscreen:
                mv.save_snapshot(recording_folder + '/%04d.png' % frame)

        if offscreen:
            mv.save_recording(recording_folder)
        else:
            mv.close_viewer()
            mv.end_gif_recording(recording_folder)
