""" Main class for the training """

import logging
import numpy as np
import os
import torch

from copy import deepcopy
from datetime import datetime
from ray import train
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from smplx.joint_names import JOINT_NAMES
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader

from core.training.training_tools import EarlyStopping
from core.evaluation.evaluation_metrics import calc_distance_body, calc_distance_obj, translation_distance
from core.data_preparation.create_graphs import prepare_inputs, select_hands, select_graphs
from core.data_preparation.dataloader import LoadData
from core.evaluation.evaluation_plots import (create_class_report, create_prediction_figure, create_confusion_matrix)
from core.neural_networks.multimodal_gcn import MultiModalGCN
from core.neural_networks.multimodal_rnn import MultiModalRNN
from graphs.graph_operations import get_arm_and_hand_indexes
from tools.utils import makelogger, makepath


class Trainer:
    def __init__(self, cfg, hyperparam_cfg, neptune_run, inference_only=False):
        self.cfg = cfg
        self.hyperparam_cfg = hyperparam_cfg
        self.work_dir = train.get_context().get_trial_dir()
        self.neptune_run = neptune_run
        makelogger(makepath(os.path.join(self.work_dir, f'{cfg.expr_ID}.log'), isfile=True)).info

        # set the destination directory for TensorBoard
        self.swriter = self.set_summary_writer()

        # set the hardware type (use GPU with CUDA if available)
        self.device = self.set_hardware_type()

        # load data and target classes
        self.ds_train, self.ds_val, self.ds_test, self.target_classes = self.load_data(inference_only)

        # select the neural network architecture, modalities, and types of data to process
        self.selected_input, self.model_name, self.model = self.select_params()

        # check if multiple GPUs can be used for training
        if cfg.use_multigpu:
            self.model = nn.DataParallel(self.model)
            logging.info("Training on Multiple GPUs")

        # define an optimizer and initialize the loss
        self.optimizer = self.set_optimizer()
        self.best_loss = np.inf

        # load weights from the file if specified
        if cfg.load_weights_path is not None:
            self.get_model()

    def fit(self):
        """ Main logic, in which the model is trained and evaluated """
        start_time = datetime.now().replace(microsecond=0)
        logging.info(f'Started training at {start_time} for {self.cfg.n_epochs} epochs\n')

        # Schedule learning rate optimization and early stopping
        prev_lr = self.hyperparam_cfg['lr']
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=10)
        early_stopping = EarlyStopping(patience=30, trace_func=logging.info)

        # Run the main train / val logic
        for epoch_num in range(1, self.cfg.n_epochs + 1):
            logging.info('--- Starting Epoch # %03d' % epoch_num)

            # Run main training and evaluation logic
            train_loss = self.train(epoch_num)
            val_loss = self.evaluate(epoch_num)

            # Update the learning rate if required by the optimizer
            lr_scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr != prev_lr:
                logging.info('--- Model learning rate changed from %.2e to %.2e ---' % (prev_lr, current_lr))
                prev_lr = current_lr

            # Save loss and the best model
            with torch.no_grad():
                if val_loss < self.best_loss:
                    self.save_model(epoch_num)
                    self.best_loss = val_loss

                self.swriter.add_scalars('total_loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch_num)
                self.neptune_run[f"metrics/total_loss/train_loss"].log(train_loss)
                self.neptune_run[f"metrics/total_loss/val_loss"].log(val_loss)

            # Check the evaluation loss for the early stopping condition
            if early_stopping(val_loss):
                logging.info('Early stopping training of the model!\n')
                break

        # Finish the training and evaluate the model on the test set
        end_time = datetime.now().replace(microsecond=0)
        logging.info(f'Finished Training at {end_time}')
        logging.info(f'Training done in {(end_time - start_time)}!')
        logging.info(f'Best val total loss achieved: {self.best_loss}')
        logging.info(f'Best model path: {self.model_path}')
        logging.info(f'Running evaluation on the test set!\n')
        _ = self.evaluate(epoch_num=1, ds_name='test')

    def train(self, epoch_num, ds_name='train'):
        """ Main training logic """
        self.model.train()
        torch.autograd.set_detect_anomaly(True)
        epoch_loss, epoch_pred, epoch_gt, regression_metrics = self.set_epoch_metrics()
        for (it, batch) in enumerate(self.ds_train, 0):

            # Create inputs for the neural network
            input_tensor, labels = self.prepare_classification(batch)
            ground_truth = self.prepare_regression(batch)

            # Generate model visualization and information about the input tensor
            if epoch_num == 1 and it == 0:
                self.visualize_model(input_tensor)

            # Clear gradients
            self.optimizer.zero_grad()
            outputs = self.model(input_tensor, self.cfg)

            # Calculate the loss function
            current_loss = self.multitask_loss(outputs, labels=labels, ground_truth=ground_truth)

            # Calculating gradients
            current_loss.backward()
            epoch_loss.append(current_loss.item())

            # Update parameters
            self.optimizer.step()

            # Compute performance of the cls/reg tasks
            if self.cfg.cls_tasks != 0:
                epoch_pred, epoch_gt, acc = self.cls_perf(epoch_pred, epoch_gt, outputs, labels)
            if self.cfg.reg_tasks != 0:
                regression_metrics, ed = self.reg_perf(outputs, ground_truth, regression_metrics)

            # Print information about the loss
            if it % self.cfg.log_every_iteration == 0:
                self.create_loss_message(current_loss, epoch_num, it, ds_name,
                                         acc=acc if self.cfg.cls_tasks != 0 else None,
                                         ed=ed if self.cfg.reg_tasks != 0 else None)

        return self.compute_epoch_summary(ds_name, epoch_loss, epoch_gt, epoch_pred, epoch_num, regression_metrics)

    def evaluate(self, epoch_num, ds_name='val'):
        """ Main evaluation logic for validation and testing of the model """
        data = self.ds_val if ds_name == 'val' else self.ds_test
        epoch_loss, epoch_pred, epoch_gt, regression_metrics = self.set_epoch_metrics()
        with torch.no_grad():
            for (it, batch) in enumerate(data, 0):

                # Create inputs for the neural network
                input_tensor, labels = self.prepare_classification(batch)
                ground_truth = self.prepare_regression(batch)

                # Forward propagation
                outputs = self.model(input_tensor, self.cfg)

                # Calculate the loss function
                current_loss = self.multitask_loss(outputs, labels=labels, ground_truth=ground_truth)
                epoch_loss.append(current_loss.item())

                # Compute performance of the cls/reg tasks
                if self.cfg.cls_tasks != 0:
                    epoch_pred, epoch_gt, acc = self.cls_perf(epoch_pred, epoch_gt, outputs, labels)
                if self.cfg.reg_tasks != 0:
                    regression_metrics, ed = self.reg_perf(outputs, ground_truth, regression_metrics)

                # Print information about the loss
                if it % self.cfg.log_every_iteration == 0:
                    self.create_loss_message(current_loss, epoch_num, it, ds_name,
                                             acc=acc if self.cfg.cls_tasks != 0 else None,
                                             ed=ed if self.cfg.reg_tasks != 0 else None)

            return self.compute_epoch_summary(ds_name, epoch_loss, epoch_gt, epoch_pred, epoch_num, regression_metrics)

    def compute_epoch_summary(self, ds_name, epoch_loss, epoch_gt, epoch_pred, epoch_num, reg_metrics):
        """ Compute the epoch summary and save the report """
        epoch_loss = sum(epoch_loss) / len(epoch_loss)
        epoch_acc = self.cls_summary(ds_name, epoch_gt, epoch_pred, epoch_num) if self.cfg.cls_tasks != 0 else None
        epoch_ed = self.reg_summary(ds_name, reg_metrics, epoch_num) if self.cfg.reg_tasks != 0 else None
        train.report({"mode": ds_name, "loss": epoch_loss, "accuracy": epoch_acc, "epoch_num": epoch_num})
        return epoch_loss

    def set_epoch_metrics(self):
        """ Set data structures where metrics for each epoch are stored """
        epoch_loss = []
        epoch_pred = [[] for _ in range(self.cfg.cls_tasks)]
        epoch_gt = [[] for _ in range(self.cfg.cls_tasks)]
        common_metrics = {'MAE': {}, 'MSE': {}, 'RMSE': {}, 'MAPE': {}, 'Euclid': {}}
        regression_metrics = {}

        # copy the regression metrics for each task
        for task_name, to_process in self.cfg.reg_task_names.items():
            if to_process:
                regression_metrics[task_name] = deepcopy(common_metrics)
        return epoch_loss, epoch_pred, epoch_gt, regression_metrics

    def set_summary_writer(self):
        """ Set the summary writer for tensorboard """
        summary_logdir = os.path.join(self.work_dir, 'summaries')
        swriter = SummaryWriter(log_dir=summary_logdir)
        logging.info(f'[{self.cfg.expr_ID}] - DeepGrasp experiment has started!')
        logging.info(f'tensorboard --logdir={summary_logdir}')
        logging.info(f'Torch Version: {torch.__version__}')
        logging.info(f'Base dataset directory: {self.cfg.dataset_dir}')
        return swriter

    def set_hardware_type(self):
        """ Set the hardware type such as CPU/GPU """
        use_cuda = torch.cuda.is_available()
        device = torch.device(f"cuda:{self.cfg.cuda_id}" if use_cuda else "cpu")
        if use_cuda:
            torch.cuda.empty_cache()
            gpu_brand = torch.cuda.get_device_name(self.cfg.cuda_id)
            gpu_count = torch.cuda.device_count() if self.cfg.use_multigpu else 1
            logging.info(f'Using {gpu_count} CUDA core(s) [{gpu_brand}] for training!')
        else:
            logging.info(f'Using CPU for training!')
        return device

    def set_optimizer(self):
        """ Set the optimizer and its parameters """
        params = [var[1] for var in self.model.named_parameters()]
        params_number = sum(p.numel() for p in params if p.requires_grad)
        logging.info('Total trainable parameters of the model: %2.2f M.' % (params_number * 1e-6))
        optimizer = optim.Adam(params, lr=self.hyperparam_cfg['lr'])
        return optimizer

    def select_params(self):
        """ Selects params such as the neural network architecture, modalities, and types of data """
        # select which hand(s) should be processed, based on the given configuration
        hand = select_hands(self.cfg)
        logging.info(f"Hand(s) to process: {hand}")

        # select which graph(s) or tensor(s) should be processed, based on the given input type
        selected_input, multimodalities = select_graphs(self.cfg, hand)
        logging.info(f"Input types for the model: {self.cfg.input_types}")

        # select and initialize the deep neural network
        model_name, model = self.select_model(selected_input, multimodalities)
        return selected_input, model_name, model

    def select_model(self, selected_input, multimodalities):
        """ Selects the neural network architecture based on the desired configuration """
        if self.cfg.parse_graph:
            model = MultiModalGCN(graphs_config=selected_input,
                                  hyperparam_cfg=self.hyperparam_cfg,
                                  multimodalities=multimodalities
                                  ).to(self.device)
        else:
            model = MultiModalRNN(tensors_config=selected_input,
                                  hyperparam_cfg=self.hyperparam_cfg,
                                  multimodalities=multimodalities
                                  ).to(self.device)
        model_name = model.__class__.__name__ if not self.cfg.model_name else self.cfg.model_name
        logging.info(f'Selected model name: {model_name}')
        return model_name, model

    def get_model(self):
        """ Loads weights of the model from the specified path """
        restored = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        restored.load_state_dict(torch.load(self.cfg.load_weights_path, map_location=self.device), strict=False)
        logging.info(f'Restored model from {self.cfg.load_weights_path}')

    def save_model(self, epoch_num, neptune_checkpoint=False):
        """ Saves a checkpoint with the model """
        checkpoint_dir = os.path.join(self.cfg.trial_dir, f"checkpoints")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.model_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch_num}")
        torch.save((self.model.state_dict(), self.optimizer.state_dict()), self.model_path)
        if neptune_checkpoint:
                self.neptune_run[f"checkpoints/{epoch_num}"].upload(self.model_path)
        logging.info(f'Checkpoint with the model saved at: {self.model_path}')

    def visualize_model(self, input_tensor):
        """ Save model visualization and input tensor """
        with open(os.path.join(self.cfg.trial_dir, f'model.txt'), 'w') as f:
            f.write(str(self.model))
        self.neptune_run['versioning/model_version'].track_files(os.path.join(self.cfg.trial_dir, f'model.txt'))
        self.neptune_run["versioning/model"] = str(self.model)
        self.neptune_run["model_name"] = self.model_name

        input_tensor = input_tensor if self.cfg.parse_graph else "input_tensor display not supported for the RNNs"
        with open(os.path.join(self.cfg.trial_dir, f'input_tensor.txt'), 'w') as f:
            logging.info(f"Input tensor: {input_tensor}")
            f.write(str(input_tensor))
        self.neptune_run['versioning/input_version'].track_files(os.path.join(self.cfg.trial_dir, f'input_tensor.txt'))
        self.neptune_run["versioning/input"] = str(input_tensor)

    def multitask_loss(self, outputs, labels, ground_truth):
        """ Define the loss function for multiple tasks """
        classification_loss = torch.nn.CrossEntropyLoss()
        regression_loss = torch.nn.MSELoss()
        loss = 0

        # Prepare the classification losses
        for index, (task_name, to_process) in enumerate(self.cfg.cls_task_names.items()):
            if to_process:
                loss += classification_loss(outputs[task_name], labels[:, index])
        # Prepare the regression losses
        for task_name, to_process in self.cfg.reg_task_names.items():
            if task_name == 'obj_pose' and to_process:
                loss += regression_loss(outputs[task_name], ground_truth[task_name])
            elif task_name == 'body_loc' and to_process:
                # compute euclidean dist. and get only the diagonal values (distance between the corresponding points)
                body_loss = torch.cdist(x1=outputs[task_name], x2=ground_truth[task_name])
                body_loss = torch.diagonal(body_loss, dim1=-2, dim2=-1)
                # compute the mean value of the distance
                loss = torch.mean(body_loss)
        return loss

    def multitask_acc(self, outputs, ground_truth):
        """ Compute accuracy for multiple tasks """
        predicted, correct, accuracy = [], [], []
        for index, (task_name, to_process) in enumerate(self.cfg.cls_task_names.items()):
            if to_process:
                _, pred = torch.max(outputs[task_name].data, 1)
                corr = (pred == ground_truth[:, index]).sum()
                acc = round((100 * corr / float(ground_truth.shape[0])).tolist(), 2)
                predicted.append(pred.tolist())
                correct.append(corr.tolist())
                accuracy.append(acc)
        return predicted, correct, accuracy

    def cls_perf(self, epoch_pred, epoch_gt, outputs, labels):
        """ Compute classification performance """
        predicted, correct, accuracy = self.multitask_acc(outputs, labels)

        # Store predictions and labels for the current epoch
        for i in range(self.cfg.cls_tasks):
            epoch_pred[i].extend(predicted[i])
            epoch_gt[i].extend(labels[:, i].cpu())
        return epoch_pred, epoch_gt, accuracy

    def fill_metrics(self, metric, param, true, pred):
        """ Compute errors for the regression tasks and fill in the metrics"""
        # initialize metrics for translation and orientation
        for metric_name, values in metric.items():
            if param not in values.keys():
                values[param] = []

        mae = mean_absolute_error(y_pred=pred, y_true=true)
        metric['MAE'][param].append(mae)

        mse = mean_squared_error(y_pred=pred, y_true=true)
        metric['MSE'][param].append(mse)

        rmse = mean_squared_error(y_pred=pred, y_true=true, squared=False)
        metric['RMSE'][param].append(rmse)

        mape = mean_absolute_percentage_error(y_pred=pred, y_true=true)
        metric['MAPE'][param].append(mape)

    def whole_body_mse(self, ground_truth, prediction):
        """ Computes the MSE for the whole body (all joints together) """
        avg_ground_truth = torch.reshape(ground_truth, (ground_truth.shape[0] * ground_truth.shape[1], 39, 3))
        avg_ground_truth = np.average(avg_ground_truth.detach().cpu().numpy(), axis=2)
        avg_prediction = torch.reshape(prediction, (prediction.shape[0] * prediction.shape[1], 39, 3))
        avg_prediction = np.average(avg_prediction.detach().cpu().numpy(), axis=2)
        mse = mean_squared_error(y_pred=avg_prediction, y_true=avg_ground_truth)
        return mse

    def body_loc_metrics(self, body_metrics, ground_truth, prediction, ds_type):
        """ Metrics to evaluate the body localization """
        # get joint names, depending on the desired body localization setting
        if self.cfg.body_loc_mode == 'wrist_loc':
            joint_names = ["left_wrist", "right_wrist"]
        elif self.cfg.body_loc_mode == 'arm_loc':
            r_arm_indexes, l_arm_indexes = get_arm_and_hand_indexes()
            joint_names = [JOINT_NAMES[i] for i in r_arm_indexes + list(set(l_arm_indexes) - set(r_arm_indexes))]
        elif self.cfg.body_loc_mode == 'body_loc':
            raise NotImplementedError("Full body regression is not supported yet. Make sure the joint order is correct")

        # compute the metrics for each individual joint
        for idx, body_joint in enumerate(joint_names):
            # get true and predicted values for each body joint by iterating each 3 (x, y, z) coordinates
            true = ground_truth[:, :, idx].detach().cpu().numpy()
            pred = prediction[:, :, idx].detach().cpu().numpy()

            # make sure that neck is set as a reference joint (0, 0, 0)
            if body_joint == 'neck':
                assert true[0, 0, :] != torch.Tensor([0, 0, 0])

            # compute the mean for all dimensions / angles
            true_mean = np.mean(true, axis=2)
            pred_mean = np.mean(pred, axis=2)
            self.fill_metrics(body_metrics, body_joint, true_mean, pred_mean)

            # visualize predictions for single sequences in the evaluation
            if ds_type is not None:
                figure, name = create_prediction_figure(true_mean, pred_mean, body_joint)
                self.neptune_run[f"{name}/{ds_type}"].log(figure)

            # compute the Euclidean Distance for translation for each joint
            euclid = translation_distance(true, pred, body_joint, neptune=None)
            body_metrics['Euclid'][body_joint].append(euclid)

            # compute the MSE for the whole body
            mse = self.whole_body_mse(ground_truth, prediction)
            if "global" not in body_metrics["MSE"]:
                body_metrics["MSE"]["global"] = [mse]
            else:
                body_metrics["MSE"]["global"].append(mse)

        # compute the Euclidean Distance for the whole body
        if "global" not in body_metrics["Euclid"]:
            body_metrics["Euclid"]["global"] = [calc_distance_body(ground_truth, prediction, neptune=None)]
        else:
            body_metrics["Euclid"]["global"].append(calc_distance_body(ground_truth, prediction, neptune=None))
        return body_metrics["Euclid"]["global"][-1]

    def obj_pose_metrics(self, obj_metrics, ground_truth, prediction):
        """ Metrics to evaluate the 6D Object Pose Estimation """
        for pose_param in ("transl", "orient"):

            # get true and predicted values for each pose parameter
            true = ground_truth[:, :, :3] if pose_param == "transl" else ground_truth[:, :, 3:]
            pred = prediction[:, :, :3] if pose_param == "transl" else prediction[:, :, 3:]

            # compute the mean for all dimensions / angles
            true = np.mean(true.detach().cpu().numpy(), axis=2)
            pred = np.mean(pred.detach().cpu().numpy(), axis=2)
            self.fill_metrics(obj_metrics, pose_param, true, pred)

        # compute the Euclidean Distance for translation
        if "global" not in obj_metrics["Euclid"]:
            obj_metrics["Euclid"]["global"] = [calc_distance_obj(ground_truth, prediction)]
        else:
            obj_metrics["Euclid"]["global"].append(calc_distance_obj(ground_truth, prediction))
        return obj_metrics["Euclid"]["global"][-1]

    def reg_perf(self, outputs, ground_truth, reg_metrics, ds_type=None):
        """ Compute regression performance """
        current_ed = []
        for i, (task_name, to_process) in enumerate(self.cfg.reg_task_names.items()):
            if not to_process:
                continue
            if task_name == "obj_pose":
                ed = self.obj_pose_metrics(reg_metrics[task_name], ground_truth[task_name], outputs[task_name])
                current_ed.append(ed)
            if task_name == "body_loc":
                ed = self.body_loc_metrics(reg_metrics[task_name], ground_truth[task_name], outputs[task_name], ds_type)
                current_ed.append(ed)
        return reg_metrics, current_ed

    def reg_summary(self, ds_name, regression_metrics, epoch_num):
        """ Generate the epoch summary for the regression tasks """
        task_number = self.cfg.reg_tasks
        epoch_ed = []

        # Iterate through the regression metrics
        for task in regression_metrics.keys():
            for metric, subtask in regression_metrics[task].items():
                for task_id, values in subtask.items():

                    # Skip the empty metrics
                    if len(values) == 0:
                        continue

                    # Generate regression metrics
                    t_val = sum(values) / len(values)
                    self.swriter.add_scalars(f"reg_task_{task}_{metric}/{task_id}", {ds_name: t_val}, epoch_num)
                    self.neptune_run[f"metrics/reg_task_{task}/{metric}/{task_id}/{ds_name}"].log(t_val)

                    # Show the global Euclidean Distance
                    if metric == "Euclid" and task_id == 'global':
                        epoch_ed.append(t_val)
                        logging.info(f'--- Epoch {epoch_num} - reg: avg {ds_name} Euc. Dist. for {task}: {t_val}  ---')

        epoch_ed = round(sum(epoch_ed) / len(epoch_ed), 2)
        self.swriter.add_scalars('total_ed', {f'{ds_name}_ed': epoch_ed}, epoch_num)
        self.neptune_run[f"metrics/reg_overall/total_ed_{task_number}_task(s)/{ds_name}"].log(epoch_ed)
        logging.info(f'--- reg: Average {ds_name} Euc. Dist. for {task_number} regression task(s): {epoch_ed}  ---')
        return epoch_ed

    def cls_summary(self, ds_name, epoch_gt, epoch_pred, epoch_num):
        """ Generate the epoch summary for classification tasks """
        targets = self.target_classes
        task_number = self.cfg.cls_tasks
        epoch_acc = []
        for task_id in range(task_number):

            # Generate confusion matrix
            cm = create_confusion_matrix(epoch_gt[task_id], epoch_pred[task_id], targets[task_id])
            self.swriter.add_figure(f"confusion_matrix_{ds_name}_task_{task_id}", cm, epoch_num)
            self.neptune_run[f"metrics/cls_task{task_id}/confusion_matrix/{ds_name}"].log(cm)

            # Generate classification report
            cr = create_class_report(epoch_gt[task_id], epoch_pred[task_id], targets[task_id])
            self.swriter.add_figure(f"classification_report_{ds_name}_task_{task_id}", cr, epoch_num)
            self.neptune_run[f"metrics/cls_task{task_id}/classification_report/{ds_name}"].log(cr)

            # Computer accuracy
            task_correct = sum(gt == pred for gt, pred in zip(epoch_gt[task_id], epoch_pred[task_id]))
            task_acc = (100 * task_correct / len(epoch_pred[task_id])).item()
            epoch_acc.append(task_acc)
            logging.info(f'--- Epoch {epoch_num} - cls: avg {ds_name} accuracy for task no. {task_id}: {task_acc}%  ---')
            self.swriter.add_scalars(f"cls_task{task_id}_acc", {ds_name: task_acc}, epoch_num)
            self.neptune_run[f"metrics/cls_task{task_id}/acc/{ds_name}"].log(task_acc)

        epoch_acc = round(sum(epoch_acc) / len(epoch_acc), 2)
        self.swriter.add_scalars('total_acc', {f'{ds_name}_acc': epoch_acc}, epoch_num)
        self.neptune_run[f"metrics/cls_overall/total_acc_{task_number}_task(s)/{ds_name}"].log(epoch_acc)
        logging.info(f'--- cls: Average {ds_name} accuracy for {task_number} task(s): {epoch_acc}%  ---')
        return epoch_acc

    def prepare_regression(self, batch):
        """ Prepare output tensors for the regression tasks """
        # prepare object pose estimation
        translation = batch['features']['obj']['transl'].to(self.device)
        global_orient = batch['features']['obj']['global_orient'].to(self.device)
        obj_pose = torch.cat((translation, global_orient), axis=2).to(self.device)

        # prepare body localization: select body localization mode (wrist, arms, whole body)
        body_loc = batch['features']['body'][self.cfg.body_loc_mode].type(torch.FloatTensor).to(self.device)

        # define the ground_truth dictionary
        ground_truth = {}
        if self.cfg.reg_task_names["obj_pose"]:
            ground_truth["obj_pose"] = obj_pose
        if self.cfg.reg_task_names["body_loc"]:
            ground_truth["body_loc"] = body_loc
        return ground_truth

    def prepare_classification(self, batch):
        """ Choose multimodal input features and labels for the neural network """
        multimodal_input = {}
        for input_type, to_process in self.cfg.input_types.items():
            if not to_process:
                continue
            input_data = prepare_inputs(batch, input_type, self.cfg, self.selected_input).to(self.device)
            multimodal_input[input_type] = input_data

        labels = torch.squeeze(batch['labels'], 1).to(self.device)
        return multimodal_input, labels

    def load_data(self, inference_only):
        """ Loads train/val/test data using the pre-defined dataloader """
        kwargs = {'num_workers': self.cfg.n_workers,
                  'batch_size': self.hyperparam_cfg['batch_size'],
                  'shuffle': True,
                  'drop_last': False
                  }

        # Load test data
        ds_test = LoadData(dataset_dir=self.cfg.dataset_dir, data_split='test')
        ds_test = DataLoader(ds_test, **kwargs)
        data_config_path = os.path.join(self.cfg.dataset_dir, 'data_preprocessing_cfg.yaml')
        self.neptune_run['versioning/data_version'].track_files(data_config_path)

        # Get target classes
        target_classes = ds_test.dataset.target_classes
        self.neptune_run['versioning/target_classes'] = target_classes

        # Print the total number of tasks
        logging.info(f'Total number of classification tasks: {self.cfg.cls_tasks}; Config: {self.cfg.cls_task_names}')
        for task_id in range(self.cfg.cls_tasks):
            logging.info(f'Number of target classes for cls task no. {task_id}: {len(target_classes[task_id])}')
        logging.info(f'Total number of regression tasks: {self.cfg.reg_tasks}; Config: {self.cfg.reg_task_names}')

        # For inference mode only, run evaluation on the test data
        if inference_only:
            logging.info(f'Test dataset size (inference only mode): {len(ds_test.dataset)}')
            return _, _, ds_test

        # Load train data
        ds_train = LoadData(dataset_dir=self.cfg.dataset_dir, data_split='train')
        ds_train = DataLoader(ds_train, **kwargs)

        # Load val data
        ds_val = LoadData(dataset_dir=self.cfg.dataset_dir, data_split='val')
        ds_val = DataLoader(ds_val, **kwargs)

        logging.info(f'Dataset Train, Val, Test size respectively:'
                    f' {len(ds_train.dataset)},'
                    f' {len(ds_val.dataset)},'
                    f' {len(ds_test.dataset)}')
        return ds_train, ds_val, ds_test, target_classes

    def create_loss_message(self, loss, epoch, it, mode, acc=None, ed=None):
        """ Generates and logs the loss message with given input parameters """
        exp = self.cfg.expr_ID + str(self.cfg.try_num)
        msg = f'Exp: {exp} - Iter: {it} - Epoch: {epoch} - Model: {self.model_name} - {mode} - Loss: {loss}'
        if acc:
            msg += f' - ACC: {acc}'
        if ed:
            msg += f' - ED: {ed}'
        logging.info(msg)
