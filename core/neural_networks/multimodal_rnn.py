""" Class of the Multimodal Recurrent Neural Network """

import torch
import torch.nn as nn


class MultiModalRNN(nn.Module):
    def __init__(self, tensors_config, hyperparam_cfg, multimodalities):
        super(MultiModalRNN, self).__init__()

        # computes touch data
        if tensors_config["contact"]:
            self.MMRNN_Touch = MMRNN_Touch(touch_config=tensors_config["contact"],
                                           hyperparam_cfg=hyperparam_cfg)

        # computes proprioceptive data
        if tensors_config["fullpose"]:
            self.MMRNN_Fullpose = MMRNN_Fullpose(proprio_config=tensors_config["fullpose"],
                                                 hyperparam_cfg=hyperparam_cfg)

        # heads for multitasking
        self.linear_object_class = nn.Linear(hyperparam_cfg['hidden'][0] * multimodalities, 51)
        self.linear_action_reco = nn.Linear(hyperparam_cfg['hidden'][0] * multimodalities, 4)
        self.linear_object_orient = nn.Linear(hyperparam_cfg['hidden'][0] * multimodalities, 6)
        self.linear_body_loc = nn.Linear(hyperparam_cfg['hidden'][0] * multimodalities, 39 * 3)

    def forward(self, input_vector, cfg):
        cls_results, reg_results = [], []

        # process the multimodalities by individual backbone(s)
        for input_type, vector in input_vector.items():
            if input_type == "contact":
                x_class, x_reg = self.MMRNN_Touch(touch_input=input_vector['contact'])
            if input_type == "fullpose":
                x_class, x_reg = self.MMRNN_Fullpose(fullpose_input=input_vector['fullpose'])
                joint_number = input_vector['fullpose'].shape[-2]
            cls_results.append(x_class)
            reg_results.append(x_reg)

        # integrate the multimodalities for classification and regression tasks
        x_class = torch.cat(cls_results, axis=1)
        x_reg = torch.cat(reg_results, axis=2)

        # process the multitasking by individual head(s)
        model_predictions = {}
        if cfg.cls_task_names["obj_cat"]:
            x_object_class = self.linear_object_class(x_class)
            model_predictions["obj_cat"] = x_object_class
        if cfg.cls_task_names["act_rec"]:
            x_action_reco = self.linear_action_reco(x_class)
            model_predictions["act_rec"] = x_action_reco
        if cfg.reg_task_names["obj_pose"]:
            x_object_orient = self.linear_object_orient(x_reg)
            model_predictions["obj_pose"] = x_object_orient
        if cfg.reg_task_names["body_loc"]:
            x_body_loc = self.linear_body_loc(x_reg)
            model_predictions["body_loc"] = x_body_loc.view(*x_body_loc.size()[:-1], joint_number, 3).float()
        return model_predictions


class MMRNN_Touch(nn.Module):
    def __init__(self, touch_config, hyperparam_cfg):
        super(MMRNN_Touch, self).__init__()
        self.lstm = nn.LSTM(input_size=touch_config["num_nodes"],
                            hidden_size=hyperparam_cfg['hidden'][0],
                            num_layers=1,
                            batch_first=True)

    def forward(self, touch_input):
        x_reg, (hn, cn) = self.lstm(touch_input)
        x_class = x_reg[:, -1, :]
        return x_class, x_reg


class MMRNN_Fullpose(nn.Module):
    def __init__(self, proprio_config, hyperparam_cfg):
        super(MMRNN_Fullpose, self).__init__()
        self.lstm = nn.LSTM(input_size=proprio_config["num_nodes"] * 4,
                            hidden_size=hyperparam_cfg['hidden'][0],
                            num_layers=1,
                            batch_first=True)

    def forward(self, fullpose_input):
        # sqeeze the input for the RNN to be (batch_size, seq_len, num_nodes * 4)
        fullpose_input = fullpose_input.view(*fullpose_input.size()[:-2], fullpose_input.size()[-2] * fullpose_input.size()[-1]).float()

        x_reg, (hn, cn) = self.lstm(fullpose_input)
        x_class = x_reg[:, -1, :]
        return x_class, x_reg
