""" Class of the Multimodal Graph Neural Network """

import torch
import torch.nn as nn

# NOTE: Patched funtions of torch_geometric_temporal
from core.neural_networks.patches.patch_tsagcn import patched_forward
from core.neural_networks.patches.patch_aagcn import AAGCN


class MultiModalGCN(nn.Module):
    def __init__(self, graphs_config, hyperparam_cfg, multimodalities):
        super(MultiModalGCN, self).__init__()
        graph_nodes = 0

        # computes touch data
        if graphs_config["contact"]:
            graph_nodes += 778
            self.MMGCN_Touch = MMGCN_Touch(touch_config=graphs_config["contact"],
                                           hyperparam_cfg=hyperparam_cfg)

        # computes fullpose data
        if graphs_config["fullpose"]:
            graph_nodes += 20
            self.MMGCN_Fullpose = MMGCN_Fullpose(proprio_config=graphs_config["fullpose"],
                                                 hyperparam_cfg=hyperparam_cfg)

        # computes joints data
        if graphs_config["joints_15"]:
            graph_nodes += 15
            self.MMGCN_Joints = MMGCN_Joints(joints_config=graphs_config["joints_15"],
                                             hyperparam_cfg=hyperparam_cfg)

        # computes joints data
        if graphs_config["joints_21"]:
            graph_nodes += 21
            self.MMGCN_Joints = MMGCN_Joints(joints_config=graphs_config["joints_21"],
                                             hyperparam_cfg=hyperparam_cfg)

        # heads for multitasking - classification
        self.linear_object_class = nn.Linear(hyperparam_cfg['hidden'][1] * multimodalities, 51)
        self.linear_action_reco = nn.Linear(hyperparam_cfg['hidden'][1] * multimodalities, 4)

        # heads for multitasking - regression
        out_shape = graph_nodes * 2 - 1
        self.linear_object_orient = nn.Linear(out_shape, 6)
        self.linear_body_loc = nn.Linear(out_shape, 117)

    def forward(self, input_vector, cfg):
        cls_results, reg_results = [], []

        # process the multimodalities by individual backbone(s)
        for input_type, vector in input_vector.items():
            x = vector.x.permute(0, 3, 1, 2).contiguous()  # Batch, Features_In, Temporal_In, Node_Number
            if input_type == "contact":
                x_class, x_reg = self.MMGCN_Touch(touch_input=x)
            if input_type == "fullpose":
                x_class, x_reg = self.MMGCN_Fullpose(fullpose_input=x)
                joint_number = x.shape[-1]
            if "joints" in input_type:
                x_class, x_reg = self.MMGCN_Joints(joints_input=x)
                joint_number = x.shape[-1]
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


class MMGCN_Touch(nn.Module):
    def __init__(self, touch_config, hyperparam_cfg):
        super(MMGCN_Touch, self).__init__()

        self.agcn1 = AAGCN(in_channels=1,
                           out_channels=hyperparam_cfg['hidden'][0],
                           edge_index=touch_config["edges"],
                           num_nodes=touch_config["num_nodes"],
                           adaptive=hyperparam_cfg["adaptive"],
                           attention=hyperparam_cfg["attention"],
                           kernel_size=hyperparam_cfg["kernel_size"])

        self.agcn2 = AAGCN(in_channels=hyperparam_cfg['hidden'][0],
                           out_channels=hyperparam_cfg['hidden'][1],
                           edge_index=touch_config["edges"],
                           num_nodes=touch_config["num_nodes"],
                           adaptive=hyperparam_cfg["adaptive"],
                           attention=hyperparam_cfg["attention"],
                           kernel_size=hyperparam_cfg["kernel_size"])

        self.drop_out = nn.Dropout(0.5)

    def forward(self, touch_input):
        x = self.agcn1(touch_input)
        x = self.drop_out(x)
        x = self.agcn2(x)
        x_class = x.mean(3).mean(2)
        x_reg = x.mean(1)
        return x_class, x_reg


class MMGCN_Fullpose(nn.Module):
    def __init__(self, proprio_config, hyperparam_cfg):
        super(MMGCN_Fullpose, self).__init__()

        self.agcn1 = AAGCN(in_channels=4,
                           out_channels=hyperparam_cfg['hidden'][0],
                           edge_index=proprio_config["edges"],
                           num_nodes=proprio_config["num_nodes"],
                           adaptive=hyperparam_cfg["adaptive"],
                           attention=hyperparam_cfg["attention"],
                           kernel_size=hyperparam_cfg["kernel_size"])

        self.agcn2 = AAGCN(in_channels=hyperparam_cfg['hidden'][0],
                           out_channels=hyperparam_cfg['hidden'][1],
                           edge_index=proprio_config["edges"],
                           num_nodes=proprio_config["num_nodes"],
                           adaptive=hyperparam_cfg["adaptive"],
                           attention=hyperparam_cfg["attention"],
                           kernel_size=hyperparam_cfg["kernel_size"])

        self.drop_out = nn.Dropout(0.5)

    def forward(self, fullpose_input):
        x = self.agcn1(fullpose_input)
        x = self.drop_out(x)
        x = self.agcn2(x)
        x_class = x.mean(3).mean(2)
        x_reg = x.mean(1)
        return x_class, x_reg


class MMGCN_Joints(nn.Module):
    def __init__(self, joints_config, hyperparam_cfg):
        super(MMGCN_Joints, self).__init__()

        self.agcn1 = AAGCN(in_channels=3,
                           out_channels=hyperparam_cfg['hidden'][0],
                           edge_index=joints_config["edges"],
                           num_nodes=joints_config["num_nodes"],
                           adaptive=hyperparam_cfg["adaptive"],
                           attention=hyperparam_cfg["attention"],
                           kernel_size=hyperparam_cfg["kernel_size"])

        self.agcn2 = AAGCN(in_channels=hyperparam_cfg['hidden'][0],
                           out_channels=hyperparam_cfg['hidden'][1],
                           edge_index=joints_config["edges"],
                           num_nodes=joints_config["num_nodes"],
                           adaptive=hyperparam_cfg["adaptive"],
                           attention=hyperparam_cfg["attention"],
                           kernel_size=hyperparam_cfg["kernel_size"])

        self.drop_out = nn.Dropout(0.5)

    def forward(self, joints_input):
        x = self.agcn1(joints_input)
        x = self.drop_out(x)
        x = self.agcn2(x)
        x_class = x.mean(3).mean(2)
        x_reg = x.mean(1)
        return x_class, x_reg
