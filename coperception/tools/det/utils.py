from coperception.datasets import V2XSimDet
from coperception.configs import Config, ConfigGlobal

from coperception.utils.detection_util import cal_local_mAP, visualization

from box_matching import associate_2_detections

import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader

from coperception.models.det import *
from coperception.utils.loss import *

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         # Input now is [N, 256, 16, 16]
#         self.conv1 = nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=1)  # [N, 64, 8, 8]
#         self.bn1 = nn.BatchNorm2d(64)

#         self.conv2 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1)   # [N, 16, 4, 4]
#         self.bn2 = nn.BatchNorm2d(16)

#         self.fc = nn.Linear(16 * 4 * 4, 1)  # Final binary logit (16 channels × 4 × 4 = 256)

#     def forward(self, x):
#         x = torch.relu(self.bn1(self.conv1(x)))
#         x = torch.relu(self.bn2(self.conv2(x)))
#         x = x.view(x.size(0), -1)  # flatten [N, 256]
#         x = self.fc(x)
#         return x

class Discriminator(nn.Module):
    def __init__(self, base='resnet50', pretrained=True):
        super(Discriminator, self).__init__()
        
        # Load pre-trained model
        if base == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
        elif base == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown base model {base}")
        
        # Replace the very first conv to accept 256 channels
        orig_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=256,  # <== your feature maps
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias is not None
        )

        # Replace the final fully connected layer to output 1 logit
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, 1)

    def forward(self, x):
        return self.backbone(x)  # returns raw logit
def setup_config(args):
    config = Config("train", binary=True, only_det=True)
    config_global = ConfigGlobal("train", binary=True, only_det=True)
    config.inference = args.inference

    if args.bound == "upperbound":
        flag = "upperbound"
    elif args.com == "when2com":
        flag = "who2com" if args.inference == "argmax_test" else "when2com"
        if args.warp_flag:
            flag += "_warp"
    elif args.com in {"v2v", "disco", "sum", "mean", "max", "cat", "agent"}:
        flag = args.com
    else:
        flag = "lowerbound"
        if args.box_com:
            flag += "_box_com"

    config.flag = flag
    config.split = "test"
    print("flag", flag)
    return config, config_global, flag

def build_dataset(args, config, config_global):
    num_agent = args.num_agent
    agent_idx_range = range(1, num_agent) if args.no_cross_road else range(num_agent)
    train_dataset = V2XSimDet(dataset_roots=[f"{args.train_data}/agent{i}" for i in agent_idx_range],
                             config=config,
                             config_global=config_global,
                             split="val",
                             val=True,
                             bound=args.bound,
                             kd_flag=args.kd_flag,
                             no_cross_road=args.no_cross_road)
    
    val_dataset = V2XSimDet(dataset_roots=[f"{args.test_data}/agent{i}" for i in agent_idx_range],
                             config=config,
                             config_global=config_global,
                             split="val",
                             val=True,
                             bound=args.bound,
                             kd_flag=args.kd_flag,
                             no_cross_road=args.no_cross_road)

    return train_dataset, val_dataset, agent_idx_range, num_agent


def build_loaders(args, train_dataset, val_dataset):
    train_loader = DataLoader(
                                train_dataset,
                                batch_size=args.batch,      # ← now driven by your --batch flag
                                shuffle=True,               # it’s almost always best to shuffle train
                                num_workers=args.nworker,
                                pin_memory=True,            # speeds up host→GPU copies
                                prefetch_factor=4,          # overlap data‐load & training
                            )
    val_loader = DataLoader(
                                val_dataset,
                                batch_size=args.batch,
                                shuffle=False,
                                num_workers=args.nworker,
                                pin_memory=True,
                                prefetch_factor=4,
                            )
    return train_loader, val_loader


def initialize_model(args, config, num_agent):
    if args.com == "mean":
        model = MeanFusion(config, layer=args.layer, kd_flag=args.kd_flag, num_agent=num_agent, compress_level=args.compress_level, only_v2i=args.only_v2i)
        model = nn.DataParallel(model)
        model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model

# def init_discriminator_training(device, lr):
#     discriminator = Discriminator().to(device)
    
#     # 1) Separate backbone vs. head parameters
#     backbone_params = []
#     head_params     = []
#     for name, param in discriminator.backbone.named_parameters():
#         if 'fc' in name:
#             head_params.append(param)      # the final classifier layer
#         else:
#             backbone_params.append(param)  # everything else in the pretrained ResNet
    
#     # 2) Build Adam optimizer with two param groups
#     optimizer_disc = torch.optim.Adam([
#         {
#             'params': backbone_params,
#             'lr': lr * 0.1,        # lower LR for pretrained layers
#             'weight_decay': 1e-5,  # lighter regularization
#         },
#         {
#             'params': head_params,
#             'lr': lr,              # full LR for new classification head
#             'weight_decay': 1e-4,  # slightly stronger regularization
#         }
#     ])
    
#     criterion_disc = nn.CrossEntropyLoss()
#     return discriminator, optimizer_disc, criterion_disc

def init_discriminator_training(device, lr):
    # 1) Instantiate and move to device
    discriminator = Discriminator().to(device)
    
    # 2) Freeze all backbone parameters
    for param in discriminator.backbone.parameters():
        param.requires_grad = False

    # 3) Unfreeze only layer4 (the last ResNet block)
    for name, param in discriminator.backbone.named_parameters():
        if name.startswith('layer4'):
            param.requires_grad = True

    # 4) Unfreeze the final fully-connected layer
    for param in discriminator.backbone.fc.parameters():
        param.requires_grad = True

    # 5) Collect all parameters that require gradients
    trainable_params = []
    for param in discriminator.parameters():
        if param.requires_grad:
            trainable_params.append(param)

    # 6) Build optimizer on just those
    optimizer_disc = torch.optim.Adam(
        trainable_params,
        lr=lr,
        weight_decay=1e-4
    )

    # 7) Loss stays the same
    criterion_disc = nn.CrossEntropyLoss()

    return discriminator, optimizer_disc, criterion_disc


def unpack_and_filter_sample(args, sample, device):
    # unzip the sample
    (padded_voxel_point_list, padded_voxel_points_teacher_list, label_one_hot_list,
    reg_target_list, reg_loss_mask_list, anchors_map_list, vis_maps_list, gt_max_iou,
    filenames, target_agent_id_list, num_agent_list, trans_matrices_list) = zip(*sample)

    # extract scene/frame from filename
    filename0 = filenames[0]  #'/home/jasdeep/Spring 2025/CSCE 753 - Computer Vision and Robot Perception/Project/ROBOSAC/V2X-Sim-det/test/agent0/5_0/0.npy'
    filename = str(filename0[0][0])
    cut = filename[filename.rfind('agent') + 7:] #5_0/0.npy
    seq_name = cut[:cut.rfind('_')] #5
    idx = cut[cut.rfind('_') + 1:cut.rfind('/')] #0

    if int(seq_name) not in args.scene_id:
        return None

    num_all_agents = torch.stack(tuple(num_agent_list), 1)
    padded_voxel_points = torch.cat(tuple(padded_voxel_point_list), 0) 
    reg_target = torch.cat(tuple(reg_target_list), 0)
    anchors_map =torch.cat(tuple(anchors_map_list), 0)

    data = {"bev_seq": torch.cat(tuple(padded_voxel_point_list), 0).to(device),
            "bev_seq_teacher": torch.cat(tuple(padded_voxel_points_teacher_list), 0).to(device),
            "labels": torch.cat(tuple(label_one_hot_list), 0).to(device),
            "reg_targets": reg_target.to(device),
            "anchors": anchors_map.to(device),
            "vis_maps": torch.cat(tuple(vis_maps_list), 0).to(device),
            "reg_loss_mask": torch.cat(tuple(reg_loss_mask_list), 0).to(device).type(dtype=torch.bool),
            "target_agent_ids": torch.stack(tuple(target_agent_id_list), 1).to(device),
            "num_agent": torch.stack(tuple(num_agent_list), 1).to(device),
            'ego_agent': args.ego_agent,
            'pert': None,
            'no_fuse': False,
            'collab_agent_list': None,
            'trial_agent_id': None,
            'confidence': None,
            'unadv_pert': None,
            'attacker_list' : None,
            'eps': None,
            "trans_matrices": torch.stack(tuple(trans_matrices_list), 1).to(device)}
    
    return num_agent_list ,num_all_agents, padded_voxel_points, data, reg_target, anchors_map, gt_max_iou, filename0

def get_pseudo_gt(data, fafmodule, batch_size):
    cls_result = fafmodule.cls_predict(data, batch_size, no_fuse=True)
    mean_score = torch.mean(cls_result, dim=2, keepdim=True)
    mask = cls_result > mean_score
    pseudo = mask.clone().detach()
    return pseudo

def init_perturbation(args):
    if args.adv_method == 'pgd':
        return torch.randn(6, 256, 32, 32) * 0.1
    elif args.adv_method in ('bim', 'cw-l2'):
        return torch.zeros(6, 256, 32, 32)
    else:
        raise NotImplementedError(f"Unknown adv_method {args.adv_method}")
    
def run_pgd_attack(data, pseudo_gt, args, fafmodule, device, pert):
    for _ in range(args.adv_iter):
        pert.requires_grad_(True)
        data['pert'] = pert
        loss = fafmodule.cls_step(
            data, args.batch, 
            ego_loss_only=args.ego_loss_only, 
            ego_agent=args.ego_agent, 
            invert_gt=True, 
            self_result=pseudo_gt, 
            adv_method=args.adv_method
        )
        # pert = (pert + args.pert_alpha * pert.grad.sign() * -1).detach()
        pert = (pert + args.pert_alpha * pert.grad.sign() * -1).clamp(-args.eps, args.eps).detach()

    return pert.clone().detach()


def extract_agent_features(num_all_agents, padded_voxel_points, model, device):
    agent_feats = []
    # Add a 1x1 convolution to adjust channels if not already defined
    if not hasattr(model, 'channel_adjust'):
        model.channel_adjust = nn.Conv2d(512, 256, kernel_size=1).to(device)
    
    for j in range(num_all_agents[0][0]):
        single_bev = padded_voxel_points[j].permute(0, 3, 1, 2).unsqueeze(0).to(device)  # [1, 13, 256, 256]
        
        # Forward through backbone
        feat_j = model.module.u_encoder(single_bev)[-1]  # Original shape: [1, 512, 16, 16]
        
        # Resize spatial dimensions to 32x32
        feat_j = F.interpolate(feat_j, size=(32, 32), mode='bilinear', align_corners=True)
        
        # Adjust channels from 512 to 256
        feat_j = model.channel_adjust(feat_j)  # New shape: [1, 256, 32, 32]
        agent_feats.append(feat_j.squeeze(0))
    return agent_feats


def time_str():
    t = time.time()- 60*60*24*30
    time_string = time.strftime("%Y_%m_%d_%H:%M:%S", time.localtime(t))
    return time_string

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

def visualize(args, config, filename0, save_fig_path, fafmodule, data, num_agent_list, padded_voxel_point, gt_max_iou, vis_tag):
    print("Visualizing: {}".format(vis_tag))
    det_results_local = [[] for i in range(6)]
    annotations_local = [[] for i in range(6)]

    padded_voxel_point = data['bev_seq']
    padded_voxel_points_teacher = data['bev_seq_teacher']
    reg_target = data['reg_targets']
    anchors_map = data['anchors']

    loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=num_agent_list[0][0])
            
    # local qualitative evaluation
    num_sensor = num_agent_list[0][0].numpy()
    print(f'num_sensor: {num_sensor}')
    for k in range(num_sensor):
        data_agents = {'bev_seq': torch.unsqueeze(padded_voxel_point[k, :, :, :, :], 1),
                    'bev_seq_teacher': torch.unsqueeze(padded_voxel_points_teacher[k, :, :, :, :], 1),
                    'reg_targets': torch.unsqueeze(reg_target[k, :, :, :, :, :], 0),
                    'anchors': torch.unsqueeze(anchors_map[k, :, :, :, :], 0)}
        temp = gt_max_iou[k]
        data_agents['gt_max_iou'] = temp[0]['gt_box'][0, :, :]
        result_temp = result[k]
        
        temp = {'bev_seq': data_agents['bev_seq'][0, -1].cpu().numpy(), 
                'bev_seq_teacher': data_agents['bev_seq_teacher'][0, -1].cpu().numpy(),
                'result': result_temp[0][0],
                'reg_targets': data_agents['reg_targets'].cpu().numpy()[0],
                'anchors_map': data_agents['anchors'].cpu().numpy()[0],
                'gt_max_iou': data_agents['gt_max_iou'],
                'vis_tag': vis_tag}
        
        det_results_local[k], annotations_local[k] = cal_local_mAP(config, temp, det_results_local[k], annotations_local[k])
        print("Agent {}:".format(k))
        filename = str(filename0[0][0])
        cut = filename[filename.rfind('agent') + 7:]
        seq_name = cut[:cut.rfind('_')]
        idx = cut[cut.rfind('_') + 1:cut.rfind('/')]
        seq_save = os.path.join(save_fig_path[k], seq_name)
        check_folder(seq_save)
        idx_save = '{}_{}.png'.format(str(idx), vis_tag)

        if args.visualization:
            visualization(config, temp, None, None, 0, os.path.join(seq_save, idx_save))

def cal_robosac_consensus(num_agent, step_budget, num_attackers):
    num_agent = num_agent - 1
    eta = num_attackers / num_agent
    s = np.floor(np.log(1-np.power(1-0.99, 1/step_budget)) / np.log(1-eta)).astype(int)
    return s

def get_jaccard_index(args, config, num_agent_list, padded_voxel_point, reg_target, anchors_map, gt_max_iou, result_1, result_2):
    num_sensor = num_agent_list[0][0].numpy()
    det_results_local_1 = [[] for i in range(num_sensor)]
    annotations_local_1 = [[] for i in range(num_sensor)]
    det_results_local_2 = [[] for i in range(num_sensor)]
    annotations_local_2 = [[] for i in range(num_sensor)]
    ego_idx = args.ego_agent
    # for k in range(num_sensor):
    data_agents = {'bev_seq': torch.unsqueeze(padded_voxel_point[ego_idx, :, :, :, :], 1),
                'reg_targets': torch.unsqueeze(reg_target[ego_idx, :, :, :, :, :], 0),
                'anchors': torch.unsqueeze(anchors_map[ego_idx, :, :, :, :], 0)}
    temp = gt_max_iou[ego_idx]
    data_agents['gt_max_iou'] = temp[0]['gt_box'][0, :, :]
    result_temp_1 = result_1[ego_idx]
    result_temp_2 = result_2[ego_idx]
    temp_1 = {'bev_seq': data_agents['bev_seq'][0, -1].cpu().numpy(), 'result': result_temp_1[0][0],
            'reg_targets': data_agents['reg_targets'].cpu().numpy()[0],
            'anchors_map': data_agents['anchors'].cpu().numpy()[0],
            'gt_max_iou': data_agents['gt_max_iou']}
    temp_2 = {'bev_seq': data_agents['bev_seq'][0, -1].cpu().numpy(), 'result': result_temp_2[0][0],
            'reg_targets': data_agents['reg_targets'].cpu().numpy()[0],
            'anchors_map': data_agents['anchors'].cpu().numpy()[0],
            'gt_max_iou': data_agents['gt_max_iou']}
    
    det_results_local_1[ego_idx], annotations_local_1[ego_idx] = cal_local_mAP(config, temp_1, det_results_local_1[ego_idx], annotations_local_1[ego_idx])
    det_results_local_2[ego_idx], annotations_local_2[ego_idx] = cal_local_mAP(config, temp_2, det_results_local_2[ego_idx], annotations_local_2[ego_idx])
    
    print("Calculating in the view of Agent {}:".format(ego_idx))
    # shape of det_results_local_1 [k][0][0] is (N, 9)
    # The final value of the array is confidence. Ignored
    if len(det_results_local_1[ego_idx]) == 0:
        # if ego have no detection, return 0
        return 0 
    det_1 = det_results_local_1[ego_idx][0][0][:,0:8]
    det_2 = det_results_local_2[ego_idx][0][0][:,0:8]
    # jac_index = calculate_jaccard(det_results_local_1[k][0][0], det_results_local_2[k][0][0])
    jac_index = associate_2_detections(det_1, det_2)
    return jac_index


def local_eval(num_agent, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result, config, det_results_local, annotations_local):
    # If has RSU, do not count RSU's output into evaluation
    # eval_start_idx = 0 if args.no_cross_road else 1
    eval_start_idx = 0
    # update global result
    for k in range(eval_start_idx, num_agent):
        data_agents = {
            "bev_seq": torch.unsqueeze(padded_voxel_points[k, :, :, :, :], 1),
            "reg_targets": torch.unsqueeze(reg_target[k, :, :, :, :, :], 0),
            "anchors": torch.unsqueeze(anchors_map[k, :, :, :, :], 0),
        }
        temp = gt_max_iou[k]

        if len(temp[0]["gt_box"]) == 0:
            data_agents["gt_max_iou"] = []
        else:
            data_agents["gt_max_iou"] = temp[0]["gt_box"][0, :, :]


        result_temp = result[k]

        temp = {
            "bev_seq": data_agents["bev_seq"][0, -1].cpu().numpy(),
            "result": [] if len(result_temp) == 0 else result_temp[0][0],
            "reg_targets": data_agents["reg_targets"].cpu().numpy()[0],
            "anchors_map": data_agents["anchors"].cpu().numpy()[0],
            "gt_max_iou": data_agents["gt_max_iou"],
        }
        det_results_local[k], annotations_local[k] = cal_local_mAP(
            config, temp, det_results_local[k], annotations_local[k]
        )
    return det_results_local, annotations_local

def cal_robosac_steps(num_agent, num_consensus, num_attackers):
    # exclude ego agent
    num_agent = num_agent - 1
    eta = num_attackers / num_agent
    # print(f'eta: {eta}')
    # print(f's(num_agent): {num_agent}')
    N = np.ceil(np.log(1 - 0.99) / np.log(1 - np.power(1 - eta, num_consensus))).astype(int)
    return N
