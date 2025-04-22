import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from coperception.datasets import V2XSimDet
from coperception.configs import Config, ConfigGlobal
from coperception.utils.CoDetModule import FaFModule
from coperception.utils.loss import SoftmaxFocalClassificationLoss, WeightedSmoothL1LocalizationLoss
from coperception.utils.mean_ap import eval_map
from coperception.models.det import MeanFusion
def check_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    return folder_path
def local_eval(num_agent, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result, config, det_results_local, annotations_local):
    for k in range(num_agent):
        data_agents = {
            "bev_seq": torch.unsqueeze(padded_voxel_points[k], 1),
            "reg_targets": torch.unsqueeze(reg_target[k], 0),
            "anchors": torch.unsqueeze(anchors_map[k], 0),
        }
        temp = gt_max_iou[k]
        data_agents["gt_max_iou"] = temp[0]["gt_box"][0] if len(temp[0]["gt_box"]) > 0 else []
        result_temp = result[k]
        temp = {
            "bev_seq": data_agents["bev_seq"][0, -1].cpu().numpy(),
            "result": [] if len(result_temp) == 0 else result_temp[0][0],
            "reg_targets": data_agents["reg_targets"].cpu().numpy()[0],
            "anchors_map": data_agents["anchors"].cpu().numpy()[0],
            "gt_max_iou": data_agents["gt_max_iou"],
        }
        from coperception.utils.detection_util import cal_local_mAP
        det_results_local[k], annotations_local[k] = cal_local_mAP(
            config, temp, det_results_local[k], annotations_local[k]
        )
    return det_results_local, annotations_local
def main(args):
    config = Config("train", binary=True, only_det=True)
    config_global = ConfigGlobal("train", binary=True, only_det=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation_dataset = V2XSimDet(
        dataset_roots=[f"{args.data}/agent{i}" for i in range(args.num_agent)],
        config=config,
        config_global=config_global,
        split="val",
        val=True,
        bound="both",
        kd_flag=0,
        no_cross_road=False,
    )
    validation_data_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=args.nworker)
    model = MeanFusion(config, layer=args.layer, kd_flag=0, num_agent=args.num_agent)
    model = nn.DataParallel(model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = {
        "cls": SoftmaxFocalClassificationLoss(),
        "loc": WeightedSmoothL1LocalizationLoss(),
    }
    fafmodule = FaFModule(model, model, config, optimizer, criterion, kd_flag=0)
    checkpoint = torch.load(args.resume, map_location="cpu")
    fafmodule.model.load_state_dict(checkpoint["model_state_dict"])
    fafmodule.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    fafmodule.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    det_results_local = [[] for _ in range(args.num_agent)]
    annotations_local = [[] for _ in range(args.num_agent)]
    for sample in tqdm(validation_data_loader):
        (
            padded_voxel_point_list,
            padded_voxel_points_teacher_list,
            label_one_hot_list,
            reg_target_list,
            reg_loss_mask_list,
            anchors_map_list,
            vis_maps_list,
            gt_max_iou,
            filenames,
            target_agent_id_list,
            num_agent_list,
            trans_matrices_list,
        ) = zip(*sample)
        padded_voxel_points = torch.cat(padded_voxel_point_list, 0).to(device)
        padded_voxel_points_teacher = torch.cat(padded_voxel_points_teacher_list, 0).to(device)
        label_one_hot = torch.cat(label_one_hot_list, 0).to(device)
        reg_target = torch.cat(reg_target_list, 0).to(device)
        reg_loss_mask = torch.cat(reg_loss_mask_list, 0).to(device).bool()
        anchors_map = torch.cat(anchors_map_list, 0).to(device)
        vis_maps = torch.cat(vis_maps_list, 0).to(device)
        trans_matrices = torch.stack(tuple(trans_matrices_list), 1).to(device)
        target_agent_ids = torch.stack(tuple(target_agent_id_list), 1).to(device)
        num_all_agents = torch.stack(tuple(num_agent_list), 1).to(device)
        data = {
            "bev_seq": padded_voxel_points,
            "bev_seq_teacher": padded_voxel_points_teacher,
            "labels": label_one_hot,
            "reg_targets": reg_target,
            "anchors": anchors_map,
            "vis_maps": vis_maps,
            "reg_loss_mask": reg_loss_mask,
            "target_agent_ids": target_agent_ids,
            "num_agent": num_all_agents,
            "ego_agent": 1,
            "pert": None,
            "no_fuse": True,
            "collab_agent_list": None,
            "trial_agent_id": None,
            "confidence": None,
            "unadv_pert": None,
            "attacker_list": None,
            "eps": None,
            "trans_matrices": trans_matrices,
        }
        cls_result = fafmodule.cls_predict(data, args.batch, no_fuse=True)
        mean = torch.mean(cls_result, dim=2)
        cls_result[:, :, 0] = cls_result[:, :, 0] > mean
        cls_result[:, :, 1] = cls_result[:, :, 1] > mean
        pseudo_gt = cls_result.clone().detach()
        pert = torch.randn(6, 256, 32, 32).to(device) * 0.1
        for _ in range(args.adv_iter):
            pert.requires_grad = True
            data["pert"] = pert
            from coperception.utils.CoDetModule import cls_step
            fafmodule.cls_step(data, args.batch, ego_loss_only=False, ego_agent=1, invert_gt=True, self_result=pseudo_gt, adv_method="pgd")
            pert = pert + args.pert_alpha * pert.grad.sign() * -1
            pert = pert.detach()
        data["pert"] = pert
        data["no_fuse"] = False
        _, _, _, result = fafmodule.predict_all(data, 1, num_agent=args.num_agent)
        det_results_local, annotations_local = local_eval(
            args.num_agent, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result, config, det_results_local, annotations_local
        )
    for k in range(args.num_agent):
        print(f"Agent {k} mAP@0.5:")
        mean_ap, _ = eval_map(det_results_local[k], annotations_local[k], iou_thr=0.5)
        print(mean_ap)
        print(f"Agent {k} mAP@0.7:")
        mean_ap, _ = eval_map(det_results_local[k], annotations_local[k], iou_thr=0.7)
        print(mean_ap)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="{Your_location_to_V2X-Sim}/V2X-Sim/test", type=str)
    parser.add_argument("--batch", default=1, type=int)
    parser.add_argument("--nworker", default=4, type=int)
    parser.add_argument("--resume", default="../../ckpt/meanfusion/epoch_49.pth", type=str)
    parser.add_argument("--layer", default=3, type=int)
    parser.add_argument("--num_agent", default=6, type=int)
    parser.add_argument("--pert_alpha", type=float, default=0.1)
    parser.add_argument("--adv_iter", type=int, default=15)
    args = parser.parse_args()
    main(args)