import numpy as np
import os
from coperception.datasets import V2XSimDet
from torch.utils.data import DataLoader
from tqdm import tqdm

import argparse
from itertools import count
import os
from copy import deepcopy

import seaborn as sns
import torch.optim as optim
from torch.utils.data import DataLoader

from coperception.datasets import V2XSimDet
from coperception.configs import Config, ConfigGlobal
from coperception.utils.CoDetModule import *
from coperception.utils.loss import *
from coperception.utils.mean_ap import eval_map
from coperception.models.det import *
from coperception.utils.detection_util import late_fusion
from coperception.utils.data_util import apply_pose_noise
import random
from tqdm import tqdm
from torch.autograd import Variable
from box_matching import associate_2_detections


# def find_filenames_with_six_agents(dataset_roots, config, config_global, split="val", val=True, bound=None, kd_flag=False, no_cross_road=False):
#     """
#     Iterates through the V2X-Dataset-det and returns filenames for scenes with exactly 6 agents.

#     Args:
#         dataset_roots (list): List of paths to the agent directories.
#         config (dict): Configuration dictionary.
#         config_global (dict): Global configuration dictionary.
#         split (str): The split of the dataset to use (e.g., "train", "val").
#         val (bool): Whether it's the validation split.
#         bound (list, optional): Bounding box parameters. Defaults to None.
#         kd_flag (bool): Knowledge distillation flag. Defaults to False.
#         no_cross_road (bool): Flag to exclude cross-road scenes. Defaults to False.

#     Returns:
#         list: A list of filenames corresponding to scenes with exactly 6 agents.
#     """

#     agent_idx_range = range(1, 6) if args.no_cross_road else range(6)
#     validation_dataset = V2XSimDet(
#         dataset_roots=[f"{args.data}/agent{i}" for i in agent_idx_range],
#         config=config,
#         config_global=config_global,
#         split=split,
#         val=val,
#         bound=bound,
#         kd_flag=kd_flag,
#         no_cross_road=no_cross_road,
#     )

#     validation_data_loader = DataLoader(
#         validation_dataset, batch_size=1, shuffle=False, num_workers=1
#     )

#     filenames_with_six_agents = []
#     for cnt, sample in enumerate(tqdm(validation_data_loader)):
#         (
#             padded_voxel_point_list,
#             padded_voxel_points_teacher_list,
#             label_one_hot_list,
#             reg_target_list,
#             reg_loss_mask_list,
#             anchors_map_list,
#             vis_maps_list,
#             gt_max_iou,
#             filenames,
#             target_agent_id_list,
#             num_agent_list,
#             trans_matrices_list,
#         ) = zip(*sample)

#         if num_agent_list[0] == 6:  # Check the number of agents in the scene
#             filename0 = filenames[0]
#             filename = str(filename0[0][0])
#             filenames_with_six_agents.append(filename)

#     return filenames_with_six_agents

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


def main(args):
    config = Config("train", binary=True, only_det=True)
    config_global = ConfigGlobal("train", binary=True, only_det=True)

    num_workers = 0


    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)


    flag = "mean" 

    config.split = "test"

    num_agent = 6
    # agent0 is the cross road
    agent_idx_range = range(1, num_agent) if args.no_cross_road else range(num_agent)
    validation_dataset = V2XSimDet(
        dataset_roots=[f"{args.data}/agent{i}" for i in agent_idx_range],
        config=config,
        config_global=config_global,
        split="val",
        val=True,
        bound=args.bound,
        kd_flag=args.kd_flag,
        no_cross_road=args.no_cross_road,
    )
    validation_data_loader = DataLoader(
        validation_dataset, batch_size=1, shuffle=False, num_workers=num_workers
    )
    print("Validation dataset size:", len(validation_dataset))

    if args.no_cross_road:
        num_agent -= 1


    filenames_with_six_agents = []
    for cnt, sample in enumerate(tqdm(validation_data_loader)):
        t = time.time()

        
        # Extract data from data loader
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

        filename0 = filenames[0]
        filename = str(filename0[0][0])
        cut = filename[filename.rfind('agent') + 7:]
        seq_name = cut[:cut.rfind('_')]
        idx = cut[cut.rfind('_') + 1:cut.rfind('/')]
        
        if num_agent_list[0] == 6:  # Check the number of agents in the scene
            filename0 = filenames[0]
            filename = str(filename0[0][0])
            filenames_with_six_agents.append(filename)

        # trans_matrices = torch.stack(tuple(trans_matrices_list), 1)
        # target_agent_ids = torch.stack(tuple(target_agent_id_list), 1)
        # num_all_agents = torch.stack(tuple(num_agent_list), 1)


        # if args.no_cross_road:
        #     num_all_agents -= 1
        # padded_voxel_points = torch.cat(tuple(padded_voxel_point_list), 0)
        # padded_voxel_points_teacher = torch.cat(tuple(padded_voxel_points_teacher_list), 0)

        # label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
        # reg_target = torch.cat(tuple(reg_target_list), 0)
        # reg_loss_mask = torch.cat(tuple(reg_loss_mask_list), 0)
        # anchors_map = torch.cat(tuple(anchors_map_list), 0)
        # vis_maps = torch.cat(tuple(vis_maps_list), 0)

        # data = {
        #     "bev_seq": padded_voxel_points.to(device),
        #     "bev_seq_teacher": padded_voxel_points_teacher.to(device),
        #     "labels": label_one_hot.to(device),
        #     "reg_targets": reg_target.to(device),
        #     "anchors": anchors_map.to(device),
        #     "vis_maps": vis_maps.to(device),
        #     "reg_loss_mask": reg_loss_mask.to(device).type(dtype=torch.bool),
        #     "target_agent_ids": target_agent_ids.to(device),
        #     "num_agent": num_all_agents.to(device),
        #     'ego_agent': args.ego_agent,
        #     'pert': None,
        #     'no_fuse': False,
        #     'collab_agent_list': None,
        #     'trial_agent_id': None,
        #     'confidence': None,
        #     'unadv_pert': None,
        #     'attacker_list' : None,
        #     'eps': None,
        #     "trans_matrices": trans_matrices.to(device),
        # }

    return filenames_with_six_agents


# Example usage (assuming you have 'args', 'config', and 'config_global' defined):
if __name__ == '__main__':
    class Args:
        def __init__(self):
            self.data = "/mnt/f/V2X-Sim/V2X-Sim-det/V2X-Sim-det/train"  # Replace with the actual path
            self.bound = None
            self.kd_flag = False
            self.no_cross_road = False
            self.scene_id = None  # You might want to filter by specific scene IDs as well
            self.sample_id = None

    args = Args()
    agent_idx_range = range(6) # Assuming a maximum of 6 agents as per the dataset structure


    dataset_roots = [f"{args.data}/agent{i}" for i in agent_idx_range]

    # six_agent_filenames = find_filenames_with_six_agents(
    #     dataset_roots=dataset_roots,
    #     config=config,
    #     config_global=config_global,
    #     split="val",
    #     val=True,
    #     bound=args.bound,
    #     kd_flag=args.kd_flag,
    #     no_cross_road=args.no_cross_road,
    # )
    six_agent_filenames = main(args)

    print("Filenames for scenes with exactly 6 agents:")
    for filename in six_agent_filenames:
        print(filename)





#############

