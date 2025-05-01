import sys
import argparse
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import random

from tqdm import tqdm
from utils import *
from coperception.utils.CoDetModule import FaFModule
from coperception.utils.mean_ap import eval_map



def main(args):
    config, config_global, flag = setup_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_agent = args.num_agent
    agent_idx_range = range(1, num_agent) if args.no_cross_road else range(num_agent)
    test_dataset = V2XSimDet(dataset_roots=[f"{args.test_data}/agent{i}" for i in agent_idx_range],
                                config=config,
                                config_global=config_global,
                                split="val",
                                val=True,
                                bound=args.bound,
                                kd_flag=args.kd_flag,
                                no_cross_road=args.no_cross_road)

    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=args.nworker, pin_memory=True, prefetch_factor=4)


    model = initialize_model(args, config, num_agent)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = {"cls": SoftmaxFocalClassificationLoss(), "loc": WeightedSmoothL1LocalizationLoss(),}

    fafmodule = FaFModule(model, model, config, optimizer, criterion, args.kd_flag)
    model_save_path = args.resume[: args.resume.rfind("/")]
    os.makedirs(model_save_path, exist_ok=True)
    checkpoint = torch.load(args.resume, map_location="cpu")
    start_epoch = checkpoint["epoch"] + 1
    fafmodule.model.load_state_dict(checkpoint["model_state_dict"])
    fafmodule.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    fafmodule.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))

    fafmodule.model.eval()

    save_fig_path = [check_folder(os.path.join(model_save_path, f"vis{i}")) for i in agent_idx_range]

    det_results_local = [[] for i in agent_idx_range]
    annotations_local = [[] for i in agent_idx_range]

    for k, v in fafmodule.model.named_parameters():
        v.requires_grad = False  # fix parameters

    discriminator, optimizer_disc, criterion_disc = init_discriminator_training(device, args.lr)
    scheduler_disc = CosineAnnealingLR(optimizer_disc, T_max=args.epochs, eta_min=1e-6)
    checkpoint = torch.load("discriminator_checkpoint.pth", map_location=device)
    discriminator.load_state_dict(checkpoint['model_state_dict'])
    optimizer_disc.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler_disc.load_state_dict(checkpoint["scheduler_state_dict"])
    discriminator.eval()  # very important to set it to eval mode for inference
    print("Loaded discriminator checkpoint successfully!")

    frame_seq = 0
    succ = 0
    fail = 0

    frame_count = 300
    steps = np.zeros(frame_count)
    ego_steps = np.zeros(frame_count)


    val_loss = 0.0
    val_correct = 0
    val_total = 0

    detection_times  = []          
    consensus_times  = [] 


    for cnt, sample in enumerate(tqdm(test_loader)):

        unpacked = unpack_and_filter_sample(args, sample, device)
        if unpacked is None:
            continue

        frame_seq += 1
        num_agent_list ,num_all_agents, padded_voxel_points, data, reg_target, anchors_map, gt_max_iou, filename0 = unpacked        

        pseudo_gt = get_pseudo_gt(data, fafmodule, args.batch)

        if args.visualization:
            # visulize ego only det result, without fusion
            data['no_fuse'] = True
            visualize(args, config, filename0, save_fig_path, fafmodule, data, num_agent_list, padded_voxel_points, gt_max_iou, vis_tag='ego_only')

        pert = init_perturbation(args)

        num_sensor = num_agent_list[0][0]
        
        ego_idx = args.ego_agent
        all_agent_list = [i for i in range(num_sensor)]
        all_agent_list.remove(ego_idx)
        attacker_list = random.sample(all_agent_list, k=args.number_of_attackers)
        data['attacker_list'] = attacker_list
        data['eps'] = args.eps
        data['no_fuse'] = False

        pert = run_pgd_attack(data, pseudo_gt, args, fafmodule, device, pert)
        data['pert'] = pert.to(device)

        
        data['pert'] = None
        data['collab_agent_list'] = None
        data['no_fuse'] = True
        _, _, _, result_reference = fafmodule.predict_all(data, 1, num_agent=num_agent)

        agent_feats = extract_agent_features(num_all_agents, padded_voxel_points, model, device)
        features_all = torch.stack(agent_feats, dim=0)  # shape [6, 512, 16, 16]

        pert = pert.to(device)
        for att_id in attacker_list:
            features_all[att_id] += pert[att_id]
        
        N = num_all_agents[0][0].item()
        feature_batch = features_all  # shape [N, 256, 32, 32]
        labels_batch = torch.zeros(N, device=device)
        for att_id in attacker_list:
            labels_batch[att_id] = 1.0

        
        t_det_start = time.perf_counter()
        
        with torch.no_grad():
            logits = discriminator(features_all).view(-1)          # [N]
            probs = torch.softmax(logits, dim=0)
            pred_attacker_idx = torch.argmax(probs).item()

            t_det_end = time.perf_counter()
            detection_times.append(t_det_end - t_det_start)

            pred_attackers = [pred_attacker_idx]

            true_label = torch.tensor(attacker_list[0], dtype=torch.long, device=device)
            loss_val = criterion_disc(logits.unsqueeze(0), true_label.unsqueeze(0))
            val_loss += loss_val.item()

            if pred_attacker_idx in attacker_list:  # If predicted attacker is actually an attacker
                val_correct += 1
            val_total += 1



            consensus_set_size = cal_robosac_consensus(num_agent, args.step_budget, args.number_of_attackers)

            print(f"consensus_set_size = {consensus_set_size}")



            found = False
            # NOTE: 0~step_budget-1
            # Step 1: Create the possible benign list
            predicted_attacker = pred_attackers[0]
            possible_benign_agents = [agent for agent in all_agent_list if agent != predicted_attacker]

            # Debugging info (optional)
            print(f"Possible benign agents (excluding predicted attacker {predicted_attacker}): {possible_benign_agents}")

            for step in range(1, args.step_budget + 1):
                # NOTE: random.choices will sample an agent more than once. eg.: [2, 3, 2]
                # So we should use random.sample(population, k) to avoid this.
                # collab_agent_list = random.sample(all_agent_list, k=args.robosac_k)

                collab_agent_list = random.sample(possible_benign_agents, k=consensus_set_size)
                # collab_agent_list = random.sample(possible_benign_agents, k=consensus_set_size)

                data['collab_agent_list'] = collab_agent_list
                data['no_fuse'] = False
                data['pert'] = pert.to(device)

                loss, cls_loss, loc_loss, result = fafmodule.predict_all(data, 1, num_agent=num_agent)

                # We use jaccard index to define the difference between two bbox sets
                jac_index = get_jaccard_index(args, config, num_agent_list, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result_reference, result)
                print("Jaccard Coefficient: {}".format(jac_index))
                if jac_index < args.box_matching_thresh:
                    # print('Attacker(s) is(are) among {}'.format(collab_agent_list))
                    continue
                else:
                    sus_agent_list = [i for i in all_agent_list if i not in collab_agent_list]
                    print('Achieved consensus at step {}, with agents {}.'.format(step, collab_agent_list, sus_agent_list))
                    found = True
                    
                    
                    steps[frame_seq-1] = step
                    succ += 1
                    if args.visualization:
                        # visualize consensus result
                        visualize(args, config, filename0, save_fig_path, fafmodule, data, num_agent_list, padded_voxel_points, gt_max_iou, vis_tag='consensus')
                    break

            if not found:
                print('No consensus!')
                # Can't achieve consensus, so fall back to original ego only result
                data['pert'] = None
                data['collab_agent_list'] = None
                data['no_fuse'] = True
                _, _, _, result_self_only = fafmodule.predict_all(data, 1, num_agent=num_agent)
                result = result_self_only
                steps[frame_seq-1] = args.step_budget
                ego_steps[frame_seq-1] = 1
                fail += 1  

            ego_steps[frame_seq - 1] = 1  

            det_results_local, annotations_local = local_eval(num_agent, padded_voxel_points, reg_target, anchors_map, gt_max_iou, result, config, det_results_local, annotations_local)

        if (cnt == 3):
            break
    avg_val_loss = val_loss / (cnt+1)
    accuracy     = val_correct / val_total if val_total > 0 else 0.0
    print(f"Test loss = {avg_val_loss:.4f}, val acc = {accuracy*100:.2f}%")

    print("\n Ego Agent:{}".format(args.ego_agent))

    print("Jeeb-Net VALIDATION: Evaluated on {} frames".format(frame_seq))
    print("Total Neighbor Agents:{}, Sampling Set Size: {}, Number of Attackers: {}".format(num_agent-1, args.robosac_k, args.number_of_attackers))
    if args.robosac_k is None:
        consensus_set_size = cal_robosac_consensus(num_agent, args.step_budget, args.number_of_attackers)
        print("Expected guaranteed Consensus Set Size at p=0.99: {}".format(consensus_set_size))
    print("Succeeded {}, Total {}, Success Rate: {}".format(succ, frame_seq, succ / frame_seq))
    print("Sampling STEP MEAN: {}, MAX: {}, MIN:{}".format(np.mean(steps), np.max(steps), np.min(steps)))
    total_steps = steps + ego_steps
    print("Total STEP(including ego only step): MEAN: {}, MAX: {}, MIN:{}".format(np.mean(total_steps), np.max(total_steps), np.min(total_steps)))
    fpss = 1000 / (27*steps+17*ego_steps) # forward time: ego only: 17ms; collaborated: 27ms
    print("FPS: MEAN: {}, MAX: {}, MIN:{}".format(np.mean(fpss), np.max(fpss), np.min(fpss)))
    print("Sampling STEP:{}, Ego STEP:{}, Total STEP:{}, FPS:{}".format(steps, ego_steps, total_steps, fpss))
    print("Box set matching threshold: {}".format(args.box_matching_thresh))

    eval_start_idx = 0

    mean_ap_local = []
    # local mAP evaluation
    det_results_all_local = []
    annotations_all_local = []
    for k in range(eval_start_idx, num_agent):
        print("Local mAP@0.5 from agent {}".format(k))
        mean_ap, _ = eval_map(det_results_local[k], annotations_local[k], scale_ranges=None, iou_thr=0.5, dataset=None, logger=None)
        mean_ap_local.append(mean_ap)
        print("Local mAP@0.7 from agent {}".format(k))

        mean_ap, _ = eval_map(det_results_local[k], annotations_local[k], scale_ranges=None, iou_thr=0.7, dataset=None, logger=None)
        mean_ap_local.append(mean_ap)

        det_results_all_local += det_results_local[k]
        annotations_all_local += annotations_local[k]

    # average local mAP evaluation
    print("Average Local mAP@0.5")

    mean_ap_local_average, _ = eval_map(det_results_all_local, annotations_all_local, scale_ranges=None, iou_thr=0.5, dataset=None, logger=None,)
    mean_ap_local.append(mean_ap_local_average)

    print("Average Local mAP@0.7")

    mean_ap_local_average, _ = eval_map(det_results_all_local, annotations_all_local, scale_ranges=None, iou_thr=0.7, dataset=None, logger=None,)
    mean_ap_local.append(mean_ap_local_average)

    print("Quantitative evaluation results of model from {}, at epoch {}".format(args.resume, start_epoch - 1))

    for k in range(eval_start_idx, num_agent):
        print("agent{} mAP@0.5 is {} and mAP@0.7 is {}".format(k, mean_ap_local[k * 2], mean_ap_local[(k * 2) + 1]))

    print("average local mAP@0.5 is {} and average local mAP@0.7 is {}".format(mean_ap_local[-2], mean_ap_local[-1]))


    avg_detection_time = sum(detection_times) / len(detection_times) if detection_times else 0.0

    print(f"\nAverage attacker-detection time  : {avg_detection_time*1e3:.2f} ms")






if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("-d", "--data", default="{Your_location_to_V2X-Sim}/V2X-Sim/test", type=str, help="The path to the preprocessed sparse BEV training data")
    parser.add_argument("--train_data",default="../../../V2X-Sim-det/train", type=str, help="The path to the preprocessed sparse BEV training data")
    parser.add_argument("--test_data", default="../../../V2X-Sim-det/test", type=str, help="The path to the preprocessed sparse BEV test data")
    
    parser.add_argument("--batch", default=1, type=int, help="The number of scene")
    parser.add_argument("--nworker", default=2, type=int, help="Number of workers")
    parser.add_argument("--lr", default=0.0001, type=float, help="Initial learning rate")
    # parser.add_argument("--log", action="store_true", help="Whether to log")
    # # parser.add_argument("--logpath", default="", help="The path to the output log file")
    parser.add_argument("--resume", default = "../../ckpt/meanfusion/epoch_advtrain_49.pth", type=str, help="The path to the saved model that is loaded to resume training")
    # parser.add_argument("--resume_teacher", default="", type=str, help="The path to the saved teacher model that is loaded to resume training")
    parser.add_argument("--layer", default=3, type=int, help="Communicate which layer in the single layer com mode")
    # parser.add_argument("--warp_flag", action="store_true", help="Whether to use pose info for When2com")
    parser.add_argument("--kd_flag", default=0, type=int, help="Whether to enable distillation (only DiscNet is 1 )")
    # parser.add_argument("--kd_weight", default=100000, type=int, help="KD loss weight")
    # parser.add_argument("--gnn_iter_times", default=3, type=int, help="Number of message passing for V2VNet")
    parser.add_argument("--visualization", action="store_true", help="Visualize validation result")
    parser.add_argument("--com", default="mean", type=str, help="disco/when2com/v2v/sum/mean/max/cat/agent")
    parser.add_argument("--bound", type=str,default="both",help="The input setting: lowerbound -> single-view or upperbound -> multi-view")
    parser.add_argument("--inference", type=str)
    # parser.add_argument("--tracking", action="store_true")
    # parser.add_argument("--box_com", action="store_true")
    parser.add_argument("--no_cross_road", action="store_true", help="Do not load data of cross roads")
    # # scene_batch => batch size in each scene
    parser.add_argument("--num_agent", default=6, type=int, help="The total number of agents")
    # parser.add_argument("--apply_late_fusion",default=0,type=int,help="1: apply late fusion. 0: no late fusion")
    parser.add_argument("--compress_level",default=0,type=int, help="Compress the communication layer channels by 2**x times in encoder",)
    # parser.add_argument("--pose_noise",default=0, type=float, help="draw noise from normal distribution with given mean (in meters), apply to transformation matrix.",)
    parser.add_argument("--only_v2i",default=0,type=int,help="1: only v2i, 0: v2v and v2i",)

    # # Adversarial perturbation
    parser.add_argument('--pert_alpha', type=float, default=0.1, help='scale of the perturbation')
    parser.add_argument('--adv_method', type=str, default='pgd', help='pgd/bim/cw-l2')
    parser.add_argument('--eps', type=float, default=0.5, help='epsilon of adv attack.')
    parser.add_argument('--adv_iter', type=int, default=15, help='adv iterations of computing perturbation')

    # # Scene and frame settings
    parser.add_argument('--scene_id',nargs='+', type=int, default=[20, 33, 34, 35, 36, 37, 41, 44, 48, 49, 50, 51, 58, 64, 72, 85, 88, 8, 96, 97], help='which scene IDs to run over')

    # parser.add_argument('--sample_id', type=int, default=None, help='target evaluation sample')

    # # Among Us modes and parameters
    # parser.add_argument('--robosac', type=str, default='', help='upperbound/lowerbound/no_defense/robosac_validation/robosac_mAP/adaptive/fix_attackers/performance_eval/probing')
    parser.add_argument('--ego_agent', type=int, default=1, help='id of ego agent')
    parser.add_argument('--robosac_k', type=int, default=None, help='specify consensus set size if needed')
    parser.add_argument('--ego_loss_only', action="store_true", help='only use ego loss to compute adv perturbation')
    parser.add_argument('--step_budget', type=int, default=3, help='sampling budget in a single frame')
    parser.add_argument('--box_matching_thresh', type=float, default=0.3, help='IoU threshold for validating two detection results')
    parser.add_argument('--number_of_attackers', type=int, default=1, help='number of malicious attackers in the scene')
    # parser.add_argument('--fix_attackers', action="store_true", help='if true, attackers will not change in different frames')
    # parser.add_argument('--use_history_frame', action="store_true", help='use history frame for computing the consensus, reduce 1 step of forward prop.')
    # parser.add_argument('--partial_upperbound', action="store_true", help='use with specifying ransan_k, to perform clean collaboration with a subset of teammates')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs for training')
    args = parser.parse_args()
    main(args)