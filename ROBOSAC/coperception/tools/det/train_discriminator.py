import sys
import argparse
import torch
import random

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from torch.optim.lr_scheduler import CosineAnnealingLR


from tqdm import tqdm
from utils import *
from coperception.utils.CoDetModule import FaFModule
import matplotlib.pyplot as plt


def main(args):
    config, config_global, flag = setup_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)


    train_dataset, val_dataset, agent_idx_range, num_agent = build_dataset(args, config, config_global)
    print(f"Train/Val sizes: {len(train_dataset)}/{len(val_dataset)}")
    train_loader, val_loader = build_loaders(args, train_dataset, val_dataset)

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
        v.requires_grad = False

    frame_seq = 0


    discriminator, optimizer_disc, criterion_disc = init_discriminator_training(device, args.lr)
    scheduler_disc = CosineAnnealingLR(optimizer_disc, T_max=args.epochs, eta_min=1e-6)

    discriminator.train()

    train_losses = []
    val_losses   = []
    val_accs     = []


    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        count = 0
        print(epoch)
        for cnt, sample in enumerate(tqdm(train_loader)):

            unpacked = unpack_and_filter_sample(args, sample, device)
            if unpacked is None:
                continue

            num_agent_list ,num_all_agents, padded_voxel_points, data, reg_target, anchors_map, gt_max_iou, filenames0 = unpacked
            pseudo_gt = get_pseudo_gt(data, fafmodule, args.batch)
            pert = init_perturbation(args)
            num_sensor = num_agent_list[0][0] # num_sensor = 6

            ego_idx = args.ego_agent # ego_idx = 1
            all_agent_list = [i for i in range(num_sensor)] #[0, 1, 2, 3, 4, 5]
            all_agent_list.remove(ego_idx) # [0, 2, 3, 4, 5]
            attacker_list = random.sample(all_agent_list, k=args.number_of_attackers) #randomly sample number_of_attackers from [0, 2, 3, 4, 5]
            data['attacker_list'] = attacker_list
            data['eps'] = args.eps
            data['no_fuse'] = False

            pert = run_pgd_attack(data, pseudo_gt, args, fafmodule, device, pert)
            data['pert'] = pert.to(device)
            with torch.no_grad():
                agent_feats = extract_agent_features(num_all_agents, padded_voxel_points, model, device)

            features_all = torch.stack(agent_feats, dim=0)  # shape [6, 512, 16, 16]
            
            pert = pert.to(device)
            for att_id in attacker_list:
                features_all[att_id] += pert[att_id]
            
            N = num_all_agents[0][0].item()

            feature_batch = features_all  # shape [N, 256, 32, 32]

            labels_batch = torch.tensor(attacker_list[0], dtype=torch.long, device=device)

            logits = discriminator(feature_batch)  # shape [N, 1]
            logits = logits.view(-1)  # flatten to shape [N]
            loss = criterion_disc(logits.unsqueeze(0), labels_batch.unsqueeze(0)) 

            optimizer_disc.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)

            optimizer_disc.step()
            total_loss += loss.item()
            count += 1
        avg_loss = total_loss / (count if count > 0 else 1)
        train_losses.append(total_loss)
        
        print(f"Epoch {epoch}/{args.epochs} - Avg Train loss: {avg_loss:.4f}")

        discriminator.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        all_preds = []
        all_trues = []

        for cnt, sample in enumerate(tqdm(val_loader)):

            t = time.time()

            unpacked = unpack_and_filter_sample(args, sample, device)
            if unpacked is None:
                continue
            num_agent_list ,num_all_agents, padded_voxel_points, data, reg_target, anchors_map, gt_max_iou, filenames0 = unpacked        

            pseudo_gt = get_pseudo_gt(data, fafmodule, args.batch)
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
            

            with torch.no_grad():
                logits = discriminator(features_all).view(-1)          # [N]
                probs = torch.softmax(logits, dim=0)
                pred_attacker_idx = torch.argmax(probs).item()

                all_preds.append(pred_attacker_idx)
                all_trues.append(attacker_list[0])

                pred_attackers = [pred_attacker_idx]

                loss_val = criterion_disc(logits, labels_batch)
                val_loss += loss_val.item()

                true_label = torch.tensor(attacker_list[0], dtype=torch.long, device=device)
                loss_val = criterion_disc(logits.unsqueeze(0), true_label.unsqueeze(0))
                val_loss += loss_val.item()

                if pred_attacker_idx in attacker_list:  # If predicted attacker is actually an attacker
                    val_correct += 1
                val_total += 1

            
        avg_val_loss = val_loss / (cnt+1)
        accuracy     = val_correct / val_total if val_total > 0 else 0.0
        # print(f"Epoch {epoch}: validation loss = {avg_val_loss:.4f}")
        print(f"Epoch {epoch}: Avg val loss = {avg_val_loss:.4f}, val acc = {accuracy*100:.2f}%")

        cm     = confusion_matrix(all_trues, all_preds)
        report = classification_report(all_trues, all_preds, digits=4)

        np.savetxt('confusion_matrix.csv',cm, delimiter=',', fmt='%d')
        with open('classification_report.txt', 'w') as f:
            f.write(report)

        val_losses.append(val_loss)
        val_accs.append(accuracy)


        scheduler_disc.step()
        print(f" LR after epoch {epoch}: {scheduler_disc.get_last_lr()}\n")

    ckpt_path = "discriminator_checkpoint.pth"
    torch.save({"epoch": epoch,
                "model_state_dict": discriminator.state_dict(),
                "optimizer_state_dict": optimizer_disc.state_dict(),
                "scheduler_state_dict": scheduler_disc.state_dict(),
                }, ckpt_path)
    
    epochs = range(1, args.epochs + 1)

    plt.figure()
    plt.semilogy(epochs, train_losses, marker='o', label='Train')
    plt.semilogy(epochs, val_losses,   marker='o', label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss_log_curve.png')


    df = pd.DataFrame({'epoch':      epochs,
                       'train_loss': train_losses,
                       'val_loss':   val_losses,
                       'val_acc':    val_accs,})
    df.to_csv( 'metrics.csv', index=False)



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
    # parser.add_argument("--visualization", action="store_true", help="Visualize validation result")
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
    # parser.add_argument('--robosac_k', type=int, default=None, help='specify consensus set size if needed')
    parser.add_argument('--ego_loss_only', action="store_true", help='only use ego loss to compute adv perturbation')
    # parser.add_argument('--step_budget', type=int, default=3, help='sampling budget in a single frame')
    # parser.add_argument('--box_matching_thresh', type=float, default=0.3, help='IoU threshold for validating two detection results')
    parser.add_argument('--number_of_attackers', type=int, default=1, help='number of malicious attackers in the scene')
    # parser.add_argument('--fix_attackers', action="store_true", help='if true, attackers will not change in different frames')
    # parser.add_argument('--use_history_frame', action="store_true", help='use history frame for computing the consensus, reduce 1 step of forward prop.')
    # parser.add_argument('--partial_upperbound', action="store_true", help='use with specifying ransan_k, to perform clean collaboration with a subset of teammates')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs for training')
    args = parser.parse_args()
    main(args)