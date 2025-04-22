#!/usr/bin/env python3
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# reuse the same dataset & config from robosac.py
from coperception.datasets import V2XSimDet
from coperception.configs import Config, ConfigGlobal
from coperception.utils.CoDetModule import FaFModule
from robosac import setup_seed

class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def extract_features(fafmodule, data, device):
    # get per‐agent class logits: [num_agent, num_anchors, 2]
    cls_logits = fafmodule.cls_predict(data, batch_size=1, no_fuse=True)
    # collapse anchors → mean over all anchors per class
    # result: [num_agent, 2]
    feat = torch.mean(cls_logits, dim=1)
    return feat.detach().cpu()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='path to V2X‐Sim test data')
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--out', default='disc.pth')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the existing detection module just to extract features
    config = Config("train", binary=True, only_det=True)
    config_global = ConfigGlobal("train", binary=True, only_det=True)
    # we only need cls_predict → use any fusion mode, it won't matter
    fafmodule = FaFModule  # imported class

    # dataset + loader
    val_ds = V2XSimDet(
        dataset_roots=[f"{args.data}/agent{i}" for i in range(1,6)],
        config=config, config_global=config_global,
        split="val", val=True
    )
    loader = DataLoader(val_ds, batch_size=args.batch, shuffle=True, num_workers=4)

    # instantiate discriminator: input_dim=2 (two class logits)
    disc = Discriminator(in_dim=2).to(device)
    optim_disc = optim.Adam(disc.parameters(), lr=args.lr)
    criterion  = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        disc.train()
        total_loss = 0
        for batch in loader:
            # batch is a tuple, we only need to reconstruct the `data` dict
            (_bev, _bev_t, labels, reg_t, anchors, vis, gt_iou,
             fnames, tgt_ids, num_agents, trans) = batch[0]

            # prepare data dict exactly as in robosac.py
            data = {
                "bev_seq": _bev.to(device),
                "bev_seq_teacher": _bev_t.to(device),
                "reg_targets": reg_t.to(device),
                "anchors": anchors.to(device),
                "vis_maps": vis.to(device),
                "reg_loss_mask": None,
                "target_agent_ids": tgt_ids.to(device),
                "num_agent": num_agents.to(device),
                "ego_agent": 1,
                "pert": None,
                "no_fuse": True,
                "collab_agent_list": None,
                "trial_agent_id": None,
                "confidence": None,
                "unadv_pert": None,
                "attacker_list": None,
                "eps": None,
                "trans_matrices": trans.to(device),
            }

            # STEP A: randomly pick attackers just like in robosac
            num_sensor = num_agents[0][0].item()
            all_agents = list(range(num_sensor))
            all_agents.remove(args.ego_agent if hasattr(args, 'ego_agent') else 1)
            attacker_list = random.sample(all_agents, k=1)

            # STEP B: generate adv pert and apply (PGD one‐step for simplicity)
            pert = torch.randn_like(data["bev_seq"]) * 0.1
            data["pert"] = pert.to(device)

            # get features & labels
            faf = fafmodule( # instantiate module
                torch.nn.DataParallel, torch.nn.DataParallel, config, optim_disc, {"cls":None,"loc":None}, 0
            )
            feats = extract_features(faf, data, device)  # [num_agent, 2]

            # build training batch
            labels = torch.zeros(num_sensor)
            labels[attacker_list] = 1.0
            labels = labels.to(device)

            # forward / backward
            optim_disc.zero_grad()
            logits = disc(feats.to(device))  # [num_agent]
            loss = criterion(logits, labels)
            loss.backward()
            optim_disc.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}/{args.epochs}] loss={total_loss/len(loader):.4f}")

    torch.save(disc.state_dict(), args.out)
    print(f"Discriminator saved to {args.out}")

if __name__ == "__main__":
    main()
