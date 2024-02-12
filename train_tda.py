import argparse
import os
# import torch_tda
# import bats
from torch_topological.nn import VietorisRipsComplex
from torch_topological.nn import WassersteinDistance

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
parser.add_argument("--model", type=str, default='punet')
parser.add_argument('--log_dir', default='logs/test', help='Log dir [default: logs/test_log]')
parser.add_argument('--npoint', type=int, default=1024,help='Point Number [1024/2048] [default: 1024]')
parser.add_argument('--up_ratio',  type=int,  default=4, help='Upsampling Ratio [default: 4]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epochs to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training')
parser.add_argument("--use_bn", action='store_true', default=False)
parser.add_argument("--use_res", action='store_true', default=False)
parser.add_argument("--alpha", type=float, default=1.0) # for repulsion loss
parser.add_argument("--beta", type=float, default=1.0) # for tda loss
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--use_decay', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay', type=float, default=0.71)
parser.add_argument('--lr_clip', type=float, default=0.000001)
parser.add_argument('--decay_step_list', type=list, default=[30, 60])
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--workers', type=int, default=4)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pointnet2 import pointnet2_utils as pn2_utils
from utils.utils import knn_point
from chamfer_distance import chamfer_distance
from auction_match import auction_match

from dataset import PUNET_Dataset
import numpy as np
import importlib


class UpsampleLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, n=16, dim=0, q=2, voxelize=False, nn_size=5, radius=0.07, h=0.03, eps=1e-12):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.n = n
        self.dim = dim
        self.q = q
        self.voxelize = voxelize

        self.nn_size = nn_size
        self.radius = radius
        self.h = h
        self.eps = eps

    def get_emd_loss(self, pred, gt, pcd_radius):
        idx, _ = auction_match(pred, gt)
        matched_out = pn2_utils.gather_operation(gt.transpose(1, 2).contiguous(), idx)
        matched_out = matched_out.transpose(1, 2).contiguous()
        dist2 = (pred - matched_out) ** 2
        dist2 = dist2.view(dist2.shape[0], -1) # <-- ???
        dist2 = torch.mean(dist2, dim=1, keepdims=True) # B,
        dist2 /= pcd_radius

        return torch.mean(dist2)

    def get_cd_loss(self, pred, gt, pcd_radius):
        cost_for, cost_bac = chamfer_distance(gt, pred)
        cost = 0.8 * cost_for + 0.2 * cost_bac
        cost /= pcd_radius
        cost = torch.mean(cost)

        return cost

    def get_repulsion_loss(self, pred):
        _, idx = knn_point(self.nn_size, pred, pred, transpose_mode=True)
        idx = idx[:, :, 1:].to(torch.int32) # remove first one
        idx = idx.contiguous() # B, N, nn

        pred = pred.transpose(1, 2).contiguous() # B, 3, N
        grouped_points = pn2_utils.grouping_operation(pred, idx) # (B, 3, N), (B, N, nn) => (B, 3, N, nn)

        grouped_points = grouped_points - pred.unsqueeze(-1)
        dist2 = torch.sum(grouped_points ** 2, dim=1)
        dist2 = torch.max(dist2, torch.tensor(self.eps).cuda())
        dist = torch.sqrt(dist2)
        weight = torch.exp(- dist2 / self.h ** 2)

        uniform_loss = torch.mean((self.radius - dist) * weight)

        return uniform_loss

    def get_voxel_loss(self, pred, gt):
        VR = VietorisRipsComplex(dim=self.dim)
        WD = WassersteinDistance(q=self.q)
        if pred.shape[0] != 0 and gt.shape[0] != 0:
            if pred.shape[0] > self.n:
                shuffled_indices_pred = torch.randperm(pred.size(0))
                sampled_indices_pred = shuffled_indices_pred[:self.n]
                pred_sampled = pred[sampled_indices_pred]
            else:
                pred_sampled = pred_voxel
            if gt_voxel.shape[0] > self.nn:
                shuffled_indices_gt = torch.randperm(gt.size(0))
                sampled_indices_gt = shuffled_indices_gt[:self.n]
                gtd = gt_voxel[sampled_indices_gt]
            else:
                gtd = gt_voxel

            dgm_pred = VR(pred_sampled)
            dgm_gt = VR(gtd)

            voxel_loss = WD(dgm_pred, dgm_gt)
            
            return voxel_loss


    def get_tda_loss(self, pred, gt):
        tda_loss = 0
        for (pred_sample, gt_sample) in list(zip(pred, gt)):
            if self.voxelize:
                sample_loss = 0

                combined_sample = torch.cat([gt_sample], dim=0)
                x_boundaries = torch.linspace(combined_sample[:, 0].min().item(), combined_sample[:, 0].max().item(), 3)
                y_boundaries = torch.linspace(combined_sample[:, 1].min().item(), combined_sample[:, 1].max().item(), 3)
                z_boundaries = torch.linspace(combined_sample[:, 2].min().item(), combined_sample[:, 2].max().item(), 3)
    
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            x_min, x_max = x_boundaries[i], x_boundaries[i + 1]
                            y_min, y_max = y_boundaries[j], y_boundaries[j + 1]
                            z_min, z_max = z_boundaries[k], z_boundaries[k + 1]

                            pred_voxel = pred_sample[
                                (pred_sample[:, 0] >= x_min) & (pred_sample[:, 0] < x_max) &
                                (pred_sample[:, 1] >= y_min) & (pred_sample[:, 1] < y_max) & 
                                (pred_sample[:, 2] >= z_min) & (pred_sample[:, 2] < z_max)
                            ]

                            gt_voxel = gt_sample[
                                (gt_sample[:, 0] >= x_min) & (gt_sample[:, 0] < x_max) &
                                (gt_sample[:, 1] >= y_min) & (gt_sample[:, 1] < y_max) & 
                                (gt_sample[:, 2] >= z_min) & (gt_sample[:, 2] < z_max)
                            ]

                            sample_loss += self.get_voxel_loss(pred_voxel, gt_voxel)
            else:
                sample_loss = self.get_voxel_loss(pred, gt) 
            print("Sample Loss: ", sample_loss)
            tda_loss += sample_loss
        
        average_batch_loss = tda_loss / pred.shape[0]

        return average_batch_loss

    def forward(self, pred, gt, pcd_radius):
        emd_loss = self.get_emd_loss(pred, gt, pcd_radius)
        repulsion_loss = self.get_repulsion_loss(pred)
        tda_loss = self.get_tda_loss(pred, gt)

        return emd_loss * 100, repulsion_loss * self.alpha, tda_loss * self.beta

def get_optimizer():
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr, 
                                momentum=0.98, 
                                weight_decay=args.weight_decay, 
                                nesterov=True)
    else:
        raise NotImplementedError
    
    if args.use_decay:
        def lr_lbmd(cur_epoch):
            cur_decay = 1
            for decay_step in args.decay_step_list:
                if cur_epoch >= decay_step:
                    cur_decay = cur_decay * args.lr_decay
            return max(cur_decay, args.lr_clip / args.lr)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lbmd)
        return optimizer, lr_scheduler
    else:
        return optimizer, None


if __name__ == '__main__':
    train_dst = PUNET_Dataset(h5_file_path='/usr/xtmp/wgk4/PU-Net_pytorch_datas/Patches_noHole_and_collected.h5', npoint=args.npoint, 
            use_random=True, use_norm=True, split='train', is_training=True)
    train_loader = DataLoader(train_dst, batch_size=args.batch_size, 
                        shuffle=True, pin_memory=True, num_workers=args.workers)

    MODEL = importlib.import_module('models.' + args.model)
    model = MODEL.get_model(npoint=args.npoint, up_ratio=args.up_ratio, 
                use_normal=False, use_bn=args.use_bn, use_res=args.use_res)
    model.cuda()
    
    optimizer, lr_scheduler = get_optimizer()
    loss_func = UpsampleLoss(alpha=args.alpha, beta=args.beta, n=16, dim=0, q=2, voxelize=True)

    model.train()
    for epoch in range(args.max_epoch):
        loss_list = []
        emd_loss_list = []
        rep_loss_list = []
        tda_loss_list = []
        for batch in train_loader:
            optimizer.zero_grad()
            input_data, gt_data, radius_data = batch

            input_data = input_data.float().cuda()
            gt_data = gt_data.float().cuda()
            gt_data = gt_data[..., :3].contiguous()
            radius_data = radius_data.float().cuda()

            preds = model(input_data)
            emd_loss, rep_loss, tda_loss = loss_func(preds, gt_data, radius_data)
            loss = emd_loss + rep_loss + tda_loss

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            emd_loss_list.append(emd_loss.item())
            rep_loss_list.append(rep_loss.item())
            tda_loss_list.append(tda_loss.item())

        print(' -- epoch {}, loss {:.4f}, weighted emd loss {:.4f}, repulsion loss {:.4f}, tda loss {:.4f} lr {}.'.format(
            epoch, np.mean(loss_list), np.mean(emd_loss_list), np.mean(rep_loss_list), np.mean(tda_loss_list), \
            optimizer.state_dict()['param_groups'][0]['lr']))
        
        if lr_scheduler is not None:
            lr_scheduler.step(epoch)
        if (epoch + 1) % 1 == 0:
            state = {'epoch': epoch, 'model_state': model.state_dict()}
            save_path = os.path.join(args.log_dir, 'punet_epoch_{}.pth'.format(epoch))
            torch.save(state, save_path)