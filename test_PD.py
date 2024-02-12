import torch_tda
import bats
import torch
import time
from torch.autograd import gradcheck

def get_dgm(pts):
    flags = (bats.standard_reduction_flag(),bats.compression_flag())
    PD = torch_tda.nn.RipsLayer(maxdim=2, reduction_flags=flags) 
    dgm = PD(pts)
    return dgm

if __name__ == "__main__":
    gt = torch.rand(1000, 3).requires_grad_()
    pred = torch.rand(1000, 3).requires_grad_()

    combined = torch.cat([gt, pred], dim=0)
    x_boundaries = torch.linspace(combined[:, 0].min().item(), combined[:, 0].max().item(), 11)
    y_boundaries = torch.linspace(combined[:, 1].min().item(), combined[:, 1].max().item(), 11)
    z_boundaries = torch.linspace(combined[:, 2].min().item(), combined[:, 2].max().item(), 11)

    loss = 0
    for i in range(10):
        for j in range(10):
            for k in range(10):
                x_min, x_max = x_boundaries[i], x_boundaries[i + 1]
                y_min, y_max = y_boundaries[j], y_boundaries[j + 1]
                z_min, z_max = z_boundaries[k], z_boundaries[k + 1]

                pred_voxel = pred[
                    (pred[:, 0] >= x_min) & (pred[:, 0] < x_max) &
                    (pred[:, 1] >= y_min) & (pred[:, 1] < y_max) & 
                    (pred[:, 2] >= z_min) & (pred[:, 2] < z_max)
                ]

                gt_voxel = gt[
                    (gt[:, 0] >= x_min) & (gt[:, 0] < x_max) &
                    (gt[:, 1] >= y_min) & (gt[:, 1] < y_max) & 
                    (gt[:, 2] >= z_min) & (gt[:, 2] < z_max)
                ]

                if pred_voxel.shape[0] != 0 and gt_voxel.shape[0] != 0:
                    shuffled_indices = torch.randperm(pred_voxel.shape[0])
                    sampled_indices = shuffled_indices[:100]

                    # if pred_voxel.shape[0] > 100:
                    #     # Get the 100 random points
                    #     pred_voxel_sampled = pred_voxel[sampled_indices]
                    # else:
                    #     pred_voxel_sampled = pred_voxel
                    # if gt_voxel.shape[0] > 100:
                    #     # Get the 100 random points
                    #     gt_voxel_sampled = gt_voxel[sampled_indices]
                    # else:
                    #     gt_voxel_sampled = gt_voxel
    
                    # dgm_gt = get_dgm(pred_voxel_sampled)
                    # dgm_pred = get_dgm(gt_voxel_sampled)

                    dgm_gt = get_dgm(gt_voxel)
                    dgm_pred = get_dgm(pred_voxel)

                    # print('gt:', dgm_gt)
                    # print('pred:', dgm_pred)

                    WD = torch_tda.nn.WassersteinLayer()
                    loss += WD(dgm_gt[0], dgm_pred[0])
                    # loss += WD(dgm_gt[1], dgm_pred[1])
                    # loss += WD(dgm_gt[2], dgm_pred[2])
    print("Backward")
    loss.backward()

    print(loss)