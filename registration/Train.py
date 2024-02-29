import os
import glob
import sys
from argparse import ArgumentParser
import scipy.ndimage
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from TS_SAR_model import TS_SAR_stage, AffineCOMTransform, Center_of_mass_initial_pairwise, multi_resolution_NCC
from Functions import Dataset_epoch_MNI152
from segmentation.models import ResUNet

def dice(im1, atlas):
    unique_class = np.unique(atlas)
    dice = 0
    num_count = 0
    for i in unique_class:
        if (i == 0) or ((im1 == i).sum() == 0) or ((atlas == i).sum() == 0):
            continue

        sub_dice = np.sum(atlas[im1 == i] == i) * 2.0 / (np.sum(im1 == i) + np.sum(atlas == i))
        dice += sub_dice
        num_count += 1
    return dice / num_count

def per_loss(feature_module, loss_func, y, y_):
    out = feature_module(y)
    out_ = feature_module(y_)
    loss = loss_func(out, out_)
    return loss

def get_feature_module(layer_index,device=None):
        per = ResUNet.cuda()
        per.eval()
        for parm in per.parameters():
            parm.requires_grad = False

        feature_module = per[0:layer_index + 1]
        feature_module.cuda()
        return feature_module

class PerceptualLoss(nn.Module):
    def __init__(self, loss_func, layer_indexs=None, device=None):
        super(PerceptualLoss, self).__init__()
        self.creation = loss_func
        self.layer_indexs = layer_indexs
        self.device = device

    def forward(self, y, y_):
        loss = 0
        for index in self.layer_indexs:
            feature_module = get_feature_module(index, self.device)
            loss += per_loss(feature_module, self.creation, y, y_)
        return loss


def train():
    model = TS_SAR_stage(img_size=128, patch_size=[3, 7, 15], stride=[2, 4, 8], num_classes=6,
                          embed_dims=[256, 256, 256],
                          num_heads=[2, 2, 2], mlp_ratios=[2, 2, 2], qkv_bias=False, qk_scale=None, drop_rate=0.,
                          attn_drop_rate=0., norm_layer=nn.Identity,
                          depths=[4, 4, 4], sr_ratios=[1, 1, 1], num_stages=3, linear=False).cuda()

    affine_transform = AffineCOMTransform().cuda()
    init_center = Center_of_mass_initial_pairwise()

    loss_similarity = multi_resolution_NCC(win=7, scale=3)


    imgs = sorted(glob.glob(datapath + "/*"))# low-resolution move feature
    labels = sorted(glob.glob(labelpath + "/*"))# high-resolution move feature
    fix_img = sorted(glob.glob(fixpath + "/*"))# low-resolution fix feature
    fix_label = sorted(glob.glob(fixlabelpath + "/*"))# high-resolution fix feature
    dof_path = sorted(glob.glob(dofpath + "/*"))


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model_dir = '../Model/' + model_name[0:-1]

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    lossall = np.zeros((4, iteration + 1))

    training_generator = Data.DataLoader(Dataset_epoch_MNI152(imgs, labels, fix_img, fix_label, dof_path, need_label=True),
                                         batch_size=1,
                                         shuffle=False, num_workers=0)
    step = 0
    load_model = False
    if load_model is True:
        model_path = "../Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
        print("Loading weight: ", model_path)
        step = 3000
        model.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
        lossall[:, 0:3000] = temp_lossall[:, 0:3000]

    while step <= iteration:
        for X, Y, X_label, Y_label, dof in training_generator:

            X = X.cuda().float()
            Y = Y.cuda().float()
            X_label = X_label.cuda().float()
            Y_label = Y_label.cuda().float()
            dof = dof.squeeze(0).to('cuda:0')
            # COM initialization
            if com_initial:
                X, _ = init_center(X, Y)

            X = F.interpolate(X, scale_factor=0.5, mode="trilinear", align_corners=True)
            Y = F.interpolate(Y, scale_factor=0.5, mode="trilinear", align_corners=True)
            warpped_x_list, y_list, affine_para_list, affm = model(X, Y, X_label, Y_label)

            criterion = nn.MSELoss()
            loss_mse = criterion(dof, affm)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(warpped_x_list[-1], Y_label)
            layer_indexs = []
            perceptual = PerceptualLoss(loss_mse, layer_indexs)
            loss_perceptual = perceptual(warpped_x_list[-1], Y_label)
            loss = (1-1 / (1 + np.exp(-(step-10000))))*loss_mse + (1 / (1 + np.exp(-(step-10000))))*(loss_multiNCC + loss_perceptual)

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_mse.item(), loss_multiNCC.item(), loss_perceptual.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - MSE "{2:4f}" - NCC "{3:4f}" - perceptual "{3:4f}"'.format(
                    step, loss.item(), loss_mse.item(), loss_multiNCC.item(), loss_perceptual.item()))
            sys.stdout.flush()

            # with lr 1e-3 + with bias
            if (step % n_checkpoint == 0):
                modelname = model_dir + '/' + model_name + "stagelvl3_" + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + model_name + "stagelvl3_" + str(step) + '.npy', lossall)

            step += 1

            if step > iteration:
                break
        print("one epoch pass")
    np.save(model_dir + '/loss' + model_name + 'stagelvl3.npy', lossall)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--modelname", type=str,
                        dest="modelname",
                        default='',
                        help="Model name")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=0.0001, help="learning rate")
    parser.add_argument("--iteration", type=int,
                        dest="iteration", default=20000,
                        help="number of total iterations")
    parser.add_argument("--checkpoint", type=int,
                        dest="checkpoint", default=50,
                        help="frequency of saving models")
    parser.add_argument("--datapath", type=str,
                        dest="datapath",
                        default='',
                        help="data path for training images")
    parser.add_argument("--labelpath", type=str,
                        dest="labelpath",
                        default='',
                        help="data path for training images")
    parser.add_argument("--fixpath", type=str,
                        dest="fixpath",
                        default='',
                        help="data path for training images")
    parser.add_argument("--fixlabelpath", type=str,
                        dest="fixlabelpath",
                        default='',
                        help="data path for training images")
    parser.add_argument("--dofpath", type=str,
                        dest="dofpath",
                        default='',
                        help="data path for training images")
    parser.add_argument("--com_initial", type=bool,
                        dest="com_initial", default=False,
                        help="True: Enable Center of Mass initialization, False: Disable")
    opt = parser.parse_args()

    lr = opt.lr
    iteration = opt.iteration
    n_checkpoint = opt.checkpoint
    datapath = opt.datapath
    labelpath = opt.labelpath
    fixpath = opt.fixpath
    fixlabelpath = opt.fixlabelpath
    dofpath = opt.dofpath

    com_initial = opt.com_initial

    model_name = opt.modelname

    # Create and initalize log file
    if not os.path.isdir("../Log"):
        os.mkdir("../Log")

    log_dir = "../Log/" + model_name + ".txt"

    with open(log_dir, "a") as log:
        log.write("Validation Dice log for " + model_name[0:-1] + ":\n")

    print("Training %s ..." % model_name)
    train()


