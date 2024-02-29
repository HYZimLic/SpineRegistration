import os
from argparse import ArgumentParser
from PIL import Image
import nibabel as nib
import numpy as np
import torch
import random
import segmentation.config
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import SimpleITK as sitk
import torchvision.transforms as transforms
from tqdm import tqdm
from segmentation.dataset.dataset_lits_test import Test_Datasets
from segmentation.models import ResUNet
from collections import OrderedDict
from reconstruction.net import ReconNet
from segmentation.utils.common import to_one_hot_3d
from segmentation.utils.metrics import DiceAverage
from segmentation.utils import logger,common
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from TS_SAR_model import TS_SAR_stage, AffineCOMTransform, Center_of_mass_initial_pairwise
from Functions import save_img, load_4D, min_max_norm


class MedReconDataset(Dataset):
    """ 3D Reconstruction Dataset."""

    def __init__(self, csv_file=None, data_dir=None, transform=None):
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        images = np.zeros((opt.input_size, opt.input_size, opt.num_views),
                          dtype=np.uint8)  ### input image size (H, W, C)
        ### load image
        for view_idx in range(opt.num_views):
            image_path = os.path.join(exp_path, 'data/2D_projection_{}.jpg'.format(view_idx + 1))
            img = Image.open(image_path).resize((opt.input_size, opt.input_size))
            images[:, :, view_idx] = np.array(img)
        if self.transform:
            images = self.transform(images)

        return (images)

def predict_one_img(model, img_dataset, args):
    dataloader = DataLoader(dataset=img_dataset, batch_size=1, num_workers=0, shuffle=False)
    model.eval()
    test_dice = DiceAverage(args.n_labels)
    target = to_one_hot_3d(img_dataset.label, args.n_labels)

    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader)):
            data = data.to(device)
            output_low, output = model(data)
            # output = nn.functional.interpolate(output, scale_factor=(1//args.slice_down_scale,1//args.xy_down_scale,1//args.xy_down_scale), mode='trilinear', align_corners=False) # 空间分辨率恢复到原始size
            img_dataset.update_result(output.detach().cpu())
            img_dataset.update_result(output_low.detach().cpu())

    pred = img_dataset.recompone_result()
    pred = torch.argmax(pred, dim=1)
    pred_img = common.to_one_hot_3d(pred, args.n_labels)
    mid = img_dataset.recompone_result()
    mid = torch.argmax(mid, dim=1)
    test_dice.update(pred_img, target)

    test_dice = OrderedDict({'Dice_liver': test_dice.avg[1]})
    if args.n_labels == 3: test_dice.update({'Dice_tumor': test_dice.avg[2]})
    pred = np.asarray(pred.numpy(), dtype='uint8')
    mid = np.asarray(mid.numpy(), dtype='uint8')
    if args.postprocess:
        pass  # TO DO
    pred = sitk.GetImageFromArray(np.squeeze(pred, axis=0))

    return test_dice, pred, target, mid

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--exp', type=int, default=1,
                        help='experiments index')
    parser.add_argument('--seed', type=int, default=1,
                        metavar='N', help='manual seed for GPUs to generate random numbers')
    parser.add_argument('--num-views', type=int, default=1,
                        help='number of views/projections in inputs')
    parser.add_argument('--input-size', type=int, default='',
                        help='dimension of input view size')
    parser.add_argument('--output-size', type=int, default='',
                        help='dimension of ouput 3D model size')
    parser.add_argument('--output-channel', type=int, default=0,
                        help='dimension of ouput 3D model size')
    parser.add_argument('--start-slice', type=int, default=0,
                        help='the idx of start slice in 3D model')
    parser.add_argument('--test', type=int, default=1,
                        help='number of total testing samples')
    parser.add_argument('--vis_plane', type=int, default=0,
                        help='visualization plane of 3D images: [0,1,2]')
    parser.add_argument("--modelpath", type=str,
                        dest="modelpath",
                        default='',
                        help="Pre-trained Model path")
    parser.add_argument("--savepath", type=str,
                        dest="savepath", default='../Result',
                        help="path for saving images")
    parser.add_argument("--fixed", type=str,
                        dest="fixed", default='',
                        help="fixed image")
    parser.add_argument("--fixedlabel", type=str,
                        dest="fixedlabel", default='',
                        help="fixed image")
    parser.add_argument("--moving", type=str,
                        dest="moving", default='',
                        help="moving image")
    parser.add_argument("--movinglabel", type=str,
                        dest="movinglabel", default='',
                        help="moving image")
    parser.add_argument("--com_initial", type=bool,
                        dest="com_initial", default=False,
                        help="True: Enable Center of Mass initialization, False: Disable")
    opt = parser.parse_args()

    exp_path = '' # 2D X-Ray file path
    midd_path = ''# low-reselution feature file path
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    # define model
    rec_model = ReconNet(in_channels=1, out_channels=opt.output_channel)
    rec_model = torch.nn.DataParallel(rec_model).cuda()
    # enable CUDNN benchmark
    cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.516], std=[0.264])
    test_dataset = MedReconDataset(
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True)
    # load model
    ckpt_file = os.path.join(exp_path, 'model/model.pth.tar')
    if os.path.isfile(ckpt_file):
        print("=> loading checkpoint '{}' ".format(ckpt_file))
        checkpoint = torch.load(ckpt_file)
        best_loss = checkpoint['best_loss']
        rec_model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' ".format(ckpt_file))
    else:
        print("=> no checkpoint found at '{}'".format(ckpt_file))

    mid = np.zeros((opt.test, opt.middle_channel, opt.middle_size, opt.middle_size), dtype=np.float32)
    pred = np.zeros((opt.test, opt.output_channel, opt.output_size, opt.output_size), dtype=np.float32)
    input_var = Variable(input)
    input_var = input_var.cuda()
    output, middle = rec_model(input_var)
    mid[:, :, :] = middle.data.float()
    pred[:, :, :] = output.data.float()
    save_path = os.path.join(exp_path, 'result')
    mid_path = os.path.join(midd_path, 'middle')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_mid_name = os.path.join(mid_path, '')
    file_name = os.path.join(save_path, '')
    np.savez(file_mid_name, mid=mid) # low-resolution move feature
    np.savez(file_name, pred=pred) # reconstructed features

    args = segmentation.config.args
    save_seg_path = os.path.join('', args.save)
    # model info
    seg_model = ResUNet(in_channel=1, out_channel=args.n_labels, training=False).cuda()
    seg_model = torch.nn.DataParallel(seg_model, device_ids=args.gpu_id)  # multi-GPU
    ckpt = torch.load(''.format(save_seg_path))
    seg_model.load_state_dict(ckpt['net'])
    test_log = logger.Test_Logger(save_seg_path, "test_log")
    # data info
    moveresult_save_path = ''.format(save_seg_path)
    if not os.path.exists(moveresult_save_path):
        os.mkdir(moveresult_save_path)
    fixresult_save_path = ''.format(save_seg_path)
    if not os.path.exists(fixresult_save_path):
        os.mkdir(fixresult_save_path)
    movedatasets = Test_Datasets(args.test_data_path, args=args)
    fixdatasets = Test_Datasets(args.test_data_path, args=args)
    for img_dataset, file_idx in movedatasets:
        test_dice1, pred_moveimg, target_moveimg, midmove = predict_one_img(seg_model, img_dataset, args)
        test_log.update(file_idx, test_dice1)
        sa_moveimg = np.multiply(pred_moveimg, target_moveimg)
        crop_size = (128,128,128)
        start_index = tuple((np.array(sa_moveimg.shape) - np.array(crop_size)) // 2)
        sailent_moveimg = sa_moveimg[start_index[0]:start_index[0] + crop_size[0],
                         start_index[1]:start_index[1] + crop_size[1],
                         start_index[2]:start_index[2] + crop_size[2]]
        sitk.WriteImage(sailent_moveimg,
                        os.path.join(moveresult_save_path))# high-resolution move feature
    for img_dataset, file_idx in fixdatasets:
        test_dice2, pred_fiximg, target_fiximg, midfix = predict_one_img(seg_model, img_dataset, args)
        test_log.update(file_idx, test_dice2)
        sa_fiximg = np.multiply(pred_fiximg, target_fiximg)
        crop_size = (128,128,128)
        start_index = tuple((np.array(sa_fiximg.shape) - np.array(crop_size)) // 2)
        sailent_fiximg = sa_fiximg[start_index[0]:start_index[0] + crop_size[0],
                         start_index[1]:start_index[1] + crop_size[1],
                         start_index[2]:start_index[2] + crop_size[2]]
        sitk.WriteImage(midfix,
                        os.path.join(fixresult_save_path), '')# low-resolution fix feature
        sitk.WriteImage(sailent_fiximg,
                        os.path.join(fixresult_save_path), '')# high-resolution fix feature

    savepath = opt.savepath
    fixed_path = opt.fixed# low-resolution fix feature
    fixed_label_path = opt.fixedlabel# high-resolution fix feature
    moving_path = opt.moving# low-resolution move feature
    moving_label_path = opt.movinglabel# high-resolution move feature
    com_initial = opt.com_initial
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    model = TS_SAR_stage(img_size=128, patch_size=[3, 7, 15], stride=[2, 4, 8], num_classes=6,
                          embed_dims=[256, 256, 256],
                          num_heads=[2, 2, 2], mlp_ratios=[2, 2, 2], qkv_bias=False, qk_scale=None, drop_rate=0.,
                          attn_drop_rate=0., norm_layer=nn.Identity,
                          depths=[4, 4, 4], sr_ratios=[1, 1, 1], num_stages=3, linear=False).to(device)

    print(f"Loading model weight {opt.modelpath} ...")
    model.load_state_dict(torch.load(opt.modelpath))
    model.eval()
    affine_transform = AffineCOMTransform().cuda()
    init_center = Center_of_mass_initial_pairwise()
    fixed_base = os.path.basename(fixed_path)
    moving_base = os.path.basename(moving_path)
    fixed_img_nii = nib.load(fixed_path)
    header, affine = fixed_img_nii.header, fixed_img_nii.affine
    fixed_img = fixed_img_nii.get_fdata()
    fixed_img = np.reshape(fixed_img, (1,) + fixed_img.shape)
    moving_img = load_4D(moving_path)
    moving_label = load_4D(moving_label_path)
    fixed_label = load_4D(fixed_label_path)
    fixed_img = torch.from_numpy(fixed_img).float().to(device).unsqueeze(dim=0)
    moving_img = torch.from_numpy(moving_img).float().to(device).unsqueeze(dim=0)
    fixed_label = torch.from_numpy(fixed_label).float().to(device).unsqueeze(dim=0)
    moving_label = torch.from_numpy(moving_label).float().to(device).unsqueeze(dim=0)
    with torch.no_grad():
        if com_initial:
            moving_img, init_flow = init_center(moving_img, fixed_img)
        X_down = F.interpolate(moving_img, scale_factor=0.5, mode="trilinear", align_corners=True)
        Y_down = F.interpolate(fixed_img, scale_factor=0.5, mode="trilinear", align_corners=True)
        warpped_x_list, y_list, affine_para_list, affm = model(X_down, Y_down, moving_label, fixed_label)
        X_Y, affine_matrix = affine_transform(moving_img, affine_para_list[-1])
        X_Y_cpu = X_Y.data.cpu().numpy()[0, 0, :, :, :]

        save_img(X_Y_cpu, f"{savepath}/warped_{moving_base}", header=header, affine=affine)

    print("Result saved to :", savepath)