import torch.nn.functional as F
import os
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import warnings
import models_dense_121_vit_b as models


import time
import numpy as np


import timm
from Mydata import ImageNetDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import lpips


def channel_first_to_last(img):
    img = img.swapaxes(0, 2)
    img = img.swapaxes(0, 1)
    return img


def cal_l2norm(image1, image2):
    l2_list = []
    for i in range(len(image1)):
        img1 = (image1[i]).detach().cpu().numpy()
        img2 = (image2[i]).detach().cpu().numpy()
        img1 = channel_first_to_last(img1)
        img2 = channel_first_to_last(img2)
        l2_list.append(np.linalg.norm(img1 - img2))
    return np.round(np.mean(l2_list), 2)


def cal_psnr(image1, image2):
    psnr_list = []
    for i in range(len(image1)):
        img1 = (image1[i] * 255.).detach().cpu().numpy().astype('uint8').squeeze()
        img2 = (image2[i] * 255.).detach().cpu().numpy().astype('uint8').squeeze()
        img1 = channel_first_to_last(img1)
        img2 = channel_first_to_last(img2)
        psnr_list.append(compare_psnr(img1, img2, data_range=255))
    return np.round(np.mean(psnr_list), 2)


def cal_ssim(image1, image2):
    ssim_list = []
    for i in range(len(image1)):
        img1 = (image1[i] * 255.).detach().cpu().numpy().astype('uint8').squeeze()
        img2 = (image2[i] * 255.).detach().cpu().numpy().astype('uint8').squeeze()
        img1 = channel_first_to_last(img1)
        img2 = channel_first_to_last(img2)
        ssim_list.append(compare_ssim(img1, img2, data_range=255, multichannel=True))
    return np.round(np.mean(ssim_list), 4)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#####################
# Model initialize: #
#####################
pretrained_G = models.Generator('densenet121', 'vit_base_patch16_224').to(device)
pretrained_G.decoder.apply(weights_init)
optimizer_G = torch.optim.Adam(list(pretrained_G.encoder.encoder1.parameters()) +
                               list(pretrained_G.fusion.parameters()) +
                               list(pretrained_G.decoder.parameters()),
                               lr=0.001)

# target_model_name = 'convnext_base'
# target_model_name = 'swin_base_patch4_window7_224'
target_model_name = 'mixer_b16_224'
# target_model_name = 'sequencer2d_m'

model = timm.create_model(target_model_name, pretrained=True)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

train_data = ImageNetDataset(csv_file='/home/zjw/HENet/dataset/NIPS2017/images.csv',
                             root_dir='/home/zjw/HENet/dataset/NIPS2017/images',
                             transform=transform)
trainloader = DataLoader(dataset=train_data, batch_size=1,
                         pin_memory=True, num_workers=8)

clip = 1.
epoch = 5001

try:
    totalTime = time.time()
    failnum = 0
    count = 0.0
    l2_adv_sum = 0
    psnr_adv_sum = 0
    ssim_adv_sum = 0
    lpips_adv_sum = 0
    total = len(train_data)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    for i_batch, mydata in enumerate(trainloader):
        test_img, test_label = mydata
        test_img, test_label = test_img.to(device), test_label.to(device)
        start_time = time.time()

        for i_epoch in range(epoch):
            #################
            #    train:   #
            #################
            perturbation = pretrained_G(test_img)
            perturbation = torch.clamp(perturbation, -clip, clip)
            adv_img = perturbation + test_img
            adv_img = torch.clamp(adv_img, 0, 1)

            out = model(adv_img).to(device)
            _, pre = torch.max(out.data, 1)

            probs_model = F.softmax(out, dim=1)
            onehot_labels = torch.eye(1000, device=device)[test_label]

            real = torch.sum(onehot_labels * probs_model, dim=1)
            other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(other)
            loss_adv = torch.max(real - other, zeros)
            loss_adv = torch.sum(loss_adv)

            suc_rate = ((pre != test_label).sum()).cpu().detach().numpy()

            perturbation = adv_img - test_img
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))

            adv_lambda = 10
            pert_lambda = 1
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb

            #################
            #     Exit:     #
            #################
            # if suc_rate == 1:
            if suc_rate == 1 and i_epoch >= 1000:
                count += 1
                break

            if i_epoch >= 5000:
                failnum += 1
                count += 1
                break

            #################
            #   Backward:   #
            #################
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

        # save_image(CGT, args.outputpath + source_name + "-" + target_name + '\\CGT.png')
        with torch.no_grad():
            print('count:', count)
            l2_adv = cal_l2norm(test_img, adv_img)
            l2_adv_sum += l2_adv

            psnr_adv = cal_psnr(test_img, adv_img)
            psnr_adv_sum += psnr_adv

            ssim_adv = cal_ssim(test_img, adv_img)
            ssim_adv_sum += ssim_adv

            lpips_adv = torch.mean(loss_fn_vgg(test_img, adv_img))
            lpips_adv_sum += lpips_adv

            print("suc rate :" + str(suc_rate))
            print("l2 :" + str(l2_adv))
            print("PSNR :" + str(psnr_adv))
            print("SSIM :" + str(ssim_adv))
            print("LPIPS :" + str(lpips_adv))

    totalstop_time = time.time()
    time_cost = totalstop_time - totalTime
    Total_suc_rate = (count - failnum) / total
    avg_psnr = psnr_adv_sum / total
    avg_ssim = ssim_adv_sum / total
    avg_l2 = l2_adv_sum / total
    avg_lpips = lpips_adv_sum / total
    print('=====================================================')
    print('Target Model: ', target_model_name)
    print("Total cost time :" + str(time_cost))
    print("Total suc rate :" + str(Total_suc_rate))
    print("Total l2 :" + str(avg_l2))
    print("Total PSNR :" + str(avg_psnr))
    print("Total SSIM :" + str(avg_ssim))
    print("Total LPIPS :" + str(avg_lpips))
except:
    raise
