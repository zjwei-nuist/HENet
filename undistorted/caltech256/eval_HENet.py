import os
import timm
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import models_dense_121_vit_b as models
from Mydata import MyDataset
from torch import nn
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import lpips

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
clip = 1
model_path = './target_models/'
dataset_name = 'caltech256'
root = '/home/zjw/HENet/dataset/' + dataset_name + '/'


use_cuda = True
image_nc = 3
batch_size = 32

transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

gen_input_nc = image_nc

# Define what device we are using
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

test_data = MyDataset(txt=root + 'dataset-val.txt', transform=transform)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size,
                             pin_memory=True, num_workers=1)
total = len(test_data)

# target_network_name = 'convnext_base'
# batch_size = 32
# pretrained_model = "/home/zjw/HENet/train_target_network/" + dataset_name + "/" + target_network_name + ".pth"
# targeted_model = timm.create_model(target_network_name, pretrained=False)
# fc_features = targeted_model.head.fc.in_features
# targeted_model.head.fc = nn.Linear(fc_features, 257)
# targeted_model.load_state_dict(torch.load(pretrained_model))
# targeted_model.to(device)
# targeted_model.eval()

# target_network_name = 'swin_base_patch4_window7_224'
# batch_size = 32
# pretrained_model = "/home/zjw/HENet/train_target_network/" + dataset_name + "/" + target_network_name + ".pth"
# targeted_model = timm.create_model(target_network_name, pretrained=False)
# fc_features = targeted_model.head.in_features
# targeted_model.head = nn.Linear(fc_features, 257)
# targeted_model.load_state_dict(torch.load(pretrained_model))
# targeted_model.to(device)
# targeted_model.eval()

# target_network_name = 'mixer_b16_224'
# batch_size = 32
# pretrained_model = "/home/zjw/HENet/train_target_network/" + dataset_name + "/" + target_network_name + ".pth"
# targeted_model = timm.create_model(target_network_name, pretrained=False)
# fc_features = targeted_model.head.in_features
# targeted_model.head = nn.Linear(fc_features, 257)
# targeted_model.load_state_dict(torch.load(pretrained_model))
# targeted_model.to(device)
# targeted_model.eval()

target_network_name = 'sequencer2d_m'
batch_size = 32
pretrained_model = "/home/zjw/HENet/train_target_network/" + dataset_name + "/" + target_network_name + ".pth"
targeted_model = timm.create_model(target_network_name, pretrained=False)
fc_features = targeted_model.head.in_features
targeted_model.head = nn.Linear(fc_features, 257)
targeted_model.load_state_dict(torch.load(pretrained_model))
targeted_model.to(device)

# load the generator of adversarial examples
pretrained_generator_path = model_path + target_network_name + '/netG_epoch_best.pth'
pretrained_G = models.Generator('densenet121', 'vit_base_patch16_224').to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

save_path = model_path + target_network_name + '/save_img/'

loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

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


def main():
    num = 0
    num_correct = 0
    l2_adv_sum = 0
    psnr_adv_sum = 0
    ssim_adv_sum = 0
    lpips_adv_sum = 0

    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            test_img, test_label = data
            test_img, test_label = test_img.to(device), test_label.to(device)

            perturbation = pretrained_G(test_img)
            perturbation = torch.clamp(perturbation, -clip, clip)
            adv_img = perturbation + test_img
            adv_img = torch.clamp(adv_img, 0, 1)

            pred_lab = torch.argmax(targeted_model(adv_img), 1)
            num_correct += torch.sum(pred_lab == test_label, 0)

            ori_pred = torch.argmax(targeted_model(test_img), 1)
            num += torch.sum(ori_pred == test_label, 0)

            l2_adv = cal_l2norm(test_img, adv_img)
            l2_adv_sum += l2_adv

            psnr_adv = cal_psnr(test_img, adv_img)
            psnr_adv_sum += psnr_adv

            ssim_adv = cal_ssim(test_img, adv_img)
            ssim_adv_sum += ssim_adv

            lpips_adv = torch.mean(loss_fn_vgg(test_img, adv_img))
            lpips_adv_sum += lpips_adv

        num_batch_test = len(test_dataloader)
        cer_ori = 100 * (1 - num.item() / total)
        cer_adv = 100 * (1 - num_correct.item() / total)

        l2_adv_epoch = l2_adv_sum / num_batch_test
        psnr_adv_epoch = psnr_adv_sum / num_batch_test
        ssim_adv_epoch = ssim_adv_sum / num_batch_test
        lpips_adv_epoch = lpips_adv_sum / num_batch_test

        print("cer_ori: %.3f, cer_adv: %.3f,\n"
              "l2_adv: %.2f, psnr_adv: %.2f, ssim_adv: %.4f, lpips_adv: %.4f\n" %
              (cer_ori, cer_adv,
               l2_adv_epoch, psnr_adv_epoch, ssim_adv_epoch, lpips_adv_epoch))

if __name__ == '__main__':
    main()