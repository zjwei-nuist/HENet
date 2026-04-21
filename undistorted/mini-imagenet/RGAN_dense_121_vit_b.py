import numpy
import torch.nn as nn
import torch
import numpy as np
import models_dense_121_vit_b as models
import torch.nn.functional as F
import torchvision
import os
from skimage.segmentation import slic

from tensorboardX import SummaryWriter
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import lpips


def channel_first_to_last(img):
    img = img.swapaxes(0, 2)
    img = img.swapaxes(0, 1)
    return img


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


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


class Attack:
    def __init__(self,
                 device,
                 model,
                 model_num_labels,
                 image_nc,
                 box_min,
                 box_max,
                 clip,
                 target_network_name,):
        output_nc = image_nc
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max
        self.clip = clip

        self.gen_input_nc = image_nc
        self.netG = models.Generator('densenet121', 'vit_base_patch16_224').to(device)
        self.netDisc = models.Discriminator(image_nc).to(device)

        # initialize all weights
        self.netG.decoder.apply(weights_init)
        self.netDisc.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(list(self.netG.encoder.encoder1.parameters()) +
                                            list(self.netG.fusion.parameters()) +
                                            list(self.netG.decoder.parameters()),
                                            lr=0.001)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=0.001)

        self.models_path = './target_models/' + target_network_name + '/'
        self.writer = SummaryWriter(logdir=self.models_path, comment=target_network_name)
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
        self.f = open(self.models_path + 'loss.txt', 'a', encoding='utf-8')

        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)

    def train_batch(self, x, labels):

        for i in range(1):
            perturbation = self.netG(x)
            adv_images = torch.clamp(perturbation, -self.clip, self.clip) + x
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            self.optimizer_D.zero_grad()
            pred_real = self.netDisc(x)
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            loss_D_real.backward()

            pred_fake = self.netDisc(adv_images.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            loss_D_fake.backward()
            loss_D_GAN = loss_D_fake + loss_D_real
            self.optimizer_D.step()

        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()

            # cal G's loss in GAN
            pred_fake = self.netDisc(adv_images)
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake.backward(retain_graph=True)

            perturbation = adv_images - x
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))

            # cal adv loss
            logits_model = self.model(adv_images)
            probs_model = F.softmax(logits_model, dim=1)
            onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]

            # C&W loss function
            real = torch.sum(onehot_labels * probs_model, dim=1)
            other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(other)
            loss_adv = torch.max(real - other, zeros)
            loss_adv = torch.sum(loss_adv)

            adv_lambda = 10
            pert_lambda = 1
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            loss_G.backward()
            self.optimizer_G.step()

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), \
               loss_adv.item()

    def train(self, train_dataloader, test_dataloader, total, epochs):
        best = 0
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            if epoch == 51:
                self.optimizer_G = torch.optim.Adam(list(self.netG.encoder.encoder1.parameters()) +
                                                    list(self.netG.fusion.parameters()) +
                                                    list(self.netG.decoder.parameters()),
                                                    lr=0.0001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.0001)

            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0

            self.netG.train()
            self.netDisc.train()

            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch \
                    = self.train_batch(images, labels)

                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch

            num_batch_train = len(train_dataloader)
            num = 0
            num_correct = 0
            l2_adv_sum = 0
            psnr_adv_sum = 0
            ssim_adv_sum = 0
            lpips_adv_sum = 0

            self.netG.eval()
            self.netDisc.eval()
            for i, data in enumerate(test_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                with torch.no_grad():
                    perturbation = self.netG(images)
                    perturbation = torch.clamp(perturbation, -self.clip, self.clip)
                    adv_img = perturbation + images

                    adv_img = torch.clamp(adv_img, 0, 1)
                    pred_lab = torch.argmax(self.model(adv_img), 1)
                    num_correct += torch.sum(pred_lab == labels, 0)

                    ori_pred = torch.argmax(self.model(images), 1)
                    num += torch.sum(ori_pred == labels, 0)

                    l2_adv = cal_l2norm(images, adv_img)
                    l2_adv_sum += l2_adv

                    psnr_adv = cal_psnr(images, adv_img)
                    psnr_adv_sum += psnr_adv

                    ssim_adv = cal_ssim(images, adv_img)
                    ssim_adv_sum += ssim_adv

                    lpips_adv = torch.mean(self.loss_fn_vgg(images, adv_img))
                    lpips_adv_sum += lpips_adv

            num_batch_test = len(test_dataloader)
            cer_ori = 100 * (1 - num.item() / total)
            cer_adv = 100 * (1 - num_correct.item() / total)

            l2_adv_epoch = l2_adv_sum / num_batch_test
            psnr_adv_epoch = psnr_adv_sum / num_batch_test
            ssim_adv_epoch = ssim_adv_sum / num_batch_test
            lpips_adv_epoch = lpips_adv_sum / num_batch_test

            print("epoch %d:\n"
                  "loss_D: %.8f, loss_G_fake: %.8f,\n"
                  "loss_perturb: %.8f, loss_adv: %.8f,\n"
                  "cer_ori: %.3f, cer_adv: %.3f,\n"
                  "L2_adv: %.2f,psnr_adv: %.2f,ssim_adv: %.4f, lpips_adv: %.4f,\n" %
                  (epoch, loss_D_sum / num_batch_train, loss_G_fake_sum / num_batch_train,
                   loss_perturb_sum / num_batch_train, loss_adv_sum / num_batch_train,
                   cer_ori, cer_adv, l2_adv_epoch, psnr_adv_epoch, ssim_adv_epoch, lpips_adv_epoch))

            print("epoch %d:\n"
                  "loss_D: %.8f, loss_G_fake: %.8f,\n"
                  "loss_perturb: %.8f, loss_adv: %.8f,\n"
                  "cer_ori: %.3f, cer_adv: %.3f,\n"
                  "L2_adv: %.2f,psnr_adv: %.2f,ssim_adv: %.4f, lpips_adv: %.4f,\n" %
                  (epoch, loss_D_sum / num_batch_train, loss_G_fake_sum / num_batch_train,
                   loss_perturb_sum / num_batch_train, loss_adv_sum / num_batch_train,
                   cer_ori, cer_adv, l2_adv_epoch, psnr_adv_epoch, ssim_adv_epoch, lpips_adv_epoch), file=self.f)

            self.writer.add_scalars("cer_adv", {"cer_adv": cer_adv}, epoch)
            self.writer.add_scalars("l2_adv", {"l2_adv": l2_adv_epoch}, epoch)
            self.writer.add_scalars("psnr_adv", {"psnr_adv": psnr_adv_epoch}, epoch)
            self.writer.add_scalars("ssim_adv", {"ssim_adv": ssim_adv_epoch}, epoch)
            self.writer.add_scalars("lpips_adv", {"lpips_adv": lpips_adv_epoch}, epoch)

            if epoch % 1 == 0:
                netG_file_name = self.models_path + 'netG_epoch_' + str(epoch) + '.pth'
                torch.save(self.netG.state_dict(), netG_file_name)

            if cer_adv > best:
                best = cer_adv
                best_epoch = epoch

        print('best epoch:  %d, best CER: %.3f' % (best_epoch, best), file=self.f)
        self.writer.close()
        self.f.close()
