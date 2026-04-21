import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from RGAN_dense_121_vit_b import Attack
from torchvision import models
from Mydata import MyDataset
from torch import nn
import os
import timm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

use_cuda = True
image_nc = 3
epochs = 150
BOX_MIN = 0  # 图像归一化
BOX_MAX = 1
clip = 1
dataset_name = 'flower102'
root = '/home/zjw/HENet/dataset/' + dataset_name + '/'

transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Define what device we are using
device = torch.device("cuda:0" if (use_cuda and torch.cuda.is_available()) else "cpu")
model_num_labels = 102

# Define what target model we are using
# target_network_name = 'convnext_base'
# batch_size = 32
# pretrained_model = "/home/zjw/HENet/train_target_network/" + dataset_name + "/" + target_network_name + ".pth"
# targeted_model = timm.create_model(target_network_name, pretrained=False)
# fc_features = targeted_model.head.fc.in_features
# targeted_model.head.fc = nn.Linear(fc_features, 102)
# targeted_model.load_state_dict(torch.load(pretrained_model))
# targeted_model.to(device)
# targeted_model.eval()

target_network_name = 'swin_base_patch4_window7_224'
batch_size = 32
pretrained_model = "/home/zjw/HENet/train_target_network/" + dataset_name + "/" + target_network_name + ".pth"
targeted_model = timm.create_model(target_network_name, pretrained=False)
fc_features = targeted_model.head.in_features
targeted_model.head = nn.Linear(fc_features, 102)
targeted_model.load_state_dict(torch.load(pretrained_model))
targeted_model.to(device)
targeted_model.eval()

# target_network_name = 'mixer_b16_224'
# batch_size = 32
# pretrained_model = "/home/zjw/HENet/train_target_network/" + dataset_name + "/" + target_network_name + ".pth"
# targeted_model = timm.create_model(target_network_name, pretrained=False)
# fc_features = targeted_model.head.in_features
# targeted_model.head = nn.Linear(fc_features, 102)
# targeted_model.load_state_dict(torch.load(pretrained_model))
# targeted_model.to(device)
# targeted_model.eval()

# target_network_name = 'sequencer2d_m'
# batch_size = 32
# pretrained_model = "/home/zjw/HENet/train_target_network/" + dataset_name + "/" + target_network_name + ".pth"
# targeted_model = timm.create_model(target_network_name, pretrained=False)
# fc_features = targeted_model.head.in_features
# targeted_model.head = nn.Linear(fc_features, 102)
# targeted_model.load_state_dict(torch.load(pretrained_model))
# targeted_model.to(device)

train_data = MyDataset(txt=root + 'flower-trn.txt', transform=transform)
dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                        pin_memory=True, num_workers=4)

test_data = MyDataset(txt=root + 'flower-val.txt', transform=transform)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size,
                             pin_memory=True, num_workers=4)

def main():
    RGAN = Attack(device,
                  targeted_model,
                  model_num_labels,
                  image_nc,
                  BOX_MIN,
                  BOX_MAX,
                  clip,
                  target_network_name,
                  )
    RGAN.train(dataloader, test_dataloader, len(test_data), epochs)


if __name__ == '__main__':
    main()
