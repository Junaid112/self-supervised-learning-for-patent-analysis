import os
import numpy as np
import pandas as pd
from torchvision.io import read_image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
import cv2
from byol_pytorch import BYOL
from torchvision import models

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        image = cv2.imread(img_path)
        image=image.transpose(2, 0, 1)
        image=torch.from_numpy(image.astype(np.float32))
        torch_resize=Resize([256,256])
        image=torch_resize(image)
        label = self.img_labels.iloc[idx, 0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image,label

train_dataset = CustomImageDataset(r"D:\uni hannover\Masterarbeit\database\clef_ip_clean_v2\train\blockorcircuit\blockorcicuitlabel.csv", r"D:\uni hannover\Masterarbeit\database\clef_ip_clean_v2\train\blockorcircuit")
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)



resnet = models.resnet50(pretrained=False)
fc_inputs=resnet.fc.in_features
resnet.fc=nn.Sequential(
    nn.Linear(fc_inputs,256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256,10),
    nn.Softmax(dim=1))
resnet.load_state_dict(torch.load('D:\\uni hannover\\Masterarbeit/improved-net.pt'))
# resnet.eval()
learner = BYOL(
    resnet,
    image_size = 256,
    hidden_layer = 'avgpool'
)
learner.load_state_dict(torch.load(r'D:\uni hannover\Masterarbeit./learnertest-net.pt'))
opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

for data in train_dataloader:
    images,targets=data
    loss = learner(images)
    print(loss)
    opt.zero_grad()
    
    loss.backward()
    opt.step()
    learner.update_moving_average() 

torch.save(resnet.state_dict(), 'D:\\uni hannover\\Masterarbeit/improved-net.pt')
torch.save(learner.state_dict(),r'D:\uni hannover\Masterarbeit./learnertest-net.pt')