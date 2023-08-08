import os
import numpy as np
import pandas as pd
from torchvision.io import read_image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from torchvision import transforms
import cv2
from byol_pytorch import BYOL
from torchvision import models
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,self.img_labels.iloc[idx, 0], self.img_labels.iloc[idx, 1])
        image = cv2.imread(img_path)
        image=image.transpose(2, 0, 1)
        image=torch.from_numpy(image.astype(np.float32))
        torch_resize=Resize([224,224])
        image=torch_resize(image)
        label = self.img_labels.iloc[idx, 0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image,label

train_dataset = CustomImageDataset(r"/nfs/home/duanj/database/clef_ip_clean_v2_train.csv", r"/nfs/home/duanj/database/clef_ip_clean_v2/train")
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
device = torch.device("cuda:0")

resnet = models.resnet50(pretrained=False)
resnet = resnet.to(device)
learner = BYOL(
    resnet,
    image_size = 256,
    hidden_layer = 'avgpool'
)

opt = torch.optim.Adam(learner.parameters(), lr=1e-6)
history=[]
for epoch in range (100):
    train_loss = 0.0
    for data in train_dataloader:
        images,targets=data
        images = images.to(device)        
        loss = learner(images)
        train_loss+=loss.item() * images.size(0)
        opt.zero_grad()    
        loss.backward()
        opt.step()
        learner.update_moving_average() 
    avg_train_loss = train_loss/train_dataset.__len__()
    history.append(avg_train_loss)
history = np.array(history)
plt.plot(history[:])
plt.legend(['Tr Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.savefig('/nfs/home/duanj/BYOLmodel/with_all_img/newloss_curve.png')


torch.save(resnet.state_dict(), '/nfs/home/duanj/BYOLmodel/with_all_img/newimproved-net.pt')
torch.save(learner.state_dict(),r'/nfs/home/duanj/BYOLmodel/with_all_img/newlearnertest-net.pt')