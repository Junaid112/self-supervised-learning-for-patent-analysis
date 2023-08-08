import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import cv2
import Simclr
import matplotlib.pyplot as plt



train_transform = transforms.Compose([transforms.RandomResizedCrop((224, 224)),                              
                                      transforms.RandomHorizontalFlip(),                                                                          
                                      transforms.Normalize(
                                          mean=torch.tensor([0.485, 0.456, 0.406]),
                                          std=torch.tensor([0.229, 0.224, 0.225]))])

device = torch.device("cuda:0")
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0], self.img_labels.iloc[idx, 1])
        image = cv2.imread(img_path)
        image=image.transpose(2, 0, 1)
        image=torch.from_numpy(image.astype(np.float32))        
        label = self.img_labels.iloc[idx, 0]
        if self.transform:
            imageL = self.transform(image)
            imageR = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return imageL, imageR ,label
    
train_dataset = CustomImageDataset(r"/nfs/home/duanj/database/train_with_allperspective.csv", r"/nfs/home/duanj/database/clef_ip_clean_v2/train",transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)


model=Simclr.SimCLRStage1().to(device)
lossLR=Simclr.ContrastiveLoss(64).to(device)
optimizer=torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

history=[]
for epoch in range(100):
    model.train()
    total_loss=0.0
    for batch,(imgL,imgR,labels) in enumerate(train_dataloader):
        
            imgL,imgR=imgL.to(device),imgR.to(device)            
            _, pre_L=model(imgL)
            _, pre_R=model(imgR)

            loss=lossLR(pre_L,pre_R)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
            total_loss += loss.detach().item()

    print("epoch loss:",total_loss/len(train_dataset)*64)
    history.append(total_loss/len(train_dataset)*64)

plt.plot(history[:])
plt.legend(['Tr Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')

plt.savefig('/nfs/home/duanj/SimCLRmodel/with_all_img/lrloss_curve_july.png')
torch.save(model.state_dict(), '/nfs/home/duanj/SimCLRmodel/with_all_img/july_SimCLR-net.pt')















