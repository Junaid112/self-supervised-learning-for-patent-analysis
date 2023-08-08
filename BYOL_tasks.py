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
from torchvision import models,datasets,transforms
import matplotlib.pyplot as plt

# load targets
species = ['bottom_view','front_view', 'left_side_view','perspective_view', 'rear_view', 'right_side_view',   'top_view'  ]
species_to_idx = dict((c, i) for i, c in enumerate(species))
train_transform = transforms.Compose([transforms.RandomResizedCrop((256, 256)),                                      
                                      transforms.RandomHorizontalFlip(),                                                                           
                                      transforms.Normalize(
                                          mean=torch.tensor([0.485, 0.456, 0.406]),
                                          std=torch.tensor([0.229, 0.224, 0.225]))])
torch_resize=Resize([256,256])
test_transform = transforms.Compose([torch_resize,    
        transforms.Normalize(
        mean=torch.tensor([0.485, 0.456, 0.406]),
        std=torch.tensor([0.229, 0.224, 0.225]))])

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv2.imread(img_path)
        image=image.transpose(2, 0, 1)
        image=torch.from_numpy(image.astype(np.float32))
        torch_resize=Resize([256,256])
        image=torch_resize(image)
        label = self.img_labels.iloc[idx, 1]
        if label in species_to_idx.keys():
            label=species_to_idx[label]            
        else:
            print('unexpected image type')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image,label
def confusion_matrix(preds, labels, conf_matrix):
    preds = preds.tolist()
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix
def compute_accuracy(counts: list , per_kinds: list):
    accuracy_of_each_kind=[]
    for rate in counts/per_kinds:
        accuracy_of_each_kind.append(rate)
    average_accuracy=sum(counts)/sum(per_kinds)
    return average_accuracy,accuracy_of_each_kind

device = torch.device("cuda:0")
#load train and test dataset
train_dataset = CustomImageDataset(r"/nfs/home/duanj/database/fc_train8000_t300.csv", r"/nfs/home/duanj/database/clef_ip_clean_v2/train",transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validata_dataset=CustomImageDataset(r"/nfs/home/duanj/database/clef_ip_clean_v2_test.csv", r"/nfs/home/duanj/database/clef_ip_clean_v2/test",transform=test_transform)
validata_dataloader=DataLoader(validata_dataset, batch_size=64, shuffle=True)
# load the trained model
resnet = models.resnet50(pretrained=False)
resnet.load_state_dict(torch.load('/nfs/home/duanj/BYOLmodel/with_all_img/newimproved-net.pt'))
fc_inputs = resnet.fc.in_features
resnet.fc=nn.Sequential(
    nn.Linear(fc_inputs, 512),
    nn.ReLU(),        
    nn.Linear(512,7)
    )
resnet=resnet.to(device)
add=nn.Sequential(nn.Softmax(dim=1))
add=add.to(device)
loss_criterion = torch.nn.CrossEntropyLoss()
loss_criterion=loss_criterion.to(device)
for name, value in resnet.named_parameters():
    if 'fc' not in name:
        value.requires_grad = False
params = filter(lambda p: p.requires_grad, resnet.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)  

history=[]
tkind_acc=[]
vkind_acc=[]
# train process
for epoch in range (100):    
    train_loss = 0.0
    train_acc = 0.0
    valid_loss = 0.0
    valid_acc = 0.0
    train_conf_matrix = torch.zeros(7, 7)
    valid_conf_matrix = torch.zeros(7, 7)
    for datas in train_dataloader:
        images,targets=datas
        images = images.to(device)     
        targets=torch.as_tensor(targets)
        targets=targets.to(device)        
        outputs=resnet(images)                       
        loss = loss_criterion(outputs, targets)
        pre=add(outputs)
        ret, predictions= torch.max(pre.data,1)  
        optimizer.zero_grad()        
        loss.backward()
        optimizer.step()        
        train_loss+=loss.item() * images.size(0)
        tconf_matrix = confusion_matrix(predictions.cpu().numpy(), targets.cpu().numpy(), train_conf_matrix)
    

# test process
    with torch.no_grad():
        resnet.eval()
    
        for datas in validata_dataloader:
            images,targets=datas
            images = images.to(device)            
            targets=torch.as_tensor(targets)
            targets=targets.to(device)            
            outputs=resnet(images)            
            pre=add(outputs)            
            loss = loss_criterion(outputs, targets)
            ret,predictions= torch.max(outputs.data,1)           
            valid_loss+=loss.item() * images.size(0)
            vconf_matrix = confusion_matrix(predictions.cpu().numpy(), targets.cpu().numpy(), valid_conf_matrix)
    
    avg_train_loss = train_loss/train_dataset.__len__()   

    
    print(vconf_matrix)
    tconf_matrix=np.array(tconf_matrix)
    tcorrects=tconf_matrix.diagonal(offset=0)
    tper_kinds=tconf_matrix.sum(axis=0)
    avg_train_acc, tper_kind_acc=compute_accuracy(tcorrects, tper_kinds)

    avg_valid_loss = valid_loss/validata_dataset.__len__()
    
    vconf_matrix=np.array(vconf_matrix)
    vcorrects=vconf_matrix.diagonal(offset=0)
    vper_kinds=vconf_matrix.sum(axis=0)
    avg_valid_acc, vper_kind_acc=compute_accuracy(vcorrects, vper_kinds)

    history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
    tkind_acc.append(tper_kind_acc)
    vkind_acc.append(vper_kind_acc)
    
history = np.array(history)
print(history)
print(tkind_acc[-1])
print(vkind_acc[-1])
plt.plot(history[:,0:2])
plt.legend(['Tr Loss','Vali Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.savefig('/nfs/home/duanj/BYOLtask2/7class/results/fc_loss_curve_5600.png')

plt.figure()
plt.plot(history[:,2:4])
plt.legend(['Tr accuracy','Vali accuray'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.savefig('/nfs/home/duanj/BYOLtask2/7class/results/fc_accuracy_curve_5600.png')

torch.save(resnet.state_dict(),r'/nfs/home/duanj/BYOLtask2/7class/results/fc-net.pt')