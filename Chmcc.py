import os
import importlib
os.environ["DATA_FOLDER"] = "./"

import argparse

import time

import torch
import torch.utils.data
import torch.nn as nn

import random

from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, hamming_loss, f1_score
from sklearn.metrics import jaccard_score as jaccard_similarity_score

import numpy as np

from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve, roc_auc_score, auc

from torchvision.transforms import Resize
import cv2
from torchvision import models,datasets,transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.models.resnet import resnet50
import pandas as pd

def get_constr_out(x, R):
    """ Given the output of the neural network x returns the output of MCM given the hierarchy constraint expressed in the matrix R """
    c_out = x.double()
    c_out = c_out.unsqueeze(1)
    c_out = c_out.expand(len(x),R.shape[1], R.shape[1])
    R_batch = R.expand(len(x),R.shape[1], R.shape[1])
    final_out, _ = torch.max(R_batch*c_out.double(), dim = 2)
    return final_out


class ConstrainedFFNNModel(nn.Module):
    """ C-HMCNN(h) model - during training it returns the not-constrained output that is then passed to MCLoss """
    def __init__(self, num_class, R):
        super(ConstrainedFFNNModel, self).__init__()       
        
        self.R = R
        basemodel=resnet50()
        encoder= nn.Sequential(*(list(basemodel.children())[:-1]))
        self.f=encoder
        self.fc = nn.Sequential(
                  nn.Linear(2048, 256),
                  nn.ReLU(),
                  nn.Dropout(0.4),
                  nn.Linear(256,num_class))
        self.sigmoid = nn.Sigmoid()
        for param in self.f.parameters():
            param.requires_grad = False
    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        out=self.sigmoid(out)
        if self.training:
            constrained_out = out
        else:
            constrained_out = get_constr_out(out, self.R)
        return constrained_out

    
 # Load train and test set
species = ['perspective_view', 'non_perspective','side_view', 'Top_bottom', 'front_rear', 
            'left_side_view', 'right_side_view', 'top_view', 'bottom_view', 'front_view', 'rear_view']
species_to_idx = dict((c, i) for i, c in enumerate(species))

train_transform = transforms.Compose([transforms.RandomResizedCrop((224, 224)),                                      
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
        label = self.img_labels.iloc[idx, 1]
        targets=[0]*11
        if label in species_to_idx.keys():
            targets[species_to_idx[label]]=1
            if targets[5] or targets[6]==1:
                targets[1]=1
                targets[2]=1
            if targets[7] or targets[8]==1:
                targets[1]=1
                targets[3]=1
            if targets[9] or targets[10]==1:
                targets[1]=1
                targets[4]=1
            
            label=torch.as_tensor(targets)                            
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
train_dataset = CustomImageDataset(r"/nfs/home/duanj/database/datafile/7class/per7c_25perb.csv", r"/nfs/home/duanj/database/perspective",transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
validata_dataset=CustomImageDataset(r"/nfs/home/duanj/database/datafile/7class/test7c.csv", r"/nfs/home/duanj/database/perspective/test",transform=test_transform)
test_dataloader=DataLoader(validata_dataset, batch_size=64, shuffle=True, drop_last=True)
device = torch.device("cuda:0")
# Compute matrix of ancestors R
# Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is descendant of class j
R = [[1,0,0,0,0,0,0,0,0,0,0],
     [0,1,0,0,0,0,0,0,0,0,0],     
     [0,1,1,0,0,0,0,0,0,0,0],
     [0,1,0,1,0,0,0,0,0,0,0],
     [0,1,0,0,1,0,0,0,0,0,0],
     [0,0,1,0,0,1,0,0,0,0,0],
     [0,0,1,0,0,0,1,0,0,0,0],
     [0,0,0,1,0,0,0,1,0,0,0],
     [0,0,0,1,0,0,0,0,1,0,0],
     [0,0,0,0,1,0,0,0,0,1,0],
     [0,0,0,0,1,0,0,0,0,0,1]]
R = torch.tensor(R)
#Transpose to get the descendants for each node 
R = R.transpose(1, 0)
R = R.unsqueeze(0).to(device)

# Create the model
model = ConstrainedFFNNModel(num_class=11, R=R)
model.load_state_dict(torch.load('/nfs/home/duanj/SimCLRmodel/with_all_img/new_SimCLR-net.pt'),strict=False) 
model.to(device)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.BCELoss()
criterion=criterion.to(device)

for epoch in range(50):
    model.train()

    for i, (x, labels) in enumerate(train_dataloader):

        x = x.to(device)
        labels=labels.type(torch.DoubleTensor)
        labels = labels.to(device) 
        
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
        output = model(x.float())
        #MCLoss
        constr_output = get_constr_out(output, R)
        train_output = labels*output.double()
        train_output = get_constr_out(train_output, R)
        train_output = (1-labels)*constr_output.double() + labels*train_output

        loss = criterion(train_output, labels) 

        predicted = constr_output.data > 0.55

        # Total number of labels
        total_train = labels.size(0) * labels.size(1)
        # Total correct predictions
        correct_train = (predicted == labels.byte()).sum()

        loss.backward()
        optimizer.step()

    for i, (x,y) in enumerate(test_dataloader):

        model.eval()
                
        x = x.to(device)
        y = y.to(device)

        constrained_output = model(x.float())
        predicted = constrained_output.data > 0.55
        # Total number of labels
        total = y.size(0) * y.size(1)
        # Total correct predictions
        correct = (predicted == y.byte()).sum()

        #Move output and label back to cpu to be processed by sklearn
        predicted = predicted.to('cpu')
        cpu_constrained_output = constrained_output.to('cpu')
        y = y.to('cpu')

        if i == 0:
            predicted_test = predicted
            constr_test = cpu_constrained_output
            y_test = y
        else:
            predicted_test = torch.cat((predicted_test, predicted), dim=0)
            constr_test = torch.cat((constr_test, cpu_constrained_output), dim=0)
            y_test = torch.cat((y_test, y), dim =0)


    score = average_precision_score(y_test, constr_test.data, average='micro')

    f = open('hmccresults/'+'hmcc_7classes'+'.csv', 'a')
    f.write(str(epoch) + ',' + str(score) + '\n')
    f.close()




    

   

    