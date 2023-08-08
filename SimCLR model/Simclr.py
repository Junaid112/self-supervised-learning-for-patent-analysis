#SimCLR model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


# stage one ,unsupervised learning
class SimCLRStage1(nn.Module):
    def __init__(self, feature_dim=128):
        super(SimCLRStage1, self).__init__()

        #self.f = []
        #for name, module in resnet50().named_children():
            #if name == 'conv1':
                #module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            #if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                #self.f.append(module)
        basemodel=resnet50()
        encoder= nn.Sequential(*(list(basemodel.children())[:-1]))
        # encoder
        #self.f = nn.Sequential(*self.f)
        self.f = encoder
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


# stage two ,supervised learning
class SimCLRStage2(torch.nn.Module):
    def __init__(self, num_class):
        super(SimCLRStage2, self).__init__()
        # encoder
        self.f = SimCLRStage1().f
        # classifier
        self.fc = nn.Sequential(
                  nn.Linear(2048, 256),
                  nn.ReLU(),
                  nn.Dropout(0.4),
                  nn.Linear(256,num_class))

        for param in self.f.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


# class Loss(torch.nn.Module):
#     def __init__(self):
#         super(Loss,self).__init__()

#     def forward(self,out_1,out_2,batch_size,temperature=0.5):
#         # 分母 ：X.X.T，再去掉对角线值，分析结果一行，可以看成它与除了这行外的其他行都进行了点积运算（包括out_1和out_2）,
#         # 而每一行为一个batch的一个取值，即一个输入图像的特征表示，
#         # 因此，X.X.T，再去掉对角线值表示，每个输入图像的特征与其所有输出特征（包括out_1和out_2）的点积，用点积来衡量相似性
#         # 加上exp操作，该操作实际计算了分母
#         # [2*B, D]
#         out = torch.cat([out_1, out_2], dim=0)
#         # [2*B, 2*B]
#         sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
#         mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
#         # [2*B, 2*B-1]
#         sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

#         # 分子： *为对应位置相乘，也是点积
#         # compute loss
#         pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
#         # [2*B]
#         pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
#         return (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
            
    def forward(self, z_i, z_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        # z_i = F.normalize(emb_i, dim=1)
        # z_j = F.normalize(emb_j, dim=1)
 
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


if __name__=="__main__":
    for name, module in resnet50().named_children():
        print(name,module)
