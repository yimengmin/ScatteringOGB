import torch.nn as nn
import torch 
import torch.nn.functional as F
from layers import GC_withres
from atten_sct_model import ScattterAttentionLayer

class SCT_GAT_ogbarxiv(nn.Module):
    def __init__(self, nfeat, hid, nclass, dropout, nheads,smoo):
        super(SCT_GAT_ogbarxiv, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.bn0 = nn.BatchNorm1d(nfeat)
        self.bn1 = nn.BatchNorm1d(hid)
        self.bn2 = nn.BatchNorm1d(hid*nheads)
        self.bn3 = nn.BatchNorm1d(hid)
        self.attentions = [ScattterAttentionLayer(nfeat, hid, dropout=dropout) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.fc1 = nn.Linear(hid,nclass)
        self.gcres = GC_withres(hid* nheads,hid,smooth=smoo)
    def forward(self,x,adj,data_idx = [1,2]):
        x = self.bn0(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([torch.nn.ReLU()(self.bn1(att(x,adj,data_idx)[0])) for att in self.attentions], dim=1)
        x = self.bn2(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcres(x,adj)
        x = self.bn3(x)
        x = nn.LeakyReLU()(x)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)
