import torch.nn as nn
import torch 
import torch.nn.functional as F
from layers import GC_withres
from atten_sct_model import ScattterAttentionLayer




class GNN_ogbproteins(nn.Module):
    def __init__(self, nfeat,nlayers,nhid, nclass, dropout,smoo):
        super(GNN_ogbproteins, self).__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.nhid = nhid
        self.convs.append(
                ScattterAttentionLayer(nfeat,nhid,dropout))
        for _ in range(nlayers - 2):
            self.convs.append(
                    ScattterAttentionLayer(nhid,nhid,dropout))
        self.dropout = dropout
        self.bn1 = nn.BatchNorm1d(nhid)
        self.gcres = GC_withres(nhid,nclass,smooth=smoo)
    def forward(self, x,adj):
#        for conv in self.convs[:-1]:
        for conv in self.convs:
            x = conv(x, adj)[0]
            x = self.bn1(x)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
#        x = self.convs[-1](x, adj)
        x = self.gcres(x,adj)
        return x




class SCT_GAT_ogbproteins(nn.Module):
    def __init__(self, nfeat, hid, nclass, dropout, nheads,smoo):
        super(SCT_GAT_ogbproteins, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.bn0 = nn.BatchNorm1d(nfeat)
        self.bn1 = nn.BatchNorm1d(hid)
        self.bn2 = nn.BatchNorm1d(hid*nheads)
        self.bn3 = nn.BatchNorm1d(hid)
        self.bn4 = nn.BatchNorm1d(nclass)
        self.attentions = [ScattterAttentionLayer(hid, hid, dropout=dropout) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.fc1 = nn.Linear(hid,nclass)
        self.gcres = GC_withres(hid* nheads,hid,smooth=smoo)

        self.mlp1 = nn.Sequential(nn.Linear(nfeat, 512),\
#                nn.BatchNorm1d(512),\
                nn.LeakyReLU(),\
                nn.Dropout(dropout),\
                nn.Linear(512,512),\
#                nn.BatchNorm1d(512),\
                nn.LeakyReLU(),\
                nn.Dropout(dropout),\
                nn.Linear(512, hid),\
#                nn.BatchNorm1d(hid),\
                nn.LeakyReLU()
#                nn.Dropout(dropout),\
#                nn.Linear(32, out_features),\
#                nn.BatchNorm1d(out_features),\
#                nn.LeakyReLU()
                )
    def forward(self,x,adj,data_idx = [1,2,3,4]):
        x = self.mlp1(x) 
#        x = self.bn0(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([torch.nn.ReLU()(att(x,adj,data_idx)[0]) for att in self.attentions], dim=1)
#        x = torch.cat([torch.nn.ReLU()(self.bn1(att(x,adj,data_idx)[0])) for att in self.attentions], dim=1)
#        x = self.bn2(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcres(x,adj)
#        x = self.bn3(x)
        x = nn.LeakyReLU()(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc1(x)
#        x = torch.sigmoid(x)
#        x = self.bn4(x)
#        x = self.mlp1(x)
#        x = nn.LeakyReLU()(x)
#        x = F.dropout(x, self.dropout, training=self.training)

        # save_attantion_list

        return x
