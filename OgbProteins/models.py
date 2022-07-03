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


