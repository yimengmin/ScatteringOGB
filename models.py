import torch.nn as nn
import torch 
import torch.nn.functional as F
from layers import GC_withres
from atten_sct_model import ScattterAttentionLayer

class SCT_ATTEN(nn.Module):
    def __init__(self, nfeat, hid, nclass, dropout):
        super(SCT_ATTEN, self).__init__()
        self.dropout = dropout
        self.gc1 = ScattterAttentionLayer(nfeat,hid,dropout=dropout)
#        self.gc2 = ScattterAttentionLayer(hid,hid,dropout=dropout)
        self.gc3 = ScattterAttentionLayer(hid,nclass,dropout=dropout)
    def forward(self,x,A_tilde,s1_sct,s2_sct,s3_sct):
        x = torch.nn.ReLU()(self.gc1(x,A_tilde,s1_sct,s2_sct,s3_sct))
#        x = torch.nn.ReLU()(self.gc2(x,A_tilde,s1_sct,s2_sct,s3_sct))
        x = torch.nn.ReLU()(self.gc3(x,A_tilde,s1_sct,s2_sct,s3_sct))
        return F.log_softmax(x, dim=1)
class SCT_GAT_ogbarxiv(nn.Module):
    def __init__(self, nfeat, hid, nclass, dropout, nheads,smoo):
        super(SCT_GAT_ogbarxiv, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.bn0 = nn.BatchNorm1d(nfeat)
        self.bn1 = nn.BatchNorm1d(hid)
        self.attentions = [ScattterAttentionLayer(nfeat, hid, dropout=dropout) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
#        self.out_att = ScattterAttentionLayer(hid * nheads, hid, dropout=dropout)
        self.gc11 = GC_withres(hid* nheads, nclass,smooth=smoo)
    def forward(self,x,adj):
        x = self.bn0(x)
        x = F.dropout(x, self.dropout, training=self.training)
#        x = torch.cat([torch.nn.ReLU()(att(x,A_tilde,s1_sct,s2_sct,s3_sct)) for att in self.attentions], dim=1)
        # save_attantion_list
        for i,att in enumerate(self.attentions):
            torch.save(att(x,adj)[1],'Attention_dir/ogbarxiv_attention_%d.pt'%i)
        x = torch.cat([torch.nn.ReLU()(self.bn1(att(x,adj)[0])) for att in self.attentions], dim=1)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc11(x,adj)

        # save_attantion_list

        return F.log_softmax(x, dim=1)
