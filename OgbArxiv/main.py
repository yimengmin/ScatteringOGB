#import pickle 
#print(pickle.format_version)
#print(pickle.__version__)
from ogb.nodeproppred import Evaluator
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, add_self_loops
from ogb.nodeproppred import PygNodePropPredDataset
import time

import os.path as osp
import scipy.sparse as sp
import argparse
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
import torch.nn as nn
from torch_geometric.datasets import WikiCS
from torch_geometric.utils import to_scipy_sparse_matrix
import torch_geometric.transforms as T
from torch_geometric.utils import to_scipy_sparse_matrix
from utils import sparse_mx_to_torch_sparse_tensor
from layers import GC_withres
import torch.optim as optim
import numpy as np
from models import SCT_GAT_ogbarxiv



dataset_name = "ogbn-arxiv"

from ogb.nodeproppred import Evaluator #use to evalatute the accuracy
evaluator = Evaluator(dataset_name)
### use gcn
from layers import GC_withres
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--l1', type=float, default=0.0,
                    help='Weight decay (L1 loss on parameters).')
parser.add_argument('--hid', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--nheads', type=int, default=1,
                    help='Number of heads in attention mechism.')
parser.add_argument('--data_spilt', type=int, default=0)
parser.add_argument('--patience', type=int, default=200)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--normalization', type=str, default='AugNormAdj',
                    choices=['AugNormAdj'],
                    help='Normalization method for the adjacency matrix.')
parser.add_argument('--smoo', type=float, default=0.1,
                    help='Smooth for Res layer')
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    print('Traning on Cuda,yeah!')
    torch.cuda.manual_seed(args.seed)


#from torch_geometric.nn import GATConv
from torch.optim.lr_scheduler import MultiStepLR,StepLR
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Num of feat:1639
dataset = PygNodePropPredDataset(name=dataset_name)
split_idx = dataset.get_idx_split()


data = dataset[0]
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
x = data.x.to(device)
edge_index = data.edge_index.to(device)
# Num of feat:1639
adj = to_scipy_sparse_matrix(edge_index = data.edge_index)
adj = adj+ adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#print(type(adj)) #<class 'scipy.sparse.csr.csr_matrix'>
#A_tilde = normalize_adjacency_matrix(adj,sp.eye(adj.shape[0]))
#adj_p = normalizemx(adj)
#
features = data.x
features = torch.FloatTensor(np.array(features))
features = features.to(device)
adj = sparse_mx_to_torch_sparse_tensor(adj).cuda()
#from diffusion import GCN_diffusion,scattering_diffusion
#gcn_diffusion_list = GCN_diffusion(adj,3,features)
#print(gcn_diffusion_list[1].size())
#h_sct1,h_sct2,h_sct3 = scattering_diffusion(adj,features)
#print(h_sct1.size())
model = SCT_GAT_ogbarxiv(features.shape[1],args.hid,dataset.num_classes,dropout=args.dropout,nheads=args.nheads,smoo=args.smoo)
model = model.cuda()

#from torch.optim.lr_scheduler import StepLR
#optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
#scheduler = StepLR(optimizer, step_size=100, gamma=0.9)
#pred = out_feature[train_idx]


### y true
y_true = data.y.to(device)
#y_true = y_true[split_idx['train']]



def train(epoch):
    model.train()
    optimizer.zero_grad()
    out_feature = model(features,adj) 
#    out_feature = model(features,adj,split_idx['train']) 
    y_pred = out_feature.argmax(dim=-1, keepdim=True)
#    print(y_pred.size())
    loss_train = F.nll_loss(out_feature[split_idx['train']], y_true.squeeze(1)[split_idx['train']])
#    acc_train = accuracy(out_feature[split_idx['train']], y_true[split_idx['train']])
    acc_train = evaluator.eval({'y_true': y_true[split_idx['train']],'y_pred': y_pred[split_idx['train']],})['acc']
    print('Training Accuracy at %d iteration: %.5f'%(epoch,acc_train))
    loss_train.backward()
    optimizer.step()
#    scheduler.step()

def vali(epoch):
    model.eval()
    out_feature = model(features,adj)
    y_pred = out_feature.argmax(dim=-1, keepdim=True)
    acc_train = evaluator.eval({'y_true': y_true[split_idx['valid']],'y_pred': y_pred[split_idx['valid']],})['acc']
    print('Validation Accuracy at %d iteration: %.5f'%(epoch,acc_train))
    return acc_train


def test(epoch):
    model.eval()
    out_feature = model(features,adj)
    y_pred = out_feature.argmax(dim=-1, keepdim=True)
    acc_train = evaluator.eval({'y_true': y_true[split_idx['test']],'y_pred': y_pred[split_idx['test']],})['acc']
    print('Championship model\'s test  accuracy in %d iterations: %.5f'%(epoch,acc_train))


highset_validation_accuracy = 0.1
for i in range(args.epochs+1):
    train(i)
    if i%5 == 0:
        validation_accuracy = vali(i)
        if validation_accuracy>highset_validation_accuracy:
            highset_validation_accuracy = validation_accuracy
            torch.save(model.state_dict(),'SAved_ogbmodels/championship_%s_model_dict.pt'%dataset_name)
#        test(i)
#    if i%1000 ==0:
print('Championship Validation Accuracy: %.5f'%highset_validation_accuracy)
model = SCT_GAT_ogbarxiv(features.shape[1],args.hid,dataset.num_classes,dropout=args.dropout,nheads=args.nheads,smoo=args.smoo)
model = model.cuda()
model.load_state_dict(torch.load('SAved_ogbmodels/championship_%s_model_dict.pt'%dataset_name))
test(args.epochs)
