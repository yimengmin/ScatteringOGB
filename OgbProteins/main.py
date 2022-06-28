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
from torch_geometric.utils import to_scipy_sparse_matrix
import torch_geometric.transforms as T
from torch_geometric.utils import to_scipy_sparse_matrix
from utils import sparse_mx_to_torch_sparse_tensor
from layers import GC_withres
import torch.optim as optim
import numpy as np
from models import SCT_GAT_ogbproteins



dataset_name = "ogbn-proteins"

from ogb.nodeproppred import Evaluator #use to evalatute the accuracy
evaluator = Evaluator(dataset_name)
### use gcn
#from torch_geometric.nn import GCNConv, ChebConv  # noqa
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.003,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--l1', type=float, default=0.0,
                    help='Weight decay (L1 loss on parameters).')
parser.add_argument('--hid', type=int, default=8,
                    help='Number of hidden units.')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='Number of batch size')
parser.add_argument('--nheads', type=int, default=4,
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
parser.add_argument('--valid_cluster_number', type=int, default=5,
                    help='the number of sub-graphs for evaluation')
parser.add_argument('--cluster_number', type=int, default=10,
                    help='the number of sub-graphs for training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
#from torch_geometric.nn import GATConv
from torch.optim.lr_scheduler import MultiStepLR,StepLR
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = PygNodePropPredDataset(name = dataset_name)
evaluator = Evaluator(name=dataset_name)
split_idx = dataset.get_idx_split()
train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
data = dataset[0]
edge_index = data.edge_index.to(device)
adj = to_scipy_sparse_matrix(edge_index = data.edge_index)
adj = adj+ adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
n_node_feats, n_edge_feats, n_classes = 0, 8, 112
##https://github.com/classicsong/dgl/blob/gat_bot_ngnn/examples/pytorch/gat/bot_ngnn/gat_bot_ngnn.py
#https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/proteins/gnn.py
#https://github.com/lightaime/deep_gcns_torch/blob/master/examples/ogb/ogbn_proteins/main.py
degs = adj.sum(axis=0)
# Move edge features to node features.
degs = degs.getA1() #https://numpy.org/doc/stable/reference/generated/numpy.matrix.A1.html
features = torch.FloatTensor(degs)
features = features.to(device)
features = torch.unsqueeze(features, 1) #torch.Size([132534, 1])
adj = sparse_mx_to_torch_sparse_tensor(adj).cuda()

model = SCT_GAT_ogbproteins(1,args.hid,n_classes,dropout=args.dropout,nheads=args.nheads,smoo=args.smoo)
model = model.cuda()

#write a batch sampler
train_batch_size = args.batch_size 
from torch.utils.data import DataLoader, BatchSampler, RandomSampler


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

y_true = data.y.to(device)
def train(epoch,criterion,batch_or_not = False):
#    model.train()
#    optimizer.zero_grad()
#    out_feature = model(features,adj,split_idx['train'])
#    y_pred = out_feature
    if batch_or_not:
        model.train()
        optimizer.zero_grad()
        sampler = BatchSampler(RandomSampler(range(len(train_idx)),replacement=True),train_batch_size, drop_last=True)
        batch_itet = 0
        for indices in sampler:
            y_pred = model(features,adj,split_idx['train'][indices])
            loss_train =  criterion(y_pred[split_idx['train']][indices], y_true.squeeze(1)[split_idx['train']][indices].float())
            acc_train = evaluator.eval({'y_true': y_true[split_idx['train']][indices],'y_pred': y_pred[split_idx['train']][indices]})["rocauc"]
            print('Training Accuracy at %d iteration: %.5f with batch id: %d'%(epoch,acc_train,batch_itet))
            batch_itet = batch_itet + 1
            loss_train.backward()
            optimizer.step()
    else:
        model.train()
        optimizer.zero_grad()
        #full batch
        y_pred = model(features,adj,split_idx['train'])
        loss_train = criterion(y_pred[split_idx['train']], y_true.squeeze(1)[split_idx['train']].float())
        acc_train = evaluator.eval({'y_true': y_true[split_idx['train']],'y_pred': y_pred[split_idx['train']]})["rocauc"]
        print('Training Accuracy at %d iteration: %.5f'%(epoch,acc_train))
        loss_train.backward()
        optimizer.step()


def vali(epoch):
    model.eval()
    out_feature = model(features,adj)
    y_pred = out_feature
    acc_train = evaluator.eval({'y_true': y_true[split_idx['valid']],'y_pred': y_pred[split_idx['valid']]})['rocauc']
    print('Validation Accuracy at %d iteration: %.5f'%(epoch,acc_train))
    return acc_train


def test(epoch):
    model.eval()
    out_feature = model(features,adj)
    y_pred = out_feature
    acc_train = evaluator.eval({'y_true': y_true[split_idx['test']],'y_pred': y_pred[split_idx['test']]})['rocauc']
    print('Championship model\'s  accuracy in %d iterations: %.5f'%(epoch,acc_train))

highset_validation_accuracy = 0.1
for i in range(args.epochs+1):
    train(i,criterion=nn.BCEWithLogitsLoss())
    if i%5 == 0:
        validation_accuracy = vali(i)
        if validation_accuracy>highset_validation_accuracy:
            highset_validation_accuracy = validation_accuracy
            torch.save(model.state_dict(),'SAved_ogbmodels/championship_%s_model_dict.pt'%dataset_name)


model = SCT_GAT_ogbproteins(1,args.hid,n_classes,dropout=args.dropout,nheads=args.nheads,smoo=args.smoo)
model = model.cuda()
model.load_state_dict(torch.load('SAved_ogbmodels/championship_%s_model_dict.pt'%dataset_name))
test(args.epochs)
