#!/usr/bin/env python
# coding: utf-8

# # Dynamic Graph CNN

# Implementation of [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829) for classification
# 

# ## Data loading
# Let's get the dataset

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import torch
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
import time
from tqdm import tqdm_notebook

pre_transform = T.NormalizeScale()
transform = T.Compose([T.SamplePoints(1024),
                       T.RandomRotate(30), 
                       T.RandomScale((0.5,2)), 
                       ])
name = '40'

train_ds = ModelNet(root='./',
             train=True,
             name=name,
             pre_transform=pre_transform,
             transform=transform)

test_ds = ModelNet(root='./',
             train=True,
             name=name,
             pre_transform=pre_transform,
             transform = T.SamplePoints(1024 * 4))


# In[3]:


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device


# Now we have to define our dataloader, these guys will handle the thread queue to feed the GPU

# In[4]:


from torch_geometric.data import DataLoader

train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)

test_dl = DataLoader(test_ds, batch_size=8)


# ## Model
# 
# Define our architecture

# In[16]:


import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import EdgeConv, knn_graph, global_max_pool
from torch_geometric.nn import knn_graph
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d
import torch.nn.functional as F

class DynamicEdgeConv(gnn.EdgeConv):
    def __init__(self, k = 6, * args, ** kwargs):
        super().__init__( * args, ** kwargs)
        self.k = k

    def forward(self, pos, batch):
        edge_index = knn_graph(pos, self.k, batch, loop = False)
        return super().forward(pos, edge_index)

def fc_block(in_features, out_features):
    return Seq(
        Linear(in_features, out_features),
        BatchNorm1d(out_features),
        ReLU(inplace=True)
              )

class DGCNNClassification(nn.Module):
    def __init__(self, in_channels, n_classes, k = 20):
        super(DGCNNClassification, self).__init__()

        self.convs = nn.ModuleList([
            DynamicEdgeConv(
                k = k,
                nn = Sequential(
                    fc_block(in_channels * 2, 64),
                    fc_block(64, 64),
                    fc_block(64, 64),
                ),
                aggr = 'max'),
            DynamicEdgeConv(
                k = k,
                nn = fc_block(64 * 2, 128),          
                aggr = 'max')
        ])

        self.point_wise_features2higher_dim = fc_block(128, 1024)           

        self.tail = nn.Sequential(
            fc_block(1024, 512),   
            fc_block(512, 256), 
            fc_block(256, n_classes)      
        )

        self.k = k

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d): # very similar to resnet
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, batch):
        out = x
        for conv in self.convs:
            out = conv(out, batch)# it can be updated to use skipped connection
        out = self.point_wise_features2higher_dim(out)
        out = global_max_pool(out, batch)
        out = self.tail(out)
        
        return out


# In[17]:


class DGCNNClassificationSmall(DGCNNClassification):
  def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__(in_channels, n_classes, *args, **kwargs)
           
        self.point_wise_features2higher_dim = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU()
        )
        
        self.tail = nn.Sequential(
            nn.Linear(512, 256),
            BatchNorm1d(256),        
            nn.ReLU(),
            nn.Linear(256, n_classes),

        )
    


# ## Training

# In[18]:


now = time.time()
save_dir = './model-{}-{}.pt'.format(name, str(now).replace('.', '-'))
save_dir


# In[19]:


from torch.optim import Adam

model = DGCNNClassification(3,40).to(device)


optimizer = Adam(model.parameters(), 0.001)
criterion = nn.CrossEntropyLoss()

EPOCHS = 50


# In[20]:


def run(epochs, dl, train=True):
    bar = tqdm_notebook(range(epochs))
    last_acc = 0
    
    for epoch in bar:
        acc_tot = 0
        if (epoch + 1) % 10 == 0: 
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.2
        bbar = tqdm_notebook(dl, leave=False)
        for i, data in enumerate(bbar):
            start = time.time()
            if train: optimizer.zero_grad()
            data = data.to(device)
            out = model(data.pos, data.batch)
            preds = torch.argmax(out, dim=-1)
            acc = (data.y == preds).float().sum() / preds.shape[0]
            acc_v = acc.cpu().item()
            acc_tot += acc_v
            loss = criterion(out, data.y)
            if train:
                loss.backward()
                optimizer.step()
            bbar.set_description('[INFO] loss={:.2f} acc={:.2f}'.format(loss, acc_v))
        mean_acc = acc_tot / i
        if train:
            if mean_acc > last_acc:
                last_acc = mean_acc
                torch.save(model.state_dict(), save_dir)

        bar.set_description('[INFO] acc={:.3f} best={:.3f}'.format(mean_acc, last_acc))


# In[ ]:


run(50, train_dl, train=True)


# In[ ]:


model = DGCNNClassification(3,10).to(device)
model.load_state_dict(torch.load('./model-40-1558634785.3589494'))
model.eval()
run(1, test_dl, train=False)

