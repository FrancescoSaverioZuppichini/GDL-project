#!/usr/bin/env python
# coding: utf-8

# # Dynamic Graph CNN

# ## Data loading
# Let's get the dataset

# In[15]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"


# In[1]:


import torch
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T

pre_transform = T.NormalizeScale()

train_ds = ModelNet(root='./',
             train=True,
             pre_transform=pre_transform)

test_ds = ModelNet(root='./',
             train=True,
             pre_transform=pre_transform)


# In[19]:


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
device


# Now we have to define our dataloader, these guys will handle the thread queue to feed the GPU

# In[3]:


from torch_geometric.data import DataLoader

train_dl = DataLoader(train_ds, batch_size=16)

test_dl = DataLoader(test_ds, batch_size=16)


# ## Model
# 
# Define our architecture

# In[4]:


import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import EdgeConv, knn_graph, global_max_pool
from torch_geometric.nn import knn_graph
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
import torch.nn.functional as F

class DynamicEdgeConv(gnn.EdgeConv):
    def __init__(self, k=6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def forward(self, pos, batch):
        edge_index = knn_graph(pos, self.k, batch, loop=False)
        return super().forward(pos, edge_index)

class DGCNNClassification(nn.Module):
  def __init__(self, in_channels, n_classes, k=20):
    super(DGCNNClassification, self).__init__()

    self.convs = nn.ModuleList([
        DynamicEdgeConv(
            k=k,
            nn=Sequential(
              Linear(in_channels * 2, 64),
              ReLU(),
              Linear(64, 64),
              ReLU(),
              Linear(64, 64),
              ReLU()
            ), 
            aggr='max'),
        DynamicEdgeConv(
            k=k,
            nn=Sequential(
              Linear(64 * 2, 128),
              ReLU()
          ), 
        aggr='max') 
    ])
    
    
    self.point_wise_features2higher_dim = nn.Sequential(
        nn.Linear(128, 512),
        nn.ReLU()
    )
    
    self.tail = nn.Sequential(
#         nn.Linear(1024, 512),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, n_classes),

    )
    
    self.k = k
    
  def forward(self, x, batch):
      out = x
      for conv in self.convs:
        out = conv(out, batch)
      
      out = self.point_wise_features2higher_dim(out)
      out = global_max_pool(out, batch)
      out = self.tail(out)
      
      return out


# ## Training

# In[7]:


from torch.optim import Adam
import time


model = DGCNNClassification(3,10).to(device)


optimizer = Adam(model.parameters(), 0.001)
criterion = nn.CrossEntropyLoss()

for data in train_dl:
    start = time.time()
    optimizer.zero_grad()
    data = data.to(device)
    print(data)
    out = model(data.pos, data.batch)
    loss = criterion(out, data.y)
    loss.backward()
    print('[INFO] loss={} elapsed={:.2f}'.format(loss.item(), time.time() - start))
    optimizer.step()


# ## Traversability

# In[ ]:


import numpy as np
x = np.ones((76, 76)) # this should be an image from somewhere
x = torch.tensor(x, dtype=torch.float).unsqueeze(-1)

edge_index, pos = grid(76,76)

data = Data(x=x, edge_index=edge_index, pos=pos.to(torch.float))
data


# In[ ]:


a = torch.tensor([1,2,3])
a.to(device)


# In[ ]:




