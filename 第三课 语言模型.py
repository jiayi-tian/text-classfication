#!/usr/bin/env python
# coding: utf-8

# # 第三课  语言模型和文本分类

# In[4]:


import torchtext
from torchtext.vocab import Vectors
import torch
import numpy as np
import random

USE_CUDA=torch.cuda.is_available()

random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
if USE_CUDA:
    torch.cuda.manual_seed(53113)

BATCH_SIZE=32
EMBEDDING_SIZE=100
HIDDEN_SIZE=100
MAX_VOCAB_SIZE=50000


# In[5]:


TEXT=torchtext.data.Field(lower=True)
train,val,test=torchtext.datasets.LanguageModelingDataset.splits(path=".",
                train="text8.train.txt",validation="text8.dev.txt",test="text8.test.txt",text_field=TEXT)


# In[6]:


TEXT.build_vocab(train,max_size=MAX_VOCAB_SIZE)


# In[7]:


TEXT.vocab.itos[:10]


# In[8]:


device=torch.device("cuda" if USE_CUDA else "cpu")


# In[9]:


device


# In[10]:


train_iter,vla_iter,test_iter=torchtext.data.BPTTIterator.splits(
 (train,val,test),batch_size=BATCH_SIZE,device=device,bptt_len=50,repeat=False,shuffle=True)


# In[11]:


it=iter(train_iter)
batch=next(it)


# In[12]:


print(" ".join(TEXT.vocab.itos[i] for i in batch.text[:,0].data))
print()
print(" ".join(TEXT.vocab.itos[i] for i in batch.target[:,0].data))


# In[13]:


#for i in range(5):
#    batch=next(it)
#    print(i)
#    print(" ".join(TEXT.vocab.itos[i] for i in batch.text[:,0].data))
#    print()
#    print(" ".join(TEXT.vocab.itos[i] for i in batch.target[:,0].data))


# In[14]:


import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size):
        super(RNNModel,self).__init__()
        self.embed=nn.Embedding(vocab_size,embed_size)
        self.lstm=nn.LSTM(embed_size,hidden_size)
        self.linear=nn.Linear(hidden_size,vocab_size)
        self.hidden_size=hidden_size
    def forward(self,text,hidden):
        emb=self.embed(text)
        output,hidden=self.lstm(emb,hidden)
        out_vocab=self.linear(output.view(-1,output.shape[2]))
        out_vocab=out_vocab.view(output.size(0),output.size(1),out_vocab.size(-1))
        return out_vocab,hidden
    def init_hidden(self,bsz,requires_grad=True):
        weight=next(self.parameters())
        return (weight.new_zeros((1,bsz,self.hidden_size),requires_grad=True),
          weight.new_zeros((1,bsz,self.hidden_size),requires_grad=True))


# In[15]:


model=RNNModel(vocab_size=len(TEXT.vocab),
              embed_size=EMBEDDING_SIZE,
              hidden_size=HIDDEN_SIZE)
if USE_CUDA:
    model=model.to(device)


# In[16]:


# next(model.parameters())


# In[17]:


def repackage_hidden(h):
    if isinstance(h,torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# In[18]:


loss_fn=nn.CrossEntropyLoss()
learning_rate=0.001
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)


# In[19]:


VOCAB_SIZE=len(TEXT.vocab)


# In[20]:


def evaluate(model,data):
    model.eval()
    total_loss=0.
    total_count=0.
    it=iter(data)
    with torch.no_grad():
        hidden=model.init_hidden(BATCH_SIZE,requires_grad=False)
        for i,batch in enumerate(it):
            data,target=batch.text,batch.target
            hidden=repackage_hidden(hidden)
            output,hidden=model(data,hidden)
            loss=loss_fn(output.view(-1,VOCAB_SIZE),target.view(-1))
            total_loss=loss.item()*np.multiply(*data.size())
            total_count=np.multiply(*data.size())
    loss=total_loss/total_count
    model.train()
    return loss


# In[ ]:


NUM_EPOCHS=2
for epoch in range(NUM_EPOCHS):
    model.train()
    it=iter(train_iter)
    hidden=model.init_hidden(BATCH_SIZE)
    for i,batch in enumerate(it):
        data,target=batch.text,batch.target
        hidden=repackage_hidden(hidden)
        output,hidden=model(data,hidden)
        loss=loss_fn(output.view(-1,VOCAB_SIZE),target.view(-1))
        optimizer.zero_grad()
        loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters()，GRAD_CLIP) #???
        optimizer.step()
        if i%100==0:
            print("loss",loss.item())
        


# In[ ]:





# In[ ]:




