#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
from torchtext import data

SEED=1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic=True

TEXT=data.Field(tokenize='spacy')
LABLE=data.LabelField(dtype=torch.float)


# In[8]:


from torchtext import datasets
train_data,test_data=datasets.IMDB.splits(TEXT,LABEL)


# In[11]:


print(f'Number of training examples:{len(train_data)}')
print(f'Number of testing examples:{len(test_data)}')


# In[12]:


print(vars(train_data.examples[0]))#vars???


# In[13]:


import random
train_data,valid_data=train_data.split(random_state=random.seed(SEED))


# In[14]:


print(f'Number of training examples:{len(train_data)}')
print(f'Number of validation examples:{len(valid_data)}')
print(f'Number of testing examples:{len(test_data)}')


# In[1]:


TEXT.build_vocab(train_data,max_size=2500,vectors="glove.6B.100d",unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)


# In[2]:


print(f"Unique tokens in TEXT vocabulary:{len(TEXT.vocab)}")
print(f"Unique tokens in LABEL vocabulary:{len(LABEL.vocab)}")


# In[3]:


print(TEXT.vocab.freqs.most_common(20))


# In[5]:


print(TEXT.vocab.itos[:10])
print(LABEL.vocab.stoi)


# In[6]:


BATCH_SIZE=64

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator,valid_iterator,test_itertor=data.BucketIterator.splits((train_data,valid_data,test_data),batch_size=BATCH_SIZE,device=device)


# In[7]:


batch=next(iter(valid_iterator))


# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
class WordAVGModel(nn.Module):
    def __init__(self,vocab_size,embedding_size,pad_idx,output_size):
        super(WordAVGModel,self).__init__()
        self.embed=nn.Embedding(vocab_size,embedding_size,padding_size=pad_idx)
        self.linear=nn.Linear(embedding_size,output_size)
    def forward(self,text):
        embedded=self.embed(text)
        embedded=embedded.permute(1,0,2)
        pooled=F.avg_pool2d(embedded,(embedded,shape[1],1)).squeeze()#???
        return self.linear(pooled)
        


# In[9]:


VOCAB_SIZE=len(TEXT.vocab)
EMBEDING_SIZE=embedding_size
PAD_IDX=TEXT.vocab.stoi(TEXT.pad_tokens)
OUTPUT_SIZE=outputsize
model=WordAVGModel(vocab_size=VOCAB_SIZE,enbedding_size=EMBEDING_SIZE,pad_idx=PAD_IDX,output_size=OUTPUT_SIZE)


# In[10]:


#以下划线结尾的function表示的都是inplace 的操作
#初始化模型
pretrained_embedding=TEXT.vocab.vectors
model.embed.weight.data.copy_(pretrained_embedding)
UNK_IDX=TEXT.vocab.stoi(TEXT.unk_token)
model.embed.weight.data[PAD_IDX]=torch.zeros(EMBEDDING_SZIE)
model.embed.weight.data[UNK_IDX]=torch.zeros(EMBEDDING_SZIE)


# In[ ]:


optimizer=torch.optim.Adam(model.parameters())
crit=nn.BCEWithLogitsLoss()

model=model.to(device)
crit=crit.to(device)


# In[13]:


def binary_accuracy(preds,y):
    rounded_preds=torch.round(torch.sigmoid(preds))
    correct=(round_preds==y).float()
    correct.sum()/len(correct)
    return acc


# In[15]:


def train(model,iterator,optimizer,crit):
    epoch_loss,epoch_acc=0.,0.
    model.train()
    total_len=0.
    for batch in iterator:
        preds=model(torch.text)
        loss=crit(pres,batch.label)
        acc=binary_accuracy(perds,batch.label)
        
        optimizer.zero.grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss+=loss.item()*len(batch.label)
        epoch_acc+=acc.item()*len(batch.label)#???
        total_len+=len(batch_label)
        
    return epoch_loss/total_len,epoch_acc/total_len


# In[1]:


def evaluate(model,iterator,optimizer,crit):
    epoch_loss,epoch_acc=0.,0.
    model.eval()
    total_len=0.
    for batch in iterator:
        preds=model(torch.text)
        loss=crit(pres,batch.label)
        acc=binary_accuracy(perds,batch.label)
        
        epoch_loss+=loss.item()*len(batch.label)
        epoch_acc+=acc.item()*len(batch.label)#???
        total_len+=len(batch_label)
        model.train()
    return epoch_loss/total_len,epoch_acc/total_len


# In[ ]:


N_EPOCHES=10
best_valid_acc=0.
for epoch in range(N_EPOCHES):
    train_loss,train_acc=train(model,iterator,optimizer,crit)
    valid_loss,valid_acc=evaluate(model,iterator,crit)

