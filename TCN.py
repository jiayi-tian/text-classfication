import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x): # Chomp1d(padding)
        return x[:, :, :-self.chomp_size].contiguous() # x为一个三维的tensor
        


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 里面是一个一维的卷积，外层是一个权重归一化
        self.chomp1 = Chomp1d(padding) # 0 padding
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 在输出的结果上又做一个一维的卷积，输入输出的维度相同 做这样的处理的意义？？？
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        # nn.Sequential 序列容器，将按照添加的顺序创建模型
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res) # 残差连接


class TemporalConvNet(nn.Module): 
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2): #卷积核 2 
    # TCN(args.emsize, n_words, num_chans, dropout=dropout, emb_dropout=emb_dropout, kernel_size=k_size, tied_weights=tied)
    # 
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels) # num_channels 存储的是每一层输出的维度
        for i in range(num_levels): 
            dilation_size = 2 ** i # 空洞卷积
            in_channels = num_inputs if i == 0 else num_channels[i-1] # num_channels 是一个列表,存储的是后面的层的输入维度
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)] 

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

#(batch_size,embedding_dim,seq_len) 在卷积神经网络中必须以这样的形式输入