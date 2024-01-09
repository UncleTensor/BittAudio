# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# (developer): ETG development team
# Copyright © 2023 ETG
# Credit for NISQA evaluation model Gabriel Mittag, TU-Berlin

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# -*- coding: utf-8 -*-
import typing
from lib.subjective import SpeechToTextEvaluator
import bittensor as bt
import os
import multiprocessing
import copy
import math
import librosa as lb
import numpy as np
import pandas as pd; pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import lib
import pandas as pd


#%% Models
class NISQA(nn.Module):
    '''
    NISQA: The main speech quality model without speech quality dimension 
    estimation (MOS only). The module loads the submodules for framewise 
    modelling (e.g. CNN), time-dependency modelling (e.g. Self-Attention     
    or LSTM), and pooling (e.g. max-pooling or attention-pooling)                                                  
    '''       
    def __init__(self,
            ms_seg_length=15,
            ms_n_mels=48,
            
            cnn_model='adapt',
            cnn_c_out_1=16, 
            cnn_c_out_2=32,
            cnn_c_out_3=64,
            cnn_kernel_size=3, 
            cnn_dropout=0.2,
            cnn_pool_1=[24,7],
            cnn_pool_2=[12,5],
            cnn_pool_3=[6,3],  
            cnn_fc_out_h=None,     
              
            td='self_att',
            td_sa_d_model=64,
            td_sa_nhead=1,
            td_sa_pos_enc=None,
            td_sa_num_layers=2,
            td_sa_h=64,
            td_sa_dropout=0.1,
            td_lstm_h=128,
            td_lstm_num_layers=1,
            td_lstm_dropout=0,
            td_lstm_bidirectional=True,
            
            td_2='skip',
            td_2_sa_d_model=None,
            td_2_sa_nhead=None,
            td_2_sa_pos_enc=None,
            td_2_sa_num_layers=None,
            td_2_sa_h=None,
            td_2_sa_dropout=None,
            td_2_lstm_h=None,
            td_2_lstm_num_layers=None,
            td_2_lstm_dropout=None,
            td_2_lstm_bidirectional=None,            
            
            pool='att',
            pool_att_h=128,
            pool_att_dropout=0.1,
               
            ):
        super().__init__()
    
        self.name = 'NISQA'
        
        self.cnn = Framewise(
            cnn_model,
            ms_seg_length=ms_seg_length,
            ms_n_mels=ms_n_mels,
            c_out_1=cnn_c_out_1, 
            c_out_2=cnn_c_out_2,
            c_out_3=cnn_c_out_3,
            kernel_size=cnn_kernel_size, 
            dropout=cnn_dropout,
            pool_1=cnn_pool_1,
            pool_2=cnn_pool_2,
            pool_3=cnn_pool_3,
            fc_out_h=cnn_fc_out_h,   
            )        
        
        self.time_dependency = TimeDependency(
            input_size=self.cnn.model.fan_out,
            td=td,
            sa_d_model=td_sa_d_model,
            sa_nhead=td_sa_nhead,
            sa_pos_enc=td_sa_pos_enc,
            sa_num_layers=td_sa_num_layers,
            sa_h=td_sa_h,
            sa_dropout=td_sa_dropout,
            lstm_h=td_lstm_h,
            lstm_num_layers=td_lstm_num_layers,
            lstm_dropout=td_lstm_dropout,
            lstm_bidirectional=td_lstm_bidirectional
            )
        
        self.time_dependency_2 = TimeDependency(
            input_size=self.time_dependency.fan_out,
            td=td_2,
            sa_d_model=td_2_sa_d_model,
            sa_nhead=td_2_sa_nhead,
            sa_pos_enc=td_2_sa_pos_enc,
            sa_num_layers=td_2_sa_num_layers,
            sa_h=td_2_sa_h,
            sa_dropout=td_2_sa_dropout,
            lstm_h=td_2_lstm_h,
            lstm_num_layers=td_2_lstm_num_layers,
            lstm_dropout=td_2_lstm_dropout,
            lstm_bidirectional=td_2_lstm_bidirectional
            )        
        
        self.pool = Pooling(
            self.time_dependency_2.fan_out,
            output_size=1,
            pool=pool,
            att_h=pool_att_h,
            att_dropout=pool_att_dropout,
            )                 

    def forward(self, x, n_wins):
        x = self.cnn(x, n_wins)
        x, n_wins = self.time_dependency(x, n_wins)
        x, n_wins = self.time_dependency_2(x, n_wins)
        x = self.pool(x, n_wins)
        return x
    


class NISQA_DIM(nn.Module):
    '''
    NISQA_DIM: The main speech quality model with speech quality dimension 
    estimation (MOS, Noisiness, Coloration, Discontinuity, and Loudness).
    The module loads the submodules for framewise modelling (e.g. CNN),
    time-dependency modelling (e.g. Self-Attention or LSTM), and pooling 
    (e.g. max-pooling or attention-pooling)                                                  
    '''         
    def __init__(self,
            ms_seg_length=15,
            ms_n_mels=48,
            
            cnn_model='adapt',
            cnn_c_out_1=16, 
            cnn_c_out_2=32,
            cnn_c_out_3=64,
            cnn_kernel_size=3, 
            cnn_dropout=0.2,
            cnn_pool_1=[24,7],
            cnn_pool_2=[12,5],
            cnn_pool_3=[6,3],  
            cnn_fc_out_h=None,     
              
            td='self_att',
            td_sa_d_model=64,
            td_sa_nhead=1,
            td_sa_pos_enc=None,
            td_sa_num_layers=2,
            td_sa_h=64,
            td_sa_dropout=0.1,
            td_lstm_h=128,
            td_lstm_num_layers=1,
            td_lstm_dropout=0,
            td_lstm_bidirectional=True,
            
            td_2='skip',
            td_2_sa_d_model=None,
            td_2_sa_nhead=None,
            td_2_sa_pos_enc=None,
            td_2_sa_num_layers=None,
            td_2_sa_h=None,
            td_2_sa_dropout=None,
            td_2_lstm_h=None,
            td_2_lstm_num_layers=None,
            td_2_lstm_dropout=None,
            td_2_lstm_bidirectional=None,               

            pool='att',
            pool_att_h=128,
            pool_att_dropout=0.1,
            
            ):
        super().__init__()
    
        self.name = 'NISQA_DIM'
        self.dev =torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.cnn = Framewise(
            cnn_model,
            ms_seg_length=ms_seg_length,
            ms_n_mels=ms_n_mels,
            c_out_1=cnn_c_out_1, 
            c_out_2=cnn_c_out_2,
            c_out_3=cnn_c_out_3,
            kernel_size=cnn_kernel_size, 
            dropout=cnn_dropout,
            pool_1=cnn_pool_1,
            pool_2=cnn_pool_2,
            pool_3=cnn_pool_3,
            fc_out_h=cnn_fc_out_h,   
            )

   
        
        self.time_dependency = TimeDependency(
            input_size=self.cnn.model.fan_out,
            td=td,
            sa_d_model=td_sa_d_model,
            sa_nhead=td_sa_nhead,
            sa_pos_enc=td_sa_pos_enc,
            sa_num_layers=td_sa_num_layers,
            sa_h=td_sa_h,
            sa_dropout=td_sa_dropout,
            lstm_h=td_lstm_h,
            lstm_num_layers=td_lstm_num_layers,
            lstm_dropout=td_lstm_dropout,
            lstm_bidirectional=td_lstm_bidirectional
            )

        self.time_dependency_2 = TimeDependency(
            input_size=self.time_dependency.fan_out,
            td=td_2,
            sa_d_model=td_2_sa_d_model,
            sa_nhead=td_2_sa_nhead,
            sa_pos_enc=td_2_sa_pos_enc,
            sa_num_layers=td_2_sa_num_layers,
            sa_h=td_2_sa_h,
            sa_dropout=td_2_sa_dropout,
            lstm_h=td_2_lstm_h,
            lstm_num_layers=td_2_lstm_num_layers,
            lstm_dropout=td_2_lstm_dropout,
            lstm_bidirectional=td_2_lstm_bidirectional
            )     

        pool = Pooling(
            self.time_dependency.fan_out,
            output_size=1,
            pool=pool,
            att_h=pool_att_h,
            att_dropout=pool_att_dropout,
            )         
        
        self.pool_layers = self._get_clones(pool, 5)
        self.to(self.dev)
        
    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])        

    def forward(self, x, n_wins):
        
        x = self.cnn(x, n_wins)
        x, n_wins = self.time_dependency(x, n_wins)
        x, n_wins = self.time_dependency_2(x, n_wins)
        out = [mod(x, n_wins) for mod in self.pool_layers]
        out = torch.cat(out, dim=1)

        return out


    
class NISQA_DE(nn.Module):
    '''
    NISQA: The main speech quality model for double-ended prediction.
    The module loads the submodules for framewise modelling (e.g. CNN), 
    time-dependency modelling (e.g. Self-Attention or LSTM), time-alignment, 
    feature fusion and pooling (e.g. max-pooling or attention-pooling)                                                  
    '''         
    def __init__(self,
            ms_seg_length=15,
            ms_n_mels=48,
            
            cnn_model='adapt',
            cnn_c_out_1=16, 
            cnn_c_out_2=32,
            cnn_c_out_3=64,
            cnn_kernel_size=3, 
            cnn_dropout=0.2,
            cnn_pool_1=[24,7],
            cnn_pool_2=[12,5],
            cnn_pool_3=[6,3],  
            cnn_fc_out_h=None,     
              
            td='self_att',
            td_sa_d_model=64,
            td_sa_nhead=1,
            td_sa_pos_enc=None,
            td_sa_num_layers=2,
            td_sa_h=64,
            td_sa_dropout=0.1,
            td_lstm_h=128,
            td_lstm_num_layers=1,
            td_lstm_dropout=0,
            td_lstm_bidirectional=True,
            
            td_2='skip',
            td_2_sa_d_model=None,
            td_2_sa_nhead=None,
            td_2_sa_pos_enc=None,
            td_2_sa_num_layers=None,
            td_2_sa_h=None,
            td_2_sa_dropout=None,
            td_2_lstm_h=None,
            td_2_lstm_num_layers=None,
            td_2_lstm_dropout=None,
            td_2_lstm_bidirectional=None,               
            
            pool='att',
            pool_att_h=128,
            pool_att_dropout=0.1,
            
            de_align = 'dot',
            de_align_apply = 'hard',
            de_fuse_dim = None,
            de_fuse = True,         
               
            ):
        
        super().__init__()
    
        self.name = 'NISQA_DE'
        
        self.cnn = Framewise(
            cnn_model,
            ms_seg_length=ms_seg_length,
            ms_n_mels=ms_n_mels,
            c_out_1=cnn_c_out_1, 
            c_out_2=cnn_c_out_2,
            c_out_3=cnn_c_out_3,
            kernel_size=cnn_kernel_size, 
            dropout=cnn_dropout,
            pool_1=cnn_pool_1,
            pool_2=cnn_pool_2,
            pool_3=cnn_pool_3,
            fc_out_h=cnn_fc_out_h,   
            )        
        
        self.time_dependency = TimeDependency(
            input_size=self.cnn.model.fan_out,
            td=td,
            sa_d_model=td_sa_d_model,
            sa_nhead=td_sa_nhead,
            sa_pos_enc=td_sa_pos_enc,
            sa_num_layers=td_sa_num_layers,
            sa_h=td_sa_h,
            sa_dropout=td_sa_dropout,
            lstm_h=td_lstm_h,
            lstm_num_layers=td_lstm_num_layers,
            lstm_dropout=td_lstm_dropout,
            lstm_bidirectional=td_lstm_bidirectional
            )
        
        self.align = Alignment(
             de_align, 
             de_align_apply,
             q_dim=self.time_dependency.fan_out,
             y_dim=self.time_dependency.fan_out,
            )
                
        self.fuse = Fusion(
            in_feat=self.time_dependency.fan_out,
            fuse_dim=de_fuse_dim, 
            fuse=de_fuse,
            )             
        
        self.time_dependency_2 = TimeDependency(
            input_size=self.fuse.fan_out,
            td=td_2,
            sa_d_model=td_2_sa_d_model,
            sa_nhead=td_2_sa_nhead,
            sa_pos_enc=td_2_sa_pos_enc,
            sa_num_layers=td_2_sa_num_layers,
            sa_h=td_2_sa_h,
            sa_dropout=td_2_sa_dropout,
            lstm_h=td_2_lstm_h,
            lstm_num_layers=td_2_lstm_num_layers,
            lstm_dropout=td_2_lstm_dropout,
            lstm_bidirectional=td_2_lstm_bidirectional
            )                
        
        self.pool = Pooling(
            self.time_dependency_2.fan_out,
            output_size=1,
            pool=pool,
            att_h=pool_att_h,
            att_dropout=pool_att_dropout,
            )                 
        
    def _split_ref_deg(self, x, n_wins):
        (x, y) = torch.chunk(x, 2, dim=2)        
        (n_wins_x, n_wins_y) = torch.chunk(n_wins, 2, dim=1)
        n_wins_x = n_wins_x.view(-1)
        n_wins_y = n_wins_y.view(-1)       
        return x, y, n_wins_x, n_wins_y 

    def forward(self, x, n_wins):
        
        x, y, n_wins_x, n_wins_y = self._split_ref_deg(x, n_wins)
        
        x = self.cnn(x, n_wins_x)
        y = self.cnn(y, n_wins_y)
        
        x, n_wins_x = self.time_dependency(x, n_wins_x)
        y, n_wins_y = self.time_dependency(y, n_wins_y)
        
        y = self.align(x, y, n_wins_y)
        
        x = self.fuse(x, y)
        
        x, n_wins_x = self.time_dependency_2(x, n_wins_x)
        
        x = self.pool(x, n_wins_x)
        
        return x
    
    
#%% Framewise
class Framewise(nn.Module):
    '''
    Framewise: The main framewise module. It loads either a CNN or feed-forward
    network for framewise modelling of the Mel-spec segments. This module can
    also be skipped by loading the SkipCNN module. There are two CNN modules
    available. AdaptCNN with adaptive maxpooling and the StandardCNN module.
    However, they could also be replaced with new modules, such as PyTorch 
    implementations of ResNet or Alexnet.                                                 
    '''         
    def __init__(
        self, 
        cnn_model,
        ms_seg_length=15,
        ms_n_mels=48,
        c_out_1=16, 
        c_out_2=32,
        c_out_3=64,
        kernel_size=3, 
        dropout=0.2,
        pool_1=[24,7],
        pool_2=[12,5],
        pool_3=[6,3],
        fc_out_h=None,        
        ):
        super().__init__()
        
        if cnn_model=='adapt':
            self.model = AdaptCNN(
                input_channels=1,
                c_out_1=c_out_1, 
                c_out_2=c_out_2,
                c_out_3=c_out_3,
                kernel_size=kernel_size, 
                dropout=dropout,
                pool_1=pool_1,
                pool_2=pool_2,
                pool_3=pool_3,
                fc_out_h=fc_out_h,
                )
        elif cnn_model=='standard':
            assert ms_n_mels == 48, "ms_n_mels is {} and should be 48, use adaptive model or change ms_n_mels".format(ms_n_mels)
            assert ms_seg_length == 15, "ms_seg_len is {} should be 15, use adaptive model or change ms_seg_len".format(ms_seg_length)
            assert ((kernel_size == 3) or (kernel_size == (3,3))), "cnn_kernel_size is {} should be 3, use adaptive model or change cnn_kernel_size".format(kernel_size)
            self.model = StandardCNN(
                input_channels=1,
                c_out_1=c_out_1, 
                c_out_2=c_out_2,
                c_out_3=c_out_3,
                kernel_size=kernel_size, 
                dropout=dropout,
                fc_out_h=fc_out_h,
                )       
        elif cnn_model=='dff':
            self.model = DFF(ms_seg_length, ms_n_mels, dropout, fc_out_h)            
        elif (cnn_model is None) or (cnn_model=='skip'):
            self.model = SkipCNN(ms_seg_length, ms_n_mels, fc_out_h)
        else:
            raise NotImplementedError('Framwise model not available')                        
        
    def forward(self, x, n_wins):
        (bs, length, channels, height, width) = x.shape
        x_packed = pack_padded_sequence(
                x,
                n_wins.cpu(),
                batch_first=True,
                enforce_sorted=False
                )     
        x = self.model(x_packed.data) 
        x = x_packed._replace(data=x)                
        x, _ = pad_packed_sequence(
            x, 
            batch_first=True, 
            padding_value=0.0,
            total_length=n_wins.max())
        return x    

class SkipCNN(nn.Module):
    '''
    SkipCNN: Can be used to skip the framewise modelling stage and directly
    apply an LSTM or Self-Attention network.                                              
    '''        
    def __init__(
        self, 
        cnn_seg_length,
        ms_n_mels,
        fc_out_h
        ):
        super().__init__()

        self.name = 'SkipCNN'
        self.cnn_seg_length = cnn_seg_length
        self.ms_n_mels = ms_n_mels
        self.fan_in = cnn_seg_length*ms_n_mels
        self.bn = nn.BatchNorm2d( 1 )
        
        if fc_out_h is not None:
            self.linear = nn.Linear(self.fan_in, fc_out_h)
            self.fan_out = fc_out_h
        else:
            self.linear = nn.Identity()
            self.fan_out = self.fan_in
        
    def forward(self, x):
        x = self.bn(x)
        x = x.view(-1, self.fan_in)
        x = self.linear(x)
        return x    
    
class DFF(nn.Module):
    '''
    DFF: Deep Feed-Forward network that was used as baseline framwise model as
    comparision to the CNN.
    '''        
    def __init__(self, 
                 cnn_seg_length,
                 ms_n_mels,
                 dropout,
                 fc_out_h=4096,
                 ):
        super().__init__()
        self.name = 'DFF'

        self.dropout_rate = dropout
        self.fc_out_h = fc_out_h
        self.fan_out = fc_out_h
        
        self.cnn_seg_length = cnn_seg_length
        self.ms_n_mels = ms_n_mels
        self.fan_in = cnn_seg_length*ms_n_mels
        
        self.lin1 = nn.Linear(self.fan_in, self.fc_out_h)
        self.lin2 = nn.Linear(self.fc_out_h, self.fc_out_h)
        self.lin3 = nn.Linear(self.fc_out_h, self.fc_out_h)
        self.lin4 = nn.Linear(self.fc_out_h, self.fc_out_h)
        
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm1d( self.fc_out_h )
        self.bn3 = nn.BatchNorm1d( self.fc_out_h )
        self.bn4 = nn.BatchNorm1d( self.fc_out_h )
        self.bn5 = nn.BatchNorm1d( self.fc_out_h )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        
        x = self.bn1(x)
        x = x.view(-1, self.fan_in)
        
        x = F.relu( self.bn2( self.lin1(x) ) )
        x = self.dropout(x)
        x = F.relu( self.bn3( self.lin2(x) ) )
        x = self.dropout(x)
        x = F.relu( self.bn4( self.lin3(x) ) )
        x = self.dropout(x)
        x = F.relu( self.bn5( self.lin4(x) ) )
                
        return x
        
    
class AdaptCNN(nn.Module):
    '''
    AdaptCNN: CNN with adaptive maxpooling that can be used as framewise model.
    Overall, it has six convolutional layers. This CNN module is more flexible
    than the StandardCNN that requires a fixed input dimension of 48x15.
    '''            
    def __init__(self, 
                 input_channels,
                 c_out_1, 
                 c_out_2,
                 c_out_3,
                 kernel_size, 
                 dropout,
                 pool_1,
                 pool_2,
                 pool_3,
                 fc_out_h=20,
                 ):
        super().__init__()
        self.name = 'CNN_adapt'

        self.input_channels = input_channels
        self.c_out_1 = c_out_1
        self.c_out_2 = c_out_2
        self.c_out_3 = c_out_3
        self.kernel_size = kernel_size
        self.pool_1 = pool_1
        self.pool_2 = pool_2
        self.pool_3 = pool_3
        self.dropout_rate = dropout
        self.fc_out_h = fc_out_h

        self.dropout = nn.Dropout2d(p=self.dropout_rate)
        
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)
            
        # Set kernel width of last conv layer to last pool width to 
        # downsample width to one.
        self.kernel_size_last = (self.kernel_size[0], self.pool_3[1])
            
        # kernel_size[1]=1 can be used for seg_length=1 -> corresponds to 
        # 1D conv layer, no width padding needed.
        if self.kernel_size[1] == 1:
            self.cnn_pad = (1,0)
        else:
            self.cnn_pad = (1,1)   
            
        self.conv1 = nn.Conv2d(
                self.input_channels,
                self.c_out_1,
                self.kernel_size,
                padding = self.cnn_pad)

        self.bn1 = nn.BatchNorm2d( self.conv1.out_channels )

        self.conv2 = nn.Conv2d(
                self.conv1.out_channels,
                self.c_out_2,
                self.kernel_size,
                padding = self.cnn_pad)

        self.bn2 = nn.BatchNorm2d( self.conv2.out_channels )

        self.conv3 = nn.Conv2d(
                self.conv2.out_channels,
                self.c_out_3,
                self.kernel_size,
                padding = self.cnn_pad)

        self.bn3 = nn.BatchNorm2d( self.conv3.out_channels )

        self.conv4 = nn.Conv2d(
                self.conv3.out_channels,
                self.c_out_3,
                self.kernel_size,
                padding = self.cnn_pad)

        self.bn4 = nn.BatchNorm2d( self.conv4.out_channels )

        self.conv5 = nn.Conv2d(
                self.conv4.out_channels,
                self.c_out_3,
                self.kernel_size,
                padding = self.cnn_pad)

        self.bn5 = nn.BatchNorm2d( self.conv5.out_channels )

        self.conv6 = nn.Conv2d(
                self.conv5.out_channels,
                self.c_out_3,
                self.kernel_size_last,
                padding = (1,0))

        self.bn6 = nn.BatchNorm2d( self.conv6.out_channels )
        
        if self.fc_out_h:
            self.fc = nn.Linear(self.conv6.out_channels * self.pool_3[0], self.fc_out_h)
            self.fan_out = self.fc_out_h
        else:
            self.fan_out = (self.conv6.out_channels * self.pool_3[0])

    def forward(self, x):
        
        x = F.relu( self.bn1( self.conv1(x) ) )
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_1))

        x = F.relu( self.bn2( self.conv2(x) ) )
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_2))
        
        x = self.dropout(x)
        x = F.relu( self.bn3( self.conv3(x) ) )
        x = self.dropout(x)
        x = F.relu( self.bn4( self.conv4(x) ) )
        x = F.adaptive_max_pool2d(x, output_size=(self.pool_3))

        x = self.dropout(x)
        x = F.relu( self.bn5( self.conv5(x) ) )
        x = self.dropout(x)
        x = F.relu( self.bn6( self.conv6(x) ) )
        x = x.view(-1, self.conv6.out_channels * self.pool_3[0])
        
        if self.fc_out_h:
            x = self.fc( x ) 
        return x

class StandardCNN(nn.Module):
    '''
    StandardCNN: CNN with fixed maxpooling that can be used as framewise model.
    Overall, it has six convolutional layers. This CNN module requires a fixed
    input dimension of 48x15.
    '''           
    def __init__(
        self, 
        input_channels, 
        c_out_1, 
        c_out_2, 
        c_out_3, 
        kernel_size, 
        dropout, 
        fc_out_h=None
        ):
        super().__init__()

        self.name = 'CNN_standard'

        self.input_channels = input_channels
        self.c_out_1 = c_out_1
        self.c_out_2 = c_out_2
        self.c_out_3 = c_out_3
        self.kernel_size = kernel_size
        self.pool_size = 2
        self.dropout_rate = dropout
        self.fc_out_h = fc_out_h

        self.output_width = 2 # input width 15 pooled 3 times
        self.output_height = 6 # input height 48 pooled 3 times

        self.dropout = nn.Dropout2d(p=self.dropout_rate)

        self.pool_first = nn.MaxPool2d(
                self.pool_size,
                stride = self.pool_size,
                padding = (0,1))

        self.pool = nn.MaxPool2d(
                self.pool_size,
                stride = self.pool_size,
                padding = 0)

        self.conv1 = nn.Conv2d(
                self.input_channels,
                self.c_out_1,
                self.kernel_size,
                padding = 1)

        self.bn1 = nn.BatchNorm2d( self.conv1.out_channels )

        self.conv2 = nn.Conv2d(
                self.conv1.out_channels,
                self.c_out_2,
                self.kernel_size,
                padding = 1)

        self.bn2 = nn.BatchNorm2d( self.conv2.out_channels )


        self.conv3 = nn.Conv2d(
                self.conv2.out_channels,
                self.c_out_3,
                self.kernel_size,
                padding = 1)

        self.bn3 = nn.BatchNorm2d( self.conv3.out_channels )

        self.conv4 = nn.Conv2d(
                self.conv3.out_channels,
                self.c_out_3,
                self.kernel_size,
                padding = 1)

        self.bn4 = nn.BatchNorm2d( self.conv4.out_channels )

        self.conv5 = nn.Conv2d(
                self.conv4.out_channels,
                self.c_out_3,
                self.kernel_size,
                padding = 1)

        self.bn5 = nn.BatchNorm2d( self.conv5.out_channels )

        self.conv6 = nn.Conv2d(
                self.conv5.out_channels,
                self.c_out_3,
                self.kernel_size,
                padding = 1)
        
        self.bn6 = nn.BatchNorm2d( self.conv6.out_channels )

        if self.fc_out_h:
            self.fc_out = nn.Linear(self.conv6.out_channels * self.output_height * self.output_width, self.fc_out_h)
            self.fan_out = self.fc_out_h
        else:
            self.fan_out = (self.conv6.out_channels * self.output_height * self.output_width)

    def forward(self, x):
        
        x = F.relu( self.bn1( self.conv1(x) ) )
        x = self.pool_first( x )

        x = F.relu( self.bn2( self.conv2(x) ) )
        x = self.pool( x )

        x = self.dropout(x)
        x = F.relu( self.bn3( self.conv3(x) ) )
        x = self.dropout(x)
        x = F.relu( self.bn4( self.conv4(x) ) )
        x = self.pool( x )
        
        x = self.dropout(x)
        x = F.relu( self.bn5( self.conv5(x) ) )
        x = self.dropout(x)
        
        x = F.relu( self.bn6( self.conv6(x) ) )
                
        x = x.view(-1, self.conv6.out_channels * self.output_height * self.output_width) 
                            
        if self.fc_out_h:
            x = self.fc_out( x )

        return x
    
#%% Time Dependency
class TimeDependency(nn.Module):
    '''
    TimeDependency: The main time-dependency module. It loads either an LSTM 
    or self-attention network for time-dependency modelling of the framewise 
    features. This module can also be skipped.                                              
    '''          
    def __init__(self,
                 input_size,
                 td='self_att',
                 sa_d_model=512,
                 sa_nhead=8,
                 sa_pos_enc=None,
                 sa_num_layers=6,
                 sa_h=2048,
                 sa_dropout=0.1,
                 lstm_h=128,
                 lstm_num_layers=1,
                 lstm_dropout=0,
                 lstm_bidirectional=True,
                 ):
        super().__init__()
        
        if td=='self_att':
            self.model = SelfAttention(
                input_size=input_size,
                d_model=sa_d_model,
                nhead=sa_nhead,
                pos_enc=sa_pos_enc,
                num_layers=sa_num_layers,
                sa_h=sa_h,
                dropout=sa_dropout,
                activation="relu"
                )
            self.fan_out = sa_d_model
            
        elif td=='lstm':
            self.model = LSTM(
                 input_size,
                 lstm_h=lstm_h,
                 num_layers=lstm_num_layers,
                 dropout=lstm_dropout,
                 bidirectional=lstm_bidirectional,
                 )  
            self.fan_out = self.model.fan_out
            
        elif (td is None) or (td=='skip'):
            self.model = self._skip
            self.fan_out = input_size
        else:
            raise NotImplementedError('Time dependency option not available')    
            
    def _skip(self, x, n_wins):
        return x, n_wins

    def forward(self, x, n_wins):
        x, n_wins = self.model(x, n_wins)
        return x, n_wins
                
class LSTM(nn.Module):
    '''
    LSTM: The main LSTM module that can be used as a time-dependency model.                                            
    '''           
    def __init__(self,
                 input_size,
                 lstm_h=128,
                 num_layers=1,
                 dropout=0.1,
                 bidirectional=True
                 ):
        super().__init__()
        
        self.lstm = nn.LSTM(
                input_size = input_size,
                hidden_size = lstm_h,
                num_layers = num_layers,
                dropout = dropout,
                batch_first = True,
                bidirectional = bidirectional
                )      
            
        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1                 
        self.fan_out = num_directions*lstm_h

    def forward(self, x, n_wins):
        
        x = pack_padded_sequence(
                x,
                n_wins.cpu(),
                batch_first=True,
                enforce_sorted=False
                )             
        
        self.lstm.flatten_parameters()
        x = self.lstm(x)[0]
        
        x, _ = pad_packed_sequence(
            x, 
            batch_first=True, 
            padding_value=0.0,
            total_length=n_wins.max())          
  
        return x, n_wins

class SelfAttention(nn.Module):
    '''
    SelfAttention: The main SelfAttention module that can be used as a
    time-dependency model.                                            
    '''         
    def __init__(self,
                 input_size,
                 d_model=512,
                 nhead=8,
                 pool_size=3,
                 pos_enc=None,
                 num_layers=6,
                 sa_h=2048,
                 dropout=0.1,
                 activation="relu"
                 ):
        super().__init__()
        
        encoder_layer = SelfAttentionLayer(d_model, nhead, pool_size, sa_h, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.linear = nn.Linear(input_size, d_model)
        
        self.layers = self._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead      
        
        if pos_enc:
            self.pos_encoder = PositionalEncoding(d_model, dropout)
        else:
            self.pos_encoder = nn.Identity()
            
        self._reset_parameters()
        
    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, n_wins=None):            
        src = self.linear(src)
        output = src.transpose(1,0)
        output = self.norm1(output)
        output = self.pos_encoder(output)
        
        for mod in self.layers:
            output, n_wins = mod(output, n_wins=n_wins)
        return output.transpose(1,0), n_wins

class SelfAttentionLayer(nn.Module):
    '''
    SelfAttentionLayer: The SelfAttentionLayer that is used by the
    SelfAttention module.                                            
    '''          
    def __init__(self, d_model, nhead, pool_size=1, sa_h=2048, dropout=0.1, activation="relu"):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, sa_h)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(sa_h, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = self._get_activation_fn(activation)
        
    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu                
        
    def forward(self, src, n_wins=None):
        
        if n_wins is not None:
            mask = ~((torch.arange(src.shape[0])[None, :]).to(src.device) < n_wins[:, None].to(torch.long).to(src.device))
        else:
            mask = None
        
        src2 = self.self_attn(src, src, src, key_padding_mask=mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)

        src = self.norm2(src)

        return src, n_wins
    
class PositionalEncoding(nn.Module):
    '''
    PositionalEncoding: PositionalEncoding taken from the PyTorch Transformer
    tutorial. Can be applied to the SelfAttention module. However, it did not 
    improve the results in previous experiments.                          
    '''       
    def __init__(self, d_model, dropout=0.1, max_len=3000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)    

#%% Pooling
class Pooling(nn.Module):
    '''
    Pooling: Main Pooling module. It can load either attention-pooling, average
    pooling, maxpooling, or last-step pooling. In case of bidirectional LSTMs
    last-step-bi pooling should be used instead of last-step pooling.
    '''      
    def __init__(self,
                 d_input,
                 output_size=1,
                 pool='att',
                 att_h=None,
                 att_dropout=0,
                 ):
        super().__init__()
        
        if pool=='att':
            if att_h is None:
                self.model = PoolAtt(d_input, output_size)
            else:
                self.model = PoolAttFF(d_input, output_size, h=att_h, dropout=att_dropout)
        elif pool=='last_step_bi':
            self.model = PoolLastStepBi(d_input, output_size)      
        elif pool=='last_step':
            self.model = PoolLastStep(d_input, output_size)                  
        elif pool=='max':
            self.model = PoolMax(d_input, output_size)  
        elif pool=='avg':
            self.model = PoolAvg(d_input, output_size)              
        else:
            raise NotImplementedError('Pool option not available')                     

    def forward(self, x, n_wins):
        return self.model(x, n_wins)
    
class PoolLastStepBi(nn.Module):
    '''
    PoolLastStepBi: last step pooling for the case of bidirectional LSTM
    '''       
    def __init__(self, input_size, output_size):
        super().__init__() 
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x, n_wins=None):    
        x = x.view(x.shape[0], n_wins.max(), 2, x.shape[-1]//2)
        x = torch.cat(
            (x[torch.arange(x.shape[0]), n_wins.type(torch.long)-1, 0, :],
            x[:,0,1,:]),
            dim=1
            )
        x = self.linear(x)
        return x    
    
class PoolLastStep(nn.Module):
    '''
    PoolLastStep: last step pooling can be applied to any one-directional 
    sequence.
    '''      
    def __init__(self, input_size, output_size):
        super().__init__() 
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x, n_wins=None):    
        x = x[torch.arange(x.shape[0]), n_wins.type(torch.long)-1]
        x = self.linear(x)
        return x        

class PoolAtt(torch.nn.Module):
    '''
    PoolAtt: Attention-Pooling module.
    '''          
    def __init__(self, d_input, output_size):
        super().__init__()
        
        self.linear1 = nn.Linear(d_input, 1)
        self.linear2 = nn.Linear(d_input, output_size)

    def forward(self, x, n_wins):
                
        att = self.linear1(x)
        
        att = att.transpose(2,1)
        mask = torch.arange(att.shape[2])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        att[~mask.unsqueeze(1)] = float("-Inf")          
        att = F.softmax(att, dim=2)
        x = torch.bmm(att, x) 
        x = x.squeeze(1)
        
        x = self.linear2(x)
            
        return x
    
class PoolAttFF(torch.nn.Module):
    '''
    PoolAttFF: Attention-Pooling module with additonal feed-forward network.
    '''         
    def __init__(self, d_input, output_size, h, dropout=0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_input, h)
        self.linear2 = nn.Linear(h, 1)
        
        self.linear3 = nn.Linear(d_input, output_size)
        
        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, n_wins):

        att = self.linear2(self.dropout(self.activation(self.linear1(x))))
        att = att.transpose(2,1)
        mask = torch.arange(att.shape[2])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        att[~mask.unsqueeze(1)] = float("-Inf")          
        att = F.softmax(att, dim=2)
        x = torch.bmm(att, x) 
        x = x.squeeze(1)
        
        x = self.linear3(x)
        
        return x    

class PoolAvg(torch.nn.Module):
    '''
    PoolAvg: Average pooling that consideres masked time-steps.
    '''          
    def __init__(self, d_input, output_size):
        super().__init__()
        
        self.linear = nn.Linear(d_input, output_size)
        
    def forward(self, x, n_wins):
                
        mask = torch.arange(x.shape[1])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        mask = ~mask.unsqueeze(2).to(x.device)
        x.masked_fill_(mask, 0)

        x = torch.div(x.sum(1), n_wins.unsqueeze(1))   
            
        x = self.linear(x)
        
        return x
    
class PoolMax(torch.nn.Module):
    '''
    PoolMax: Max-pooling that consideres masked time-steps.
    '''        
    def __init__(self, d_input, output_size):
        super().__init__()
        
        self.linear = nn.Linear(d_input, output_size)
        
    def forward(self, x, n_wins):
                
        mask = torch.arange(x.shape[1])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        mask = ~mask.unsqueeze(2).to(x.device)
        x.masked_fill_(mask, float("-Inf"))

        x = x.max(1)[0]
        
        x = self.linear(x)
            
        return x    
    
#%% Alignment
class Alignment(torch.nn.Module):
    '''
    Alignment: Alignment module for the double-ended NISQA_DE model. It 
    supports five different alignment mechanisms.
    '''       
    def __init__(self,
                 att_method,
                 apply_att_method,
                 q_dim=None,
                 y_dim=None,
                 ):
        super().__init__()
    
        # Attention method --------------------------------------------------------
        if att_method=='bahd':
            self.att = AttBahdanau(
                    q_dim=q_dim,
                    y_dim=y_dim) 
            
        elif att_method=='luong':
            self.att = AttLuong(
                    q_dim=q_dim, 
                    y_dim=y_dim) 
            
        elif att_method=='dot':
            self.att = AttDot()
            
        elif att_method=='cosine':
            self.att = AttCosine()            

        elif att_method=='distance':
            self.att = AttDistance()
            
        elif (att_method=='none') or (att_method is None):
            self.att = None
        else:
            raise NotImplementedError    
        
        # Apply method ----------------------------------------------------------
        if apply_att_method=='soft':
            self.apply_att = ApplySoftAttention() 
        elif apply_att_method=='hard':
            self.apply_att = ApplyHardAttention() 
        else:
            raise NotImplementedError            
            
    def _mask_attention(self, att, y, n_wins):       
        mask = torch.arange(att.shape[2])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        mask = mask.unsqueeze(1).expand_as(att)
        att[~mask] = float("-Inf")    
        
    def forward(self, query, y, n_wins_y):        
        if self.att is not None:
            att_score, sim = self.att(query, y)     
            self._mask_attention(att_score, y, n_wins_y)
            att_score = F.softmax(att_score, dim=2)
            y = self.apply_att(y, att_score) 
        return y        

class AttDot(torch.nn.Module):
    '''
    AttDot: Dot attention that can be used by the Alignment module.
    '''       
    def __init__(self):
        super().__init__()
    def forward(self, query, y):
        att = torch.bmm(query, y.transpose(2,1))
        sim = att.max(2)[0].unsqueeze(1)
        return att, sim
    
class AttCosine(torch.nn.Module):
    '''
    AttCosine: Cosine attention that can be used by the Alignment module.
    '''          
    def __init__(self):
        super().__init__()
        self.pdist = nn.CosineSimilarity(dim=3)
    def forward(self, query, y):
        att = self.pdist(query.unsqueeze(2), y.unsqueeze(1))
        sim = att.max(2)[0].unsqueeze(1)
        return att, sim    
    
class AttDistance(torch.nn.Module):
    '''
    AttDistance: Distance attention that can be used by the Alignment module.
    '''        
    def __init__(self, dist_norm=1, weight_norm=1):
        super().__init__()
        self.dist_norm = dist_norm
        self.weight_norm = weight_norm
    def forward(self, query, y):
        att = (query.unsqueeze(1)-y.unsqueeze(2)).abs().pow(self.dist_norm)
        att = att.mean(dim=3).pow(self.weight_norm)
        att = - att.transpose(2,1)
        sim = att.max(2)[0].unsqueeze(1)
        return att, sim
    
class AttBahdanau(torch.nn.Module):
    '''
    AttBahdanau: Attention according to Bahdanau that can be used by the 
    Alignment module.
    ''' 
    def __init__(self, q_dim, y_dim, att_dim=128):
        super().__init__()
        self.q_dim = q_dim
        self.y_dim = y_dim
        self.att_dim = att_dim
        self.Wq = nn.Linear(self.q_dim, self.att_dim)
        self.Wy = nn.Linear(self.y_dim, self.att_dim)
        self.v = nn.Linear(self.att_dim, 1)
    def forward(self, query, y):
        att = torch.tanh( self.Wq(query).unsqueeze(1) + self.Wy(y).unsqueeze(2) )
        att = self.v(att).squeeze(3).transpose(2,1)
        sim = att.max(2)[0].unsqueeze(1)
        return att, sim

class AttLuong(torch.nn.Module):
    '''
    AttLuong: Attention according to Luong that can be used by the 
    Alignment module.
    '''     
    def __init__(self, q_dim, y_dim):
        super().__init__()
        self.q_dim = q_dim
        self.y_dim = y_dim
        self.W = nn.Linear(self.y_dim, self.q_dim)
    def forward(self, query, y):
        att = torch.bmm(query, self.W(y).transpose(2,1))
        sim = att.max(2)[0].unsqueeze(1)
        return att, sim

class ApplyHardAttention(torch.nn.Module):
    '''
    ApplyHardAttention: Apply hard attention for the purpose of time-alignment.
    '''       
    def __init__(self):
        super().__init__()
    def forward(self, y, att):        
        self.idx = att.argmax(2)
        y = y[torch.arange(y.shape[0]).unsqueeze(-1), self.idx]        
        return y    
    
class ApplySoftAttention(torch.nn.Module):
    '''
    ApplySoftAttention: Apply soft attention for the purpose of time-alignment.
    '''        
    def __init__(self):
        super().__init__()
    def forward(self, y, att):        
        y = torch.bmm(att, y)       
        return y     
    
class Fusion(torch.nn.Module):
    '''
    Fusion: Used by the double-ended NISQA_DE model and used to fuse the
    degraded and reference features.
    '''      
    def __init__(self, fuse_dim=None, in_feat=None, fuse=None):
        super().__init__()
        self.fuse_dim = fuse_dim
        self.fuse = fuse

        if self.fuse=='x/y/-':
            self.fan_out = 3*in_feat        
        elif self.fuse=='+/-':
             self.fan_out = 2*in_feat                     
        elif self.fuse=='x/y':
            self.fan_out = 2*in_feat
        else:
            raise NotImplementedError         
            
        if self.fuse_dim:
            self.lin_fusion = nn.Linear(self.fan_out, self.fuse_dim)
            self.fan_out = fuse_dim       
                        
    def forward(self, x, y):
        
        if self.fuse=='x/y/-':
            x = torch.cat((x, y, x-y), 2)
        elif self.fuse=='+/-':
            x = torch.cat((x+y, x-y), 2)     
        elif self.fuse=='x/y':
            x = torch.cat((x, y), 2)   
        else:
            raise NotImplementedError           
            
        if self.fuse_dim:
            x = self.lin_fusion(x)
            
        return x
        
#%% Evaluation 
def predict_mos(model, ds, bs, dev, num_workers=0):    
    '''
    predict_mos: predicts MOS of the given dataset with given model. Used for
    NISQA and NISQA_DE model.
    '''       
    dl = DataLoader(ds,
                    batch_size=bs,
                    shuffle=False,
                    drop_last=False,
                    pin_memory=False,
                    num_workers=num_workers)
    model.to(dev)
    model.eval()
    with torch.no_grad():
        y_hat_list = [ [model(xb.to(dev), n_wins.to(dev)).cpu().numpy(), yb.cpu().numpy()] for xb, yb, (idx, n_wins) in dl]
    yy = np.concatenate( y_hat_list, axis=1 )
    y_hat = yy[0,:,0].reshape(-1,1)
    y = yy[1,:,0].reshape(-1,1)
    ds.df['mos_pred'] = y_hat.astype(dtype=float)
    return y_hat, y

def predict_dim(model, ds, bs, dev, num_workers=0):     
    '''
    predict_dim: predicts MOS and dimensions of the given dataset with given 
    model. Used for NISQA_DIM model.
    '''        
    dl = DataLoader(ds,
                    batch_size=bs,
                    shuffle=False,
                    drop_last=False,
                    pin_memory=False,
                    num_workers=num_workers)
    model.to(dev)
    model.eval()
    with torch.no_grad():
        y_hat_list = [ [model(xb.to(dev), n_wins.to(dev)).cpu().numpy(), yb.cpu().numpy()] for xb, yb, (idx, n_wins) in dl]
    yy = np.concatenate( y_hat_list, axis=1 )
    
    y_hat = yy[0,:,:]
    y = yy[1,:,:]
    
    ds.df['mos_pred'] = y_hat[:,0].reshape(-1,1)
    ds.df['noi_pred'] = y_hat[:,1].reshape(-1,1)
    ds.df['dis_pred'] = y_hat[:,2].reshape(-1,1)
    ds.df['col_pred'] = y_hat[:,3].reshape(-1,1)
    ds.df['loud_pred'] = y_hat[:,4].reshape(-1,1)
    
    return y_hat, y

def is_const(x):
    if np.linalg.norm(x - np.mean(x)) < 1e-13 * np.abs(np.mean(x)):
        return True
    elif np.all(x==x[0]):
        return True
    else:
        return False
    
def calc_eval_metrics(y, y_hat, y_hat_map=None, d=None, ci=None):
    '''
    Calculate RMSE, mapped RMSE, mapped RMSE* and Pearson's correlation.
    See ITU-T P.1401 for details on RMSE*.
    '''         
    r = {
        'r_p': np.nan,
        'rmse': np.nan,
        'rmse_map': np.nan,
        'rmse_star_map': np.nan,
        }
    if is_const(y_hat) or any(np.isnan(y)):
        r['r_p'] = np.nan
    else:        
        r['r_p'] = pearsonr(y, y_hat)[0]
    r['rmse'] = calc_rmse(y, y_hat)
    if y_hat_map is not None:
        r['rmse_map'] = calc_rmse(y, y_hat_map, d=d)     
        if ci is not None:
            r['rmse_star_map'] = calc_rmse_star(y, y_hat_map, ci, d)[0]
    return r

def calc_rmse(y_true, y_pred, d=0):
    if d==0:
        rmse = np.sqrt(np.mean(np.square(y_true-y_pred)))
    else:
        N = y_true.shape[0]
        if (N-d)<1:
            rmse = np.nan
        else:
            rmse = np.sqrt( 1/(N-d) * np.sum( np.square(y_true-y_pred) ) )  # Eq (7-29) P.1401
    return rmse

def calc_rmse_star(mos_sub, mos_obj, ci, d):
    N = mos_sub.shape[0]
    error = mos_sub-mos_obj

    if np.isnan(ci).any():
        p_error = np.nan
        rmse_star = np.nan
    else:
        p_error = (abs(error)-ci).clip(min=0)   # Eq (7-27) P.1401
        if (N-d)<1:
            rmse_star = np.nan
        else:        
            rmse_star = np.sqrt( 1/(N-d) * sum(p_error**2) )  # Eq (7-29) P.1401

    return rmse_star, p_error, error

def calc_mapped(x, b):
    N = x.shape[0]
    order = b.shape[0]-1
    A = np.zeros([N,order+1])
    for i in range(order+1):
        A[:,i] = x**(i)
    return A @ b

def fit_first_order(y_con, y_con_hat):
    A = np.vstack([np.ones(len(y_con_hat)), y_con_hat]).T
    b = np.linalg.lstsq(A, y_con, rcond=None)[0]
    return b
    
def fit_second_order(y_con, y_con_hat):
    A = np.vstack([np.ones(len(y_con_hat)), y_con_hat, y_con_hat**2]).T
    b = np.linalg.lstsq(A, y_con, rcond=None)[0]
    return b

def fit_third_order(y_con, y_con_hat):
    A = np.vstack([np.ones(len(y_con_hat)), y_con_hat, y_con_hat**2, y_con_hat**3]).T
    b = np.linalg.lstsq(A, y_con, rcond=None)[0]
    
    p = np.poly1d(np.flipud(b))
    p2 = np.polyder(p)
    rr = np.roots(p2)
    r = rr[np.imag(rr)==0]
    monotonic = all( np.logical_or(r>max(y_con_hat),r<min(y_con_hat)) )
    if monotonic==False:
        print('Not monotonic!!!')    
    return b

def fit_monotonic_third_order(
        dfile_db,
        dcon_db=None,
        pred=None,
        target_mos=None,
        target_ci=None, 
        mapping=None):
    '''
    Fits third-order function with the constrained to be monotonically.
    increasing. This function may not return an optimal fitting.
    '''        
    y = dfile_db[target_mos].to_numpy()
    
    y_hat = dfile_db[pred].to_numpy()
    
    if dcon_db is None:
        if target_ci in dfile_db:
            ci = dfile_db[target_ci].to_numpy()   
        else:
            ci = 0        
    else:
        y_con = dcon_db[target_mos].to_numpy()
        
        if target_ci in dcon_db:
            ci = dcon_db[target_ci].to_numpy()   
        else:
            ci = 0         
                
    x = y_hat
    y_hat_min = min(y_hat) - 0.01
    y_hat_max = max(y_hat) + 0.01
 
    def polynomial(p, x):
        return p[0]+p[1]*x+p[2]*x**2+p[3]*x**3

    def constraint_2nd_der(p):
        return 2*p[2]+6*p[3]*x

    def constraint_1st_der(p):
        x = np.arange(y_hat_min, y_hat_max, 0.1)
        return p[1]+2*p[2]*x+3*p[3]*x**2

    def objective_con(p):
        x_map = polynomial(p, x)
        dfile_db['x_map'] = x_map
        x_map_con = dfile_db.groupby('con').mean().x_map.to_numpy() 
        err = x_map_con-y_con
        if mapping=='pError':
            p_err = (abs(err)-ci).clip(min=0)
            return (p_err**2).sum()
        elif mapping=='error':
            return (err**2).sum()
        else:
            raise NotImplementedError         

    def objective_file(p):
        x_map = polynomial(p, x)
        err = x_map-y
        if mapping=='pError':
            p_err = (abs(err)-ci).clip(min=0)
            return (p_err**2).sum()
        elif mapping=='error':
            return (err**2).sum()
        else:
            raise NotImplementedError 

    cons = dict(type='ineq', fun=constraint_1st_der)
    
    if dcon_db is None:
        res = minimize(
            objective_file,
            x0=np.array([0., 1., 0., 0.]),
            method='SLSQP', 
            constraints=cons,
            )
    else:
        res = minimize(
            objective_con,
            x0=np.array([0., 1., 0., 0.]),
            method='SLSQP', 
            constraints=cons,
            )
    b = res.x
    return b

def calc_mapping(
        dfile_db,
        mapping=None,
        dcon_db=None,
        target_mos=None,
        target_ci=None,
        pred=None,
        ):
    '''
    Computes mapping between subjective and predicted MOS. 
    '''       
    if dcon_db is not None:
        y = dcon_db[target_mos].to_numpy()        
        y_hat = dfile_db.groupby('con').mean().get(pred).to_numpy()
    else:
        y = dfile_db[target_mos].to_numpy()        
        y_hat = dfile_db[pred].to_numpy()    
        
    if mapping==None:
        b = np.array([0,1,0,0])
        d_map = 0
    elif mapping=='first_order':
        b = fit_first_order(y, y_hat)
        d_map = 1
    elif mapping=='second_order':
        b = fit_second_order(y, y_hat)            
        d_map = 3
    elif mapping=='third_order_not_monotonic':
        b = fit_third_order(y, y_hat)            
        d_map = 4        
    elif mapping=='third_order':
        b = fit_monotonic_third_order(
            dfile_db,
            dcon_db=dcon_db, 
            pred=pred, 
            target_mos=target_mos, 
            target_ci=target_ci,
            mapping='error',
            )
        d_map = 4
    else:
        raise NotImplementedError      
    
    return b, d_map

def eval_results(
        df,
        dcon=None, 
        target_mos = 'mos', 
        target_ci = 'mos_ci',
        pred = 'mos_pred',
        mapping = None,
        do_print = False,
        do_plot = False
        ):
    '''
    Evaluates a trained model on given dataset.
    '''            
    # Loop through databases
    db_results_df = []
    df['y_hat_map'] = np.nan
    
    for db_name in df.db.astype("category").cat.categories:

        df_db = df.loc[df.db==db_name]
        if dcon is not None:
            dcon_db = dcon.loc[dcon.db==db_name]
        else:
            dcon_db = None
        
        # per file -----------------------------------------------------------        
        y = df_db[target_mos].to_numpy()
        if np.isnan(y).any():
            r = {'r_p': np.nan,'r_s': np.nan,'rmse': np.nan,'r_p_map': np.nan,
                 'r_s_map': np.nan,'rmse_map': np.nan}
        else:
            y_hat = df_db[pred].to_numpy()      
            
            b, d = calc_mapping(
                df_db,
                mapping=mapping,
                target_mos=target_mos,
                target_ci=target_ci,
                pred=pred
                )
            y_hat_map = calc_mapped(y_hat, b)
            
            r = calc_eval_metrics(y, y_hat, y_hat_map=y_hat_map, d=d)
            r.pop('rmse_star_map')
        r = {f'{k}_file': v for k, v in r.items()}
            
        # per con ------------------------------------------------------------
        r_con = {'r_p': np.nan,'r_s': np.nan,'rmse': np.nan,'r_p_map': np.nan,
             'r_s_map': np.nan,'rmse_map': np.nan,'rmse_star_map': np.nan}
        
        if (dcon_db is not None) and ('con' in df_db):
            
            y_con = dcon_db[target_mos].to_numpy()
            y_con_hat = df_db.groupby('con').mean().get(pred).to_numpy()

            if not np.isnan(y_con).any():
                
                if target_ci in dcon_db:
                    ci_con = dcon_db[target_ci].to_numpy()   
                else:
                    ci_con = None            
    
                b_con, d = calc_mapping(
                    df_db,
                    dcon_db=dcon_db,
                    mapping=mapping,
                    target_mos=target_mos,
                    target_ci=target_ci,
                    pred=pred
                    )            
                
                df_db['y_hat_map'] = calc_mapped(y_hat, b_con)  
                df['y_hat_map'].loc[df.db==db_name] = df_db['y_hat_map'] 
                
                y_con_hat_map = df_db.groupby('con').mean().get('y_hat_map').to_numpy()
                r_con = calc_eval_metrics(y_con, y_con_hat, y_hat_map=y_con_hat_map, d=d, ci=ci_con)
                
        r_con = {f'{k}_con': v for k, v in r_con.items()}
        r = {**r, **r_con}
            
        # ---------------------------------------------------------------------
        db_results_df.append({'db': db_name, **r})
        # Plot  ------------------------------------------------------------------
        if do_plot and (not np.isnan(y).any()):
            xx = np.arange(0, 6, 0.01)
            yy = calc_mapped(xx, b)            
        
            plt.figure(figsize=(3.0, 3.0), dpi=300)
            plt.clf()
            plt.plot(y_hat, y, 'o', label='Original data', markersize=2)
            plt.plot([0, 5], [0, 5], 'gray')
            plt.plot(xx, yy, 'r', label='Fitted line')
            plt.axis([1, 5, 1, 5])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid(True)
            plt.xticks(np.arange(1, 6))
            plt.yticks(np.arange(1, 6))
            plt.title(db_name + ' per file')
            plt.ylabel('Subjective ' + target_mos.upper())
            plt.xlabel('Predicted ' + target_mos.upper())
            # plt.savefig('corr_diagram_fr_' + db_name + '.pdf', dpi=300, bbox_inches="tight")
            plt.show()

                
            if (dcon_db is not None) and ('con' in df_db):
                
                xx = np.arange(0, 6, 0.01)
                yy = calc_mapped(xx, b_con)     
                
                plt.figure(figsize=(3.0, 3.0), dpi=300)
                plt.clf()
                plt.plot(y_con_hat, y_con, 'o', label='Original data', markersize=3)
                plt.plot([0, 5], [0, 5], 'gray')
                plt.plot(xx, yy, 'r', label='Fitted line')
                plt.axis([1, 5, 1, 5])
                plt.gca().set_aspect('equal', adjustable='box')
                plt.grid(True)
                plt.xticks(np.arange(1, 6))
                plt.yticks(np.arange(1, 6))
                plt.title(db_name + ' per con')
                plt.ylabel('Sub ' + target_mos.upper())
                plt.xlabel('Pred ' + target_mos.upper())
                # plt.savefig(db_name + '.pdf', dpi=300, bbox_inches="tight")
                plt.show()            
            
        # Print ------------------------------------------------------------------
        if do_print and (not np.isnan(y).any()):
            if (dcon_db is not None) and ('con' in df_db):
                print('%-30s r_p_file: %0.2f, rmse_map_file: %0.2f, r_p_con: %0.2f, rmse_map_con: %0.2f, rmse_star_map_con: %0.2f'
                      % (db_name+':', r['r_p_file'], r['rmse_map_file'], r['r_p_con'], r['rmse_map_con'], r['rmse_star_map_con']))
            else:
                print('%-30s r_p_file: %0.2f, rmse_map_file: %0.2f'
                      % (db_name+':', r['r_p_file'], r['rmse_map_file']))

    # Save individual database results in DataFrame
    db_results_df = pd.DataFrame(db_results_df)
    
    r_average = {}
    r_average['r_p_mean_file'] = db_results_df.r_p_file.mean()
    r_average['rmse_mean_file'] = db_results_df.rmse_file.mean()
    r_average['rmse_map_mean_file'] = db_results_df.rmse_map_file.mean()
    
    if dcon_db is not None:
        r_average['r_p_mean_con'] = db_results_df.r_p_con.mean()
        r_average['rmse_mean_con'] = db_results_df.rmse_con.mean()
        r_average['rmse_map_mean_con'] = db_results_df.rmse_map_con.mean()
        r_average['rmse_star_map_mean_con'] = db_results_df.rmse_star_map_con.mean()    
    else:        
        r_average['r_p_mean_con'] = np.nan
        r_average['rmse_mean_con'] = np.nan
        r_average['rmse_map_mean_con'] = np.nan
        r_average['rmse_star_map_mean_con'] = np.nan

    # Get overall per file results      
    y = df[target_mos].to_numpy()
    y_hat = df[pred].to_numpy()
            
    r_total_file = calc_eval_metrics(y, y_hat)
    r_total_file = {'r_p_all': r_total_file['r_p'], 'rmse_all': r_total_file['rmse']} 
    
    overall_results = {
        **r_total_file,
        **r_average
        }
    
    return db_results_df, overall_results


#%% Loss
class biasLoss(object):
    '''
    Bias loss class. 
    
    Calculates loss while considering database bias.
    '''    
    def __init__(self, db, anchor_db=None, mapping='first_order', min_r=0.7, loss_weight=0.0, do_print=True):
        
        self.db = db
        self.mapping = mapping
        self.min_r = min_r
        self.anchor_db = anchor_db
        self.loss_weight = loss_weight
        self.do_print = do_print
        
        self.b = np.zeros((len(db),4))
        self.b[:,1] = 1
        self.do_update = False
        
        self.apply_bias_loss = True
        if (self.min_r is None) or (self.mapping is None):
            self.apply_bias_loss = False

    def get_loss(self, yb, yb_hat, idx):
        
        if self.apply_bias_loss:
            b = torch.tensor(self.b, dtype=torch.float).to(yb_hat.device)
            b = b[idx,:]
    
            yb_hat_map = (b[:,0]+b[:,1]*yb_hat[:,0]+b[:,2]*yb_hat[:,0]**2+b[:,3]*yb_hat[:,0]**3).view(-1,1)
            
            loss_bias = self._nan_mse(yb_hat_map, yb)   
            loss_normal = self._nan_mse(yb_hat, yb)           
            
            loss = loss_bias + self.loss_weight * loss_normal
        else:
            loss = self._nan_mse(yb_hat, yb)

        return loss
    
    def update_bias(self, y, y_hat):
        
        if self.apply_bias_loss:
            y_hat = y_hat.reshape(-1)
            y = y.reshape(-1)
            
            if not self.do_update:
                r = pearsonr(y[~np.isnan(y)], y_hat[~np.isnan(y)])[0]
                
                if self.do_print:
                    print('--> bias update: min_r {:0.2f}, r_p {:0.2f}'.format(r, self.min_r))
                if r>self.min_r:
                    self.do_update = True
                
            if self.do_update:
                if self.do_print:
                    print('--> bias updated')
                for db_name in self.db.unique():
                    
                    db_idx = (self.db==db_name).to_numpy().nonzero()
                    y_hat_db = y_hat[db_idx]
                    y_db = y[db_idx]
                    
                    if not np.isnan(y_db).any():
                        if self.mapping=='first_order':
                            b_db = self._calc_bias_first_order(y_hat_db, y_db)
                        else:
                            raise NotImplementedError
                        if not db_name==self.anchor_db:
                            self.b[db_idx,:len(b_db)] = b_db                   
                
    def _calc_bias_first_order(self, y_hat, y):
        A = np.vstack([np.ones(len(y_hat)), y_hat]).T
        btmp = np.linalg.lstsq(A, y, rcond=None)[0]
        b = np.zeros((4))
        b[0:2] = btmp
        return b
    
    def _nan_mse(self, y, y_hat):
        err = (y-y_hat).view(-1)
        idx_not_nan = ~torch.isnan(err)
        nan_err = err[idx_not_nan]
        return torch.mean(nan_err**2)    

#%% Early stopping 
class earlyStopper(object):
    '''
    Early stopping class. 
    
    Training is stopped if neither RMSE or Pearson's correlation
    is improving after "patience" epochs.
    '''            
    def __init__(self, patience):
        self.best_rmse = 1e10
        self.best_r_p = -1e10
        self.cnt = -1
        self.patience = patience
        self.best = False
        
    def step(self, r):
        self.best = False
        if r['r_p_mean_file'] > self.best_r_p:
            self.best_r_p = r['r_p_mean_file']
            self.cnt = -1   
        if r['rmse_map_mean_file'] < self.best_rmse:
            self.best_rmse = r['rmse_map_mean_file']
            self.cnt = -1    
            self.best = True
        self.cnt += 1 

        if self.cnt >= self.patience:
            stop_early = True
            return stop_early
        else:
            stop_early = False
            return stop_early
        
class earlyStopper_dim(object):
    '''
    Early stopping class for dimension model. 
    
    Training is stopped if neither RMSE or Pearson's correlation
    is improving after "patience" epochs.
    '''            
    def __init__(self, patience):
        
        self.best_rmse = 1e10
        self.best_rmse_noi = 1e10
        self.best_rmse_col = 1e10
        self.best_rmse_dis = 1e10
        self.best_rmse_loud = 1e10

        self.best_r_p = -1e10
        self.best_r_p_noi = -1e10
        self.best_r_p_col = -1e10
        self.best_r_p_dis = -1e10
        self.best_r_p_loud = -1e10
    
        self.cnt = -1
        self.patience = patience
        self.best = False
        
    def step(self, r):
        
        self.best = False
        
        if r['r_p_mean_file'] > self.best_r_p:
            self.best_r_p = r['r_p_mean_file']
            self.cnt = -1   
        if r['r_p_mean_file_noi'] > self.best_r_p_noi:
            self.best_r_p_noi = r['r_p_mean_file_noi']
            self.cnt = -1               
        if r['r_p_mean_file_col'] > self.best_r_p_col:
            self.best_r_p_col = r['r_p_mean_file_col']
            self.cnt = -1   
        if r['r_p_mean_file_dis'] > self.best_r_p_dis:
            self.best_r_p_dis = r['r_p_mean_file_dis']
            self.cnt = -1   
        if r['r_p_mean_file_loud'] > self.best_r_p_loud:
            self.best_r_p_loud = r['r_p_mean_file_loud']
            self.cnt = -1   
            
        if r['rmse_map_mean_file'] < self.best_rmse:
            self.best_rmse = r['rmse_map_mean_file']
            self.cnt = -1
            self.best = True
        if r['rmse_map_mean_file_noi'] < self.best_rmse_noi:
            self.best_rmse_noi = r['rmse_map_mean_file_noi']
            self.cnt = -1
        if r['rmse_map_mean_file_col'] < self.best_rmse_col:
            self.best_rmse_col = r['rmse_map_mean_file_col']
            self.cnt = -1
        if r['rmse_map_mean_file_dis'] < self.best_rmse_dis:
            self.best_rmse_dis = r['rmse_map_mean_file_dis']
            self.cnt = -1
        if r['rmse_map_mean_file_loud'] < self.best_rmse_loud:
            self.best_rmse_loud = r['rmse_map_mean_file_loud']
            self.cnt = -1
           
        self.cnt += 1 

        if self.cnt >= self.patience:
            stop_early = True
            return stop_early
        else:
            stop_early = False
            return stop_early        

def get_lr(optimizer):
    '''
    Get current learning rate from Pytorch optimizer.
    '''         
    for param_group in optimizer.param_groups:
        return param_group['lr']

#%% Dataset
class SpeechQualityDataset(Dataset):
    '''
    Dataset for Speech Quality Model.
    '''  
    def __init__(
        self,
        df,
        df_con=None,
        data_dir='',
        folder_column='',
        filename_column='filename',
        mos_column='MOS',
        seg_length=15,
        max_length=None,
        to_memory=False,
        to_memory_workers=0,
        transform=None,
        seg_hop_length=1,
        ms_n_fft = 1024,
        ms_hop_length = 80,
        ms_win_length = 170,
        ms_n_mels=32,
        ms_sr=48e3,
        ms_fmax=16e3,
        ms_channel=None,
        double_ended=False,
        filename_column_ref=None,
        dim=False,
        ):

        self.df = df
        self.df_con = df_con
        self.data_dir = data_dir
        self.folder_column = folder_column
        self.filename_column = filename_column
        self.filename_column_ref = filename_column_ref
        self.mos_column = mos_column        
        self.seg_length = seg_length
        self.seg_hop_length = seg_hop_length
        self.max_length = max_length
        self.transform = transform
        self.to_memory_workers = to_memory_workers
        self.ms_n_fft = ms_n_fft
        self.ms_hop_length = ms_hop_length
        self.ms_win_length = ms_win_length
        self.ms_n_mels = ms_n_mels
        self.ms_sr = ms_sr
        self.ms_fmax = ms_fmax
        self.ms_channel = ms_channel
        self.double_ended = double_ended
        self.dim = dim

        # if True load all specs to memory
        self.to_memory = False
        if to_memory:
            self._to_memory()
            
    def _to_memory_multi_helper(self, idx):
        return [self._load_spec(i) for i in idx]
    
    def _to_memory(self):
        if self.to_memory_workers==0:
            self.mem_list = [self._load_spec(idx) for idx in tqdm(range(len(self)))]
        else: 
            buffer_size = 128
            idx = np.arange(len(self))
            n_bufs = int(len(idx)/buffer_size) 
            idx = idx[:buffer_size*n_bufs].reshape(-1,buffer_size).tolist() + idx[buffer_size*n_bufs:].reshape(1,-1).tolist()  
            pool = multiprocessing.Pool(processes=self.to_memory_workers)
            mem_list = []
            for out in tqdm(pool.imap(self._to_memory_multi_helper, idx), total=len(idx)):
                mem_list = mem_list + out
            self.mem_list = mem_list
            pool.terminate()
            pool.join()    
        self.to_memory=True 

    def _load_spec(self, index):
        
            # Load spec    
            file_path = os.path.join(self.data_dir, self.df[self.filename_column].iloc[index])

            if self.double_ended:
                file_path_ref = os.path.join(self.data_dir, self.df[self.filename_column_ref].iloc[index])
       
            spec = get_librosa_melspec(
                file_path,
                sr = self.ms_sr,
                n_fft=self.ms_n_fft,
                hop_length=self.ms_hop_length,
                win_length=self.ms_win_length,
                n_mels=self.ms_n_mels,
                fmax=self.ms_fmax,
                ms_channel=self.ms_channel
                )   
            
            if self.double_ended:
                spec_ref = get_librosa_melspec(
                    file_path_ref,
                    sr = self.ms_sr,
                    n_fft=self.ms_n_fft,
                    hop_length=self.ms_hop_length,
                    win_length=self.ms_win_length,
                    n_mels=self.ms_n_mels,
                    fmax=self.ms_fmax
                    )     
                spec = (spec, spec_ref)
                  
            return spec
            
    def __getitem__(self, index):
        assert isinstance(index, int), 'index must be integer (no slice)'

        if self.to_memory:
            spec = self.mem_list[index]
        else:
            spec = self._load_spec(index)
            
        if self.double_ended:               
            spec, spec_ref = spec
        
        # Apply transformation if given
        if self.transform:
            spec = self.transform(spec)      
            
        # Segment specs
        file_path = os.path.join(self.data_dir, self.df[self.filename_column].iloc[index])
        if self.seg_length is not None:
            x_spec_seg, n_wins = segment_specs(file_path,
                spec,
                self.seg_length,
                self.seg_hop_length,
                self.max_length)
            
            if self.double_ended:               
                x_spec_seg_ref, n_wins_ref = segment_specs(file_path,
                    spec_ref,
                    self.seg_length,
                    self.seg_hop_length,
                    self.max_length)              
        else:
            x_spec_seg = spec
            n_wins = spec.shape[1]
            if self.max_length is not None:
                x_padded = np.zeros((x_spec_seg.shape[0], self.max_length))
                x_padded[:,:n_wins] = x_spec_seg
                x_spec_seg = np.expand_dims(x_padded.transpose(1,0), axis=(1, 3))      
                if not torch.is_tensor(x_spec_seg):
                    x_spec_seg = torch.tensor(x_spec_seg, dtype=torch.float)                  
            
            if self.double_ended:    
                x_spec_seg_ref = spec
                n_wins_ref = spec.shape[1]      
                if self.max_length is not None:
                    x_padded = np.zeros((x_spec_seg_ref.shape[0], self.max_length))
                    x_padded[:,:n_wins] = x_spec_seg_ref
                    x_spec_seg_ref = np.expand_dims(x_padded.transpose(1,0), axis=(1, 3))                     
                    if not torch.is_tensor(x_spec_seg_ref):
                        x_spec_seg_ref = torch.tensor(x_spec_seg_ref, dtype=torch.float)            
                
        if self.double_ended: 
            x_spec_seg = torch.cat((x_spec_seg, x_spec_seg_ref), dim=1)
            n_wins = np.concatenate((n_wins.reshape(1), n_wins_ref.reshape(1)), axis=0)            

        # Get MOS (apply NaN in case of prediction only mode)
        if self.dim:
            if self.mos_column=='predict_only':
                y = np.full((5,1), np.nan).reshape(-1).astype('float32')
            else:                
                y_mos = self.df['mos'].iloc[index].reshape(-1).astype('float32') 
                y_noi = self.df['noi'].iloc[index].reshape(-1).astype('float32')
                y_dis = self.df['dis'].iloc[index].reshape(-1).astype('float32')         
                y_col = self.df['col'].iloc[index].reshape(-1).astype('float32')                
                y_loud = self.df['loud'].iloc[index].reshape(-1).astype('float32')                
                y = np.concatenate((y_mos, y_noi, y_dis, y_col, y_loud), axis=0)
        else:
            if self.mos_column=='predict_only':
                y = np.full(1, np.nan).reshape(-1).astype('float32') 
            else:
                y = self.df[self.mos_column].iloc[index].reshape(-1).astype('float32')   

        return x_spec_seg, y, (index, n_wins)

    def __len__(self):
        return len(self.df)

#%% Spectrograms
def segment_specs(file_path, x, seg_length, seg_hop=1, max_length=None):
    '''
    Segment a spectrogram into "seg_length" wide spectrogram segments.
    Instead of using only the frequency bin of the current time step, 
    the neighboring bins are included as input to the CNN. For example 
    for a seg_length of 7, the previous 3 and the follwing 3 frequency 
    bins are included.

    A spectrogram with input size [H x W] will be segmented to:
    [W-(seg_length-1) x C x H x seg_length], where W is the width of the 
    original mel-spec (corresponding to the length of the speech signal),
    H is the height of the mel-spec (corresponding to the number of mel bands),
    C is the number of CNN input Channels (always one in our case).
    '''      
    if seg_length % 2 == 0:
        raise ValueError('seg_length must be odd! (seg_lenth={})'.format(seg_length))
    if not torch.is_tensor(x):
        x = torch.tensor(x)

    n_wins = x.shape[1]-(seg_length-1)
    
    # broadcast magic to segment melspec
    idx1 = torch.arange(seg_length)
    idx2 = torch.arange(n_wins)
    idx3 = idx1.unsqueeze(0) + idx2.unsqueeze(1)
    x = x.transpose(1,0)[idx3,:].unsqueeze(1).transpose(3,2)
        
    if seg_hop>1:
        x = x[::seg_hop,:]
        n_wins = int(np.ceil(n_wins/seg_hop))
        
    if max_length is not None:
        if max_length < n_wins:
            raise ValueError('n_wins {} > max_length {} --- {}. Increase max window length ms_max_segments!'.format(n_wins, max_length, file_path))
        x_padded = torch.zeros((max_length, x.shape[1], x.shape[2], x.shape[3]))
        x_padded[:n_wins,:] = x
        x = x_padded
                
    return x, np.array(n_wins)

def get_librosa_melspec(
    file_path,
    sr=48e3,
    n_fft=1024, 
    hop_length=80, 
    win_length=170,
    n_mels=32,
    fmax=16e3,
    ms_channel=None,
    ):
    '''
    Calculate mel-spectrograms with Librosa.
    '''    
    # Calc spec
    try:
        if ms_channel is not None:
            y, sr = lb.load(file_path, sr=sr, mono=False)
            if len(y.shape)>1:
                y = y[ms_channel, :]
        else:
            y, sr = lb.load(file_path, sr=sr)
    except:
        raise ValueError('Could not load file {}'.format(file_path))
    
    hop_length = int(sr * hop_length)
    win_length = int(sr * win_length)

    S = lb.feature.melspectrogram(
        y=y,
        sr=sr,
        S=None,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window='hann',
        center=True,
        pad_mode='reflect',
        power=1.0,
    
        n_mels=n_mels,
        fmin=0.0,
        fmax=fmax,
        htk=False,
        norm='slaney',
        )

    spec = lb.core.amplitude_to_db(S, ref=1.0, amin=1e-4, top_db=80.0)
    return spec




# -*- coding: utf-8 -*-
"""
@author: Gabriel Mittag, TU-Berlin
"""
import time
import os
from glob import glob
import datetime
from pathlib import Path

import numpy as np
import pandas as pd; pd.options.mode.chained_assignment=None
from tqdm import tqdm
import yaml
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
# from . import NISQA_lib as NL
        
class nisqaModel(object):
    '''
    nisqaModel: Main class that loads the model and the datasets. Contains
    the training loop, prediction, and evaluation function.                                               
    '''      
    def __init__(self, args):
        self.args = args
        
        if 'mode' not in self.args:
            self.args['mode'] = 'main'
            
        self.runinfos = {}       
        self._getDevice()
        self._loadModel()
        self._loadDatasets()
        self.args['now'] = datetime.datetime.today()
        
        if self.args['mode']=='main':
            print(yaml.dump(self.args, default_flow_style=None, sort_keys=False))

    def train(self):
        
        if self.args['dim']==True:
            self._train_dim()
        else:
            self._train_mos()    
            
    def evaluate(self, mapping='first_order', do_print=True, do_plot=False):
        if self.args['dim']==True:
            self._evaluate_dim(mapping=mapping, do_print=do_print, do_plot=do_plot)
        else:
            self._evaluate_mos(mapping=mapping, do_print=do_print, do_plot=do_plot)      
            
    def predict(self):
        print('---> Predicting ...')
        if self.args['tr_parallel']:
            self.model = nn.DataParallel(self.model)           
        
        if self.args['dim']==True:
            y_val_hat, y_val = predict_dim(
                self.model, 
                self.ds_val, 
                self.args['tr_bs_val'],
                self.dev,
                num_workers=self.args['tr_num_workers'])
        else:
            y_val_hat, y_val = predict_mos(
                self.model, 
                self.ds_val, 
                self.args['tr_bs_val'],
                self.dev,
                num_workers=self.args['tr_num_workers'])                 
                    
        if self.args['output_dir']:
            self.ds_val.df['model'] = self.args['name']
            self.ds_val.df.to_csv(
                os.path.join(self.args['output_dir'], 'NISQA_results.csv'), 
                index=False)
            
        print(self.ds_val.df.to_string(index=False))
        return self.ds_val.df

    def _train_mos(self):
        '''
        Trains speech quality model.
        '''
        # Initialize  -------------------------------------------------------------
        if self.args['tr_parallel']:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.dev)

        # Runname and savepath  ---------------------------------------------------
        self.runname = self._makeRunnameAndWriteYAML()

        # Optimizer  -------------------------------------------------------------
        opt = optim.Adam(self.model.parameters(), lr=self.args['tr_lr'])        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                'min',
                verbose=True,
                threshold=0.003,
                patience=self.args['tr_lr_patience'])
        earlyStp = earlyStopper(self.args['tr_early_stop'])      
        
        biasLoss = biasLoss(
            self.ds_train.df.db, 
            anchor_db=self.args['tr_bias_anchor_db'], 
            mapping=self.args['tr_bias_mapping'], 
            min_r=self.args['tr_bias_min_r'],
            do_print=(self.args['tr_verbose']>0),
            )

        # Dataloader    -----------------------------------------------------------
        dl_train = DataLoader(
            self.ds_train,
            batch_size=self.args['tr_bs'],
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=self.args['tr_num_workers'])
        
        # Start training loop   ---------------------------------------------------
        print('--> start training')
        for epoch in range(self.args['tr_epochs']):

            tic_epoch = time.time()
            batch_cnt = 0
            loss = 0.0
            y_train = self.ds_train.df[self.args['csv_mos_train']].to_numpy().reshape(-1)
            y_train_hat = np.zeros((len(self.ds_train), 1))
            self.model.train()
            
            # Progress bar
            if self.args['tr_verbose'] == 2:
                pbar = tqdm(iterable=batch_cnt, total=len(dl_train), ascii=">—",
                            bar_format='{bar} {percentage:3.0f}%, {n_fmt}/{total_fmt}, {elapsed}<{remaining}{postfix}')
                
            for xb_spec, yb_mos, (idx, n_wins) in dl_train:

                # Estimate batch ---------------------------------------------------
                xb_spec = xb_spec.to(self.dev)
                yb_mos = yb_mos.to(self.dev)
                n_wins = n_wins.to(self.dev)

                # Forward pass ----------------------------------------------------
                yb_mos_hat = self.model(xb_spec, n_wins)
                y_train_hat[idx] = yb_mos_hat.detach().cpu().numpy()

                # Loss ------------------------------------------------------------       
                lossb = biasLoss.get_loss(yb_mos, yb_mos_hat, idx)
                    
                # Backprop  -------------------------------------------------------
                lossb.backward()
                opt.step()
                opt.zero_grad()

                # Update total loss -----------------------------------------------
                loss += lossb.item()
                batch_cnt += 1

                if self.args['tr_verbose'] == 2:
                    pbar.set_postfix(loss=lossb.item())
                    pbar.update()

            if self.args['tr_verbose'] == 2:
                pbar.close()

            loss = loss/batch_cnt
            
            biasLoss.update_bias(y_train, y_train_hat)

            # Evaluate   -----------------------------------------------------------
            if self.args['tr_verbose']>0:
                print('\n<---- Training ---->')
            self.ds_train.df['mos_pred'] = y_train_hat
            db_results_train, r_train = eval_results(
                self.ds_train.df, 
                dcon=self.ds_train.df_con, 
                target_mos=self.args['csv_mos_train'],
                target_ci=self.args['csv_mos_train'] + '_ci',
                pred='mos_pred',
                mapping = 'first_order',
                do_print=(self.args['tr_verbose']>0)
                )
            
            if self.args['tr_verbose']>0:
                print('<---- Validation ---->')
            predict_mos(self.model, self.ds_val, self.args['tr_bs_val'], self.dev, num_workers=self.args['tr_num_workers'])
            db_results, r_val = eval_results(
                self.ds_val.df, 
                dcon=self.ds_val.df_con, 
                target_mos=self.args['csv_mos_val'],
                target_ci=self.args['csv_mos_val'] + '_ci',
                pred='mos_pred',
                mapping = 'first_order',
                do_print=(self.args['tr_verbose']>0)
                )            
            
            r = {'train_r_p_mean_file': r_train['r_p_mean_file'],
                 'train_rmse_map_mean_file': r_train['rmse_map_mean_file'],
                 **r_val}
            
            # Scheduler update    ---------------------------------------------
            scheduler.step(loss)
            earl_stp = earlyStp.step(r)            

            # Print    --------------------------------------------------------
            ep_runtime = time.time() - tic_epoch
            print(
                'ep {} sec {:0.0f} es {} lr {:0.0e} loss {:0.4f} // '
                'r_p_tr {:0.2f} rmse_map_tr {:0.2f} // r_p {:0.2f} rmse_map {:0.2f} // '
                'best_r_p {:0.2f} best_rmse_map {:0.2f},'
                .format(epoch+1, ep_runtime, earlyStp.cnt, get_lr(opt), loss, 
                        r['train_r_p_mean_file'], r['train_rmse_map_mean_file'],
                        r['r_p_mean_file'], r['rmse_map_mean_file'],
                        earlyStp.best_r_p, earlyStp.best_rmse))

            # Save results and model  -----------------------------------------
            self._saveResults(self.model, self.model_args, opt, epoch, loss, ep_runtime, r, db_results, earlyStp.best)

            # Early stopping    -----------------------------------------------
            if earl_stp:
                print('--> Early stopping. best_r_p {:0.2f} best_rmse {:0.2f}'
                    .format(earlyStp.best_r_p, earlyStp.best_rmse))
                return        

        # Training done --------------------------------------------------------
        print('--> Training done. best_r_p {:0.2f} best_rmse_map {:0.2f}'
                            .format(earlyStp.best_r_p, earlyStp.best_rmse))        
        return        
     
        
     
    def _train_dim(self):
        '''
        Trains multidimensional speech quality model.
        '''
        # Initialize  -------------------------------------------------------------
        if self.args['tr_parallel']:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.dev)

        # Runname and savepath  ---------------------------------------------------
        self.runname = self._makeRunnameAndWriteYAML()

        # Optimizer  -------------------------------------------------------------
        opt = optim.Adam(self.model.parameters(), lr=self.args['tr_lr'])        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                'min',
                verbose=True,
                threshold=0.003,
                patience=self.args['tr_lr_patience'])
        earlyStp = earlyStopper_dim(self.args['tr_early_stop'])      
        
        biasLoss_1 = biasLoss(
            self.ds_train.df.db, 
            anchor_db=self.args['tr_bias_anchor_db'], 
            mapping=self.args['tr_bias_mapping'], 
            min_r=self.args['tr_bias_min_r'],
            do_print=(self.args['tr_verbose']>0),
            )
        
        biasLoss_2 = biasLoss(
            self.ds_train.df.db, 
            anchor_db=self.args['tr_bias_anchor_db'], 
            mapping=self.args['tr_bias_mapping'], 
            min_r=self.args['tr_bias_min_r'],
            do_print=(self.args['tr_verbose']>0),
            )
        
        biasLoss_3 = biasLoss(
            self.ds_train.df.db, 
            anchor_db=self.args['tr_bias_anchor_db'], 
            mapping=self.args['tr_bias_mapping'], 
            min_r=self.args['tr_bias_min_r'],
            do_print=(self.args['tr_verbose']>0),
            )
           
        biasLoss_4 = biasLoss(
            self.ds_train.df.db, 
            anchor_db=self.args['tr_bias_anchor_db'], 
            mapping=self.args['tr_bias_mapping'], 
            min_r=self.args['tr_bias_min_r'],
            do_print=(self.args['tr_verbose']>0),
            )
   
        biasLoss_5 = biasLoss(
            self.ds_train.df.db, 
            anchor_db=self.args['tr_bias_anchor_db'], 
            mapping=self.args['tr_bias_mapping'], 
            min_r=self.args['tr_bias_min_r'],
            do_print=(self.args['tr_verbose']>0),
            )
   
        # Dataloader    -----------------------------------------------------------
        dl_train = DataLoader(
            self.ds_train,
            batch_size=self.args['tr_bs'],
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=self.args['tr_num_workers'])
        
        
        # Start training loop   ---------------------------------------------------
        print('--> start training')
        for epoch in range(self.args['tr_epochs']):

            tic_epoch = time.time()
            batch_cnt = 0
            loss = 0.0
            y_mos = self.ds_train.df['mos'].to_numpy().reshape(-1,1)
            y_noi = self.ds_train.df['noi'].to_numpy().reshape(-1,1)
            y_dis = self.ds_train.df['dis'].to_numpy().reshape(-1,1)        
            y_col = self.ds_train.df['col'].to_numpy().reshape(-1,1)    
            y_loud = self.ds_train.df['loud'].to_numpy().reshape(-1,1)          
            y_train = np.concatenate((y_mos, y_noi, y_dis, y_col, y_loud), axis=1)
            y_train_hat = np.zeros((len(self.ds_train), 5))
                                    
            self.model.train()
            
            
            # Progress bar
            if self.args['tr_verbose'] == 2:
                pbar = tqdm(iterable=batch_cnt, total=len(dl_train), ascii=">—",
                            bar_format='{bar} {percentage:3.0f}%, {n_fmt}/{total_fmt}, {elapsed}<{remaining}{postfix}')
                
            for xb_spec, yb_mos, (idx, n_wins) in dl_train:

                # Estimate batch ---------------------------------------------------
                xb_spec = xb_spec.to(self.dev)
                yb_mos = yb_mos.to(self.dev)
                n_wins = n_wins.to(self.dev)
                
                # Forward pass ----------------------------------------------------
                yb_mos_hat = self.model(xb_spec, n_wins)
                y_train_hat[idx,:] = yb_mos_hat.detach().cpu().numpy()

                # Loss ------------------------------------------------------------                       
                lossb_1 = biasLoss_1.get_loss(yb_mos[:,0].view(-1,1), yb_mos_hat[:,0].view(-1,1), idx)
                lossb_2 = biasLoss_2.get_loss(yb_mos[:,1].view(-1,1), yb_mos_hat[:,1].view(-1,1), idx)
                lossb_3 = biasLoss_3.get_loss(yb_mos[:,2].view(-1,1), yb_mos_hat[:,2].view(-1,1), idx)
                lossb_4 = biasLoss_4.get_loss(yb_mos[:,3].view(-1,1), yb_mos_hat[:,3].view(-1,1), idx)
                lossb_5 = biasLoss_5.get_loss(yb_mos[:,4].view(-1,1), yb_mos_hat[:,4].view(-1,1), idx)
                
                lossb = lossb_1 + lossb_2 + lossb_3 + lossb_4 + lossb_5
                    
                # Backprop  -------------------------------------------------------
                lossb.backward()
                opt.step()
                opt.zero_grad()

                # Update total loss -----------------------------------------------
                loss += lossb.item()
                batch_cnt += 1

                if self.args['tr_verbose'] == 2:
                    pbar.set_postfix(loss=lossb.item())
                    pbar.update()

            if self.args['tr_verbose'] == 2:
                pbar.close()

            loss = loss/batch_cnt
     
            biasLoss_1.update_bias(y_train[:,0].reshape(-1,1), y_train_hat[:,0].reshape(-1,1))
            biasLoss_2.update_bias(y_train[:,1].reshape(-1,1), y_train_hat[:,1].reshape(-1,1))
            biasLoss_3.update_bias(y_train[:,2].reshape(-1,1), y_train_hat[:,2].reshape(-1,1))
            biasLoss_4.update_bias(y_train[:,3].reshape(-1,1), y_train_hat[:,3].reshape(-1,1))
            biasLoss_5.update_bias(y_train[:,4].reshape(-1,1), y_train_hat[:,4].reshape(-1,1))  
                
            # Evaluate   -----------------------------------------------------------
            self.ds_train.df['mos_pred'] = y_train_hat[:,0].reshape(-1,1)
            self.ds_train.df['noi_pred'] = y_train_hat[:,1].reshape(-1,1)
            self.ds_train.df['dis_pred'] = y_train_hat[:,2].reshape(-1,1)
            self.ds_train.df['col_pred'] = y_train_hat[:,3].reshape(-1,1)
            self.ds_train.df['loud_pred'] = y_train_hat[:,4].reshape(-1,1)
            
            if self.args['tr_verbose']>0:
                print('\n<---- Training ---->')
                print('--> MOS:')
            db_results_train_mos, r_train_mos = eval_results(
                self.ds_train.df, 
                dcon=self.ds_train.df_con, 
                target_mos='mos',
                target_ci='mos_ci',
                pred='mos_pred',
                mapping = 'first_order',
                do_print=(self.args['tr_verbose']>0)
                )
            
            if self.args['tr_verbose']>0:
                print('--> NOI:')
            db_results_train_noi, r_train_noi = eval_results(
                self.ds_train.df, 
                dcon=self.ds_train.df_con, 
                target_mos='noi',
                target_ci='noi_ci',
                pred='noi_pred',
                mapping = 'first_order',
                do_print=(self.args['tr_verbose']>0)
                )            
            
            if self.args['tr_verbose']>0:
                print('--> DIS:')
            db_results_train_dis, r_train_dis = eval_results(
                self.ds_train.df, 
                dcon=self.ds_train.df_con, 
                target_mos='dis',
                target_ci='dis_ci',
                pred='dis_pred',
                mapping = 'first_order',
                do_print=(self.args['tr_verbose']>0)
                )            
            
            if self.args['tr_verbose']>0:
                print('--> COL:')
            db_results_train_col, r_train_col = eval_results(
                self.ds_train.df, 
                dcon=self.ds_train.df_con, 
                target_mos='col',
                target_ci='col_ci',
                pred='col_pred',
                mapping = 'first_order',
                do_print=(self.args['tr_verbose']>0)
                )      
            
            if self.args['tr_verbose']>0:
                print('--> LOUD:')
            db_results_train_loud, r_train_loud = eval_results(
                self.ds_train.df, 
                dcon=self.ds_train.df_con, 
                target_mos='loud',
                target_ci='loud_ci',
                pred='loud_pred',
                mapping = 'first_order',
                do_print=(self.args['tr_verbose']>0)
                )           
            
            predict_dim(self.model, self.ds_val, self.args['tr_bs_val'], self.dev, num_workers=self.args['tr_num_workers'])
            
            if self.args['tr_verbose']>0:
                print('<---- Validation ---->')
                print('--> MOS:')
            db_results_val_mos, r_val_mos = eval_results(
                self.ds_val.df, 
                dcon=self.ds_val.df_con, 
                target_mos='mos',
                target_ci='mos_ci',
                pred='mos_pred',
                mapping = 'first_order',
                do_print=(self.args['tr_verbose']>0)
                )    
            
            if self.args['tr_verbose']>0:
                print('--> NOI:')
            db_results_val_noi, r_val_noi = eval_results(
                self.ds_val.df, 
                dcon=self.ds_val.df_con, 
                target_mos='noi',
                target_ci='noi_ci',
                pred='noi_pred',
                mapping = 'first_order',
                do_print=(self.args['tr_verbose']>0)
                )           
            r_val_noi = {k+'_noi': v for k, v in r_val_noi.items()}
            
            if self.args['tr_verbose']>0:
                print('--> DIS:')
            db_results_val_dis, r_val_dis = eval_results(
                self.ds_val.df, 
                dcon=self.ds_val.df_con, 
                target_mos='dis',
                target_ci='dis_ci',
                pred='dis_pred',
                mapping = 'first_order',
                do_print=(self.args['tr_verbose']>0)
                )          
            r_val_dis = {k+'_dis': v for k, v in r_val_dis.items()}
            
            if self.args['tr_verbose']>0:
                print('--> COL:')
            db_results_val_col, r_val_col = eval_results(
                self.ds_val.df, 
                dcon=self.ds_val.df_con, 
                target_mos='col',
                target_ci='col_ci',
                pred='col_pred',
                mapping = 'first_order',
                do_print=(self.args['tr_verbose']>0)
                )      
            r_val_col = {k+'_col': v for k, v in r_val_col.items()}
            
            if self.args['tr_verbose']>0:
                print('--> LOUD:')
            db_results_val_loud, r_val_loud = eval_results(
                self.ds_val.df, 
                dcon=self.ds_val.df_con, 
                target_mos='loud',
                target_ci='loud_ci',
                pred='loud_pred',
                mapping = 'first_order',
                do_print=(self.args['tr_verbose']>0)
                )            
            r_val_loud = {k+'_loud': v for k, v in r_val_loud.items()}
            
            r = {
                'train_r_p_mean_file': r_train_mos['r_p_mean_file'],
                 'train_rmse_map_mean_file': r_train_mos['rmse_map_mean_file'],
                 
                'train_r_p_mean_file_noi': r_train_noi['r_p_mean_file'],
                 'train_rmse_map_mean_file_noi': r_train_noi['rmse_map_mean_file'],

                'train_r_p_mean_file_dis': r_train_dis['r_p_mean_file'],
                 'train_rmse_map_mean_file_dis': r_train_dis['rmse_map_mean_file'],

                'train_r_p_mean_file_col': r_train_col['r_p_mean_file'],
                 'train_rmse_map_mean_file_col': r_train_col['rmse_map_mean_file'],

                'train_r_p_mean_file_loud': r_train_loud['r_p_mean_file'],
                 'train_rmse_map_mean_file_loud': r_train_loud['rmse_map_mean_file'],
                 
                 **r_val_mos, **r_val_noi, **r_val_dis, **r_val_col, **r_val_loud, }
            
            db_results = {
                'db_results_val_mos': db_results_val_mos,
                'db_results_val_noi': db_results_val_noi,
                'db_results_val_dis': db_results_val_dis,
                'db_results_val_col': db_results_val_col,
                'db_results_val_loud': db_results_val_loud
                          }             
            
            # Scheduler update    ---------------------------------------------
            scheduler.step(loss)
            earl_stp = earlyStp.step(r)            

            # Print    --------------------------------------------------------
            ep_runtime = time.time() - tic_epoch

            r_dim_mos_mean = 1/5 * (r['r_p_mean_file'] + 
                      r['r_p_mean_file_noi'] +
                      r['r_p_mean_file_col'] +
                      r['r_p_mean_file_dis'] +
                      r['r_p_mean_file_loud'])

            print(
                'ep {} sec {:0.0f} es {} lr {:0.0e} loss {:0.4f} // '
                'r_p_tr {:0.2f} rmse_map_tr {:0.2f} // r_dim_mos_mean {:0.2f}, r_p {:0.2f} rmse_map {:0.2f} // '
                'best_r_p {:0.2f} best_rmse_map {:0.2f},'
                .format(epoch+1, ep_runtime, earlyStp.cnt, get_lr(opt), loss, 
                        r['train_r_p_mean_file'], r['train_rmse_map_mean_file'],
                        r_dim_mos_mean,
                        r['r_p_mean_file'], r['rmse_map_mean_file'],
                        earlyStp.best_r_p, earlyStp.best_rmse))

            # Save results and model  -----------------------------------------
            self._saveResults(self.model, self.model_args, opt, epoch, loss, ep_runtime, r, db_results, earlyStp.best)

            # Early stopping    -----------------------------------------------
            if earl_stp:
                print('--> Early stopping. best_r_p {:0.2f} best_rmse {:0.2f}'
                    .format(earlyStp.best_r_p, earlyStp.best_rmse))
                return        

        # Training done --------------------------------------------------------
        print('--> Training done. best_r_p {:0.2f} best_rmse {:0.2f}'
                            .format(earlyStp.best_r_p, earlyStp.best_rmse))        
        return        
            
    
    def _evaluate_mos(self, mapping='first_order', do_print=True, do_plot=False):
        '''
        Evaluates the model's predictions.
        '''        
        print('--> MOS:')
        self.db_results, self.r = eval_results(
            self.ds_val.df,
            dcon=self.ds_val.df_con,
            target_mos='mos',
            target_ci='mos_ci',
            pred='mos_pred',
            mapping=mapping,
            do_print=do_print,
            do_plot=do_plot
            )
        if self.ds_val.df_con is None:
            print('r_p_mean_file: {:0.2f}, rmse_mean_file: {:0.2f}'
                .format(self.r['r_p_mean_file'], self.r['rmse_mean_file'])
                )                  
        else:
            print('r_p_mean_con: {:0.2f}, rmse_mean_con: {:0.2f}, rmse_star_map_mean_con: {:0.2f}'
                .format(self.r['r_p_mean_con'], self.r['rmse_mean_con'], self.r['rmse_star_map_mean_con'])
                )             
    
    def _evaluate_dim(self, mapping='first_order', do_print=True, do_plot=False):
        '''
        Evaluates the predictions of a multidimensional model.
        '''            
        print('--> MOS:')
        self.db_results_val_mos, r_val_mos = eval_results(
            self.ds_val.df, 
            dcon=self.ds_val.df_con, 
            target_mos='mos',
            target_ci='mos_ci',
            pred='mos_pred',
            mapping=mapping,
            do_print=do_print,
            do_plot=do_plot
            )       
        if self.ds_val.df_con is None:
            print('r_p_mean_file: {:0.2f}, rmse_mean_file: {:0.2f}'
                .format(r_val_mos['r_p_mean_file'], r_val_mos['rmse_mean_file'])
                )                  
        else:
            print('r_p_mean_con: {:0.2f}, rmse_mean_con: {:0.2f}, rmse_star_map_mean_con: {:0.2f}'
                .format(r_val_mos['r_p_mean_con'], r_val_mos['rmse_mean_con'], r_val_mos['rmse_star_map_mean_con'])
                )    
                
        print('--> NOI:')
        self.db_results_val_noi, r_val_noi = eval_results(
            self.ds_val.df, 
            dcon=self.ds_val.df_con, 
            target_mos='noi',
            target_ci='noi_ci',
            pred='noi_pred',
            mapping=mapping,
            do_print=do_print,
            do_plot=do_plot
            )  
        if self.ds_val.df_con is None:
            print('r_p_mean_file: {:0.2f}, rmse_mean_file: {:0.2f}'
                .format(r_val_noi['r_p_mean_file'], r_val_noi['rmse_mean_file'])
                )                  
        else:
            print('r_p_mean_con: {:0.2f}, rmse_mean_con: {:0.2f}'
                .format(r_val_noi['r_p_mean_con'], r_val_noi['rmse_mean_con'], r_val_noi['rmse_star_map_mean_con'])
                )            
        r_val_noi = {k+'_noi': v for k, v in r_val_noi.items()}
        
        print('--> DIS:')
        self.db_results_val_dis, r_val_dis = eval_results(
            self.ds_val.df, 
            dcon=self.ds_val.df_con, 
            target_mos='dis',
            target_ci='dis_ci',
            pred='dis_pred',
            mapping=mapping,
            do_print=do_print,
            do_plot=do_plot
            )
        if self.ds_val.df_con is None:
            print('r_p_mean_file: {:0.2f}, rmse_mean_file: {:0.2f}'
                .format(r_val_dis['r_p_mean_file'], r_val_dis['rmse_mean_file'])
                )                  
        else:
            print('r_p_mean_con: {:0.2f}, rmse_mean_con: {:0.2f}, rmse_star_map_mean_con: {:0.2f}'
                .format(r_val_dis['r_p_mean_con'], r_val_dis['rmse_mean_con'], r_val_dis['rmse_star_map_mean_con'])
                )               
        r_val_dis = {k+'_dis': v for k, v in r_val_dis.items()}
        
        print('--> COL:')
        self.db_results_val_col, r_val_col = eval_results(
            self.ds_val.df, 
            dcon=self.ds_val.df_con, 
            target_mos='col',
            target_ci='col_ci',
            pred='col_pred',
            mapping=mapping,
            do_print=do_print,
            do_plot=do_plot
            )  
        if self.ds_val.df_con is None:
            print('r_p_mean_file: {:0.2f}, rmse_mean_file: {:0.2f}'
                .format(r_val_col['r_p_mean_file'], r_val_col['rmse_mean_file'])
                )                  
        else:
            print('r_p_mean_con: {:0.2f}, rmse_mean_con: {:0.2f}, rmse_star_map_mean_con: {:0.2f}'
                .format(r_val_col['r_p_mean_con'], r_val_col['rmse_mean_con'], r_val_col['rmse_star_map_mean_con'])
                )            
        r_val_col = {k+'_col': v for k, v in r_val_col.items()}
        
        print('--> LOUD:')
        self.db_results_val_loud, r_val_loud = eval_results(
            self.ds_val.df, 
            dcon=self.ds_val.df_con, 
            target_mos='loud',
            target_ci='loud_ci',
            pred='loud_pred',
            mapping=mapping,
            do_print=do_print,
            do_plot=do_plot
            )
        if self.ds_val.df_con is None:
            print('r_p_mean_file: {:0.2f}, rmse_mean_file: {:0.2f}'
                .format(r_val_loud['r_p_mean_file'], r_val_loud['rmse_mean_file'])
                )                  
        else:
            print('r_p_mean_con: {:0.2f}, rmse_mean_con: {:0.2f}, rmse_star_map_mean_con: {:0.2f}'
                .format(r_val_loud['r_p_mean_con'], r_val_loud['rmse_mean_con'], r_val_loud['rmse_star_map_mean_con'])
                )                    
        r_val_loud = {k+'_loud': v for k, v in r_val_loud.items()}
        
        self.r = {             
             **r_val_mos, **r_val_noi, **r_val_dis, **r_val_col, **r_val_loud, }            
        
        r_mean = 1/5 * (self.r['r_p_mean_con'] + 
                  self.r['r_p_mean_con_noi'] +
                  self.r['r_p_mean_con_col'] +
                  self.r['r_p_mean_con_dis'] +
                  self.r['r_p_mean_con_loud'])
                  
        print('\nAverage over MOS and dimensions: r_p={:0.3f}'
            .format(r_mean)
            )
                

    def _makeRunnameAndWriteYAML(self):
        '''
        Creates individual run name.
        '''        
        runname = self.args['name'] + '_' + self.args['now'].strftime("%y%m%d_%H%M%S%f")
        print('runname: ' + runname)
        run_output_dir = os.path.join(self.args['output_dir'], runname)
        Path(run_output_dir).mkdir(parents=True, exist_ok=True)
        yaml_path = os.path.join(run_output_dir, runname+'.yaml')
        with open(yaml_path, 'w') as file:
            yaml.dump(self.args, file, default_flow_style=None, sort_keys=False)          

        return runname
    
    def _loadDatasets(self):
        if self.args['mode']=='predict_file':
            self._loadDatasetsFile()
        elif self.args['mode']=='predict_dir':
            self._loadDatasetsFolder()  
        elif self.args['mode']=='predict_csv':
            self._loadDatasetsCSVpredict()
        elif self.args['mode']=='main':
            self._loadDatasetsCSV()
        else:
            raise NotImplementedError('mode not available')                        
            
    
    def _loadDatasetsFolder(self):
        files = glob( os.path.join(self.args['data_dir'], '*.wav') ) # @TODO: change the path
        files = [os.path.basename(files) for files in files]
        df_val = pd.DataFrame(files, columns=['deg'])
     
        print('# files: {}'.format( len(df_val) ))
        if len(df_val)==0:
            raise ValueError('No wav files found in data_dir')   
        
        # creating Datasets ---------------------------------------------------                        
        self.ds_val = SpeechQualityDataset(
            df_val,
            df_con=None,
            data_dir = self.args['data_dir'],
            filename_column = 'deg',
            mos_column = 'predict_only',              
            seg_length = self.args['ms_seg_length'],
            max_length = self.args['ms_max_segments'],
            to_memory = None,
            to_memory_workers = None,
            seg_hop_length = self.args['ms_seg_hop_length'],
            transform = None,
            ms_n_fft = self.args['ms_n_fft'],
            ms_hop_length = self.args['ms_hop_length'],
            ms_win_length = self.args['ms_win_length'],
            ms_n_mels = self.args['ms_n_mels'],
            ms_sr = self.args['ms_sr'],
            ms_fmax = self.args['ms_fmax'],
            ms_channel = self.args['ms_channel'],
            double_ended = self.args['double_ended'],
            dim = self.args['dim'],
            filename_column_ref = None,
            )
        
        
    def _loadDatasetsFile(self):
        data_dir = os.path.dirname(self.args['deg'])
        file_name = os.path.basename(self.args['deg'])        
        df_val = pd.DataFrame([file_name], columns=['deg'])
                
        # creating Datasets ---------------------------------------------------                        
        self.ds_val = SpeechQualityDataset(
            df_val,
            df_con=None,
            data_dir = data_dir,
            filename_column = 'deg',
            mos_column = 'predict_only',              
            seg_length = self.args['ms_seg_length'],
            max_length = self.args['ms_max_segments'],
            to_memory = None,
            to_memory_workers = None,
            seg_hop_length = self.args['ms_seg_hop_length'],
            transform = None,
            ms_n_fft = self.args['ms_n_fft'],
            ms_hop_length = self.args['ms_hop_length'],
            ms_win_length = self.args['ms_win_length'],
            ms_n_mels = self.args['ms_n_mels'],
            ms_sr = self.args['ms_sr'],
            ms_fmax = self.args['ms_fmax'],
            ms_channel = self.args['ms_channel'],
            double_ended = self.args['double_ended'],
            dim = self.args['dim'],
            filename_column_ref = None,
        )
                
        
    def _loadDatasetsCSVpredict(self):         
        '''
        Loads validation dataset for prediction only.
        '''            
        csv_file_path = os.path.join(self.args['data_dir'], self.args['csv_file'])
        dfile = pd.read_csv(csv_file_path)
        if 'csv_con' in self.args:
            csv_con_file_path = os.path.join(self.args['data_dir'], self.args['csv_con'])
            dcon = pd.read_csv(csv_con_file_path)        
        else:
            dcon = None
        

        # creating Datasets ---------------------------------------------------                        
        self.ds_val = SpeechQualityDataset(
            dfile,
            df_con=dcon,
            data_dir = self.args['data_dir'],
            filename_column = self.args['csv_deg'],
            mos_column = 'predict_only',              
            seg_length = self.args['ms_seg_length'],
            max_length = self.args['ms_max_segments'],
            to_memory = False,
            to_memory_workers = None,
            seg_hop_length = self.args['ms_seg_hop_length'],
            transform = None,
            ms_n_fft = self.args['ms_n_fft'],
            ms_hop_length = self.args['ms_hop_length'],
            ms_win_length = self.args['ms_win_length'],
            ms_n_mels = self.args['ms_n_mels'],
            ms_sr = self.args['ms_sr'],
            ms_fmax = self.args['ms_fmax'],
            ms_channel = self.args['ms_channel'],
            double_ended = self.args['double_ended'],
            dim = self.args['dim'],
            filename_column_ref = self.args['csv_ref'],
        )

        
    def _loadDatasetsCSV(self):    
        '''
        Loads training and validation dataset for training.
        '''          
        csv_file_path = os.path.join(self.args['data_dir'], self.args['csv_file'])
        dfile = pd.read_csv(csv_file_path)

        if not set(self.args['csv_db_train'] + self.args['csv_db_val']).issubset(dfile.db.unique().tolist()):
            missing_datasets = set(self.args['csv_db_train'] + self.args['csv_db_val']).difference(dfile.db.unique().tolist())
            raise ValueError('Not all dbs found in csv:', missing_datasets)
        
        df_train = dfile[dfile.db.isin(self.args['csv_db_train'])].reset_index()
        df_val = dfile[dfile.db.isin(self.args['csv_db_val'])].reset_index()
        
        if self.args['csv_con'] is not None:
            csv_con_path = os.path.join(self.args['data_dir'], self.args['csv_con'])
            dcon = pd.read_csv(csv_con_path)
            dcon_train = dcon[dcon.db.isin(self.args['csv_db_train'])].reset_index()
            dcon_val = dcon[dcon.db.isin(self.args['csv_db_val'])].reset_index()        
        else:
            dcon = None        
            dcon_train = None        
            dcon_val = None        
        
        print('Training size: {}, Validation size: {}'.format(len(df_train), len(df_val)))
        
        # creating Datasets ---------------------------------------------------                        
        self.ds_train = SpeechQualityDataset(
            df_train,
            df_con=dcon_train,
            data_dir = self.args['data_dir'],
            filename_column = self.args['csv_deg'],
            mos_column = self.args['csv_mos_train'],            
            seg_length = self.args['ms_seg_length'],
            max_length = self.args['ms_max_segments'],
            to_memory = self.args['tr_ds_to_memory'],
            to_memory_workers = self.args['tr_ds_to_memory_workers'],
            seg_hop_length = self.args['ms_seg_hop_length'],
            transform = None,
            ms_n_fft = self.args['ms_n_fft'],
            ms_hop_length = self.args['ms_hop_length'],
            ms_win_length = self.args['ms_win_length'],
            ms_n_mels = self.args['ms_n_mels'],
            ms_sr = self.args['ms_sr'],
            ms_fmax = self.args['ms_fmax'],
            ms_channel = self.args['ms_channel'],
            double_ended = self.args['double_ended'],
            dim = self.args['dim'],
            filename_column_ref = self.args['csv_ref'],
        )

        self.ds_val = SpeechQualityDataset(
            df_val,
            df_con=dcon_val,
            data_dir = self.args['data_dir'],
            filename_column = self.args['csv_deg'],
            mos_column = self.args['csv_mos_val'],              
            seg_length = self.args['ms_seg_length'],
            max_length = self.args['ms_max_segments'],
            to_memory = self.args['tr_ds_to_memory'],
            to_memory_workers = self.args['tr_ds_to_memory_workers'],
            seg_hop_length = self.args['ms_seg_hop_length'],
            transform = None,
            ms_n_fft = self.args['ms_n_fft'],
            ms_hop_length = self.args['ms_hop_length'],
            ms_win_length = self.args['ms_win_length'],
            ms_n_mels = self.args['ms_n_mels'],
            ms_sr = self.args['ms_sr'],
            ms_fmax = self.args['ms_fmax'],
            ms_channel = self.args['ms_channel'],
            double_ended = self.args['double_ended'],
            dim = self.args['dim'],
            filename_column_ref = self.args['csv_ref'],                        
            )

        self.runinfos['ds_train_len'] = len(self.ds_train)
        self.runinfos['ds_val_len'] = len(self.ds_val)
    
    def _loadModel(self):    
        '''
        Loads the Pytorch models with given input arguments.
        '''   
        # if True overwrite input arguments from pretrained model
        if self.args['pretrained_model']:
            if os.path.isabs(self.args['pretrained_model']):
                model_path = os.path.join(self.args['pretrained_model'])
            else:
                model_path = os.path.join(os.getcwd(), self.args['pretrained_model'])
            checkpoint = torch.load(model_path, map_location=self.dev)
            
            # update checkpoint arguments with new arguments
            checkpoint['args'].update(self.args)
            self.args = checkpoint['args']
            
        if self.args['model']=='NISQA_DIM':
            self.args['dim'] = True
            self.args['csv_mos_train'] = None # column names hardcoded for dim models
            self.args['csv_mos_val'] = None  
        else:
            self.args['dim'] = False
            
        if self.args['model']=='NISQA_DE':
            self.args['double_ended'] = True
        else:
            self.args['double_ended'] = False     
            self.args['csv_ref'] = None

        # Load Model
        self.model_args = {
            
            'ms_seg_length': self.args['ms_seg_length'],
            'ms_n_mels': self.args['ms_n_mels'],
            
            'cnn_model': self.args['cnn_model'],
            'cnn_c_out_1': self.args['cnn_c_out_1'],
            'cnn_c_out_2': self.args['cnn_c_out_2'],
            'cnn_c_out_3': self.args['cnn_c_out_3'],
            'cnn_kernel_size': self.args['cnn_kernel_size'],
            'cnn_dropout': self.args['cnn_dropout'],
            'cnn_pool_1': self.args['cnn_pool_1'],
            'cnn_pool_2': self.args['cnn_pool_2'],
            'cnn_pool_3': self.args['cnn_pool_3'],
            'cnn_fc_out_h': self.args['cnn_fc_out_h'],
            
            'td': self.args['td'],
            'td_sa_d_model': self.args['td_sa_d_model'],
            'td_sa_nhead': self.args['td_sa_nhead'],
            'td_sa_pos_enc': self.args['td_sa_pos_enc'],
            'td_sa_num_layers': self.args['td_sa_num_layers'],
            'td_sa_h': self.args['td_sa_h'],
            'td_sa_dropout': self.args['td_sa_dropout'],
            'td_lstm_h': self.args['td_lstm_h'],
            'td_lstm_num_layers': self.args['td_lstm_num_layers'],
            'td_lstm_dropout': self.args['td_lstm_dropout'],
            'td_lstm_bidirectional': self.args['td_lstm_bidirectional'],
            
            'td_2': self.args['td_2'],
            'td_2_sa_d_model': self.args['td_2_sa_d_model'],
            'td_2_sa_nhead': self.args['td_2_sa_nhead'],
            'td_2_sa_pos_enc': self.args['td_2_sa_pos_enc'],
            'td_2_sa_num_layers': self.args['td_2_sa_num_layers'],
            'td_2_sa_h': self.args['td_2_sa_h'],
            'td_2_sa_dropout': self.args['td_2_sa_dropout'],
            'td_2_lstm_h': self.args['td_2_lstm_h'],
            'td_2_lstm_num_layers': self.args['td_2_lstm_num_layers'],
            'td_2_lstm_dropout': self.args['td_2_lstm_dropout'],
            'td_2_lstm_bidirectional': self.args['td_2_lstm_bidirectional'],                
            
            'pool': self.args['pool'],
            'pool_att_h': self.args['pool_att_h'],
            'pool_att_dropout': self.args['pool_att_dropout'],
            }
            
        if self.args['double_ended']:
            self.model_args.update({
                'de_align': self.args['de_align'],
                'de_align_apply': self.args['de_align_apply'],
                'de_fuse_dim': self.args['de_fuse_dim'],
                'de_fuse': self.args['de_fuse'],        
                })
                      
        print('Model architecture: ' + self.args['model'])
        if self.args['model']=='NISQA':
            self.model = NISQA(**self.model_args)     
        elif self.args['model']=='NISQA_DIM':
            self.model = NISQA_DIM(**self.model_args)     
        elif self.args['model']=='NISQA_DE':
            self.model = NISQA_DE(**self.model_args)     
        else:
            raise NotImplementedError('Model not available')                        
        
        # Load weights if pretrained model is used ------------------------------------
        if self.args['pretrained_model']:
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print('Loaded pretrained model from ' + self.args['pretrained_model'])
            if missing_keys:
                print('missing_keys:')
                print(missing_keys)
            if unexpected_keys:
                print('unexpected_keys:')
                print(unexpected_keys)        

    def _getDevice(self):
        '''
        Train on GPU if available.
        '''         
        if torch.cuda.is_available():
            self.dev = torch.device("cuda")
        else:
            self.dev = torch.device("cpu")
    
        if "tr_device" in self.args:
            if self.args['tr_device']=='cpu':
                self.dev = torch.device("cpu")
            elif self.args['tr_device']=='cuda':
                self.dev = torch.device("cuda")
        print('Device: {}'.format(self.dev))
        
        if "tr_parallel" in self.args:
            if (self.dev==torch.device("cpu")) and self.args['tr_parallel']==True:
                self.args['tr_parallel']==False 
                print('Using CPU -> tr_parallel set to False')

    def _saveResults(self, model, model_args, opt, epoch, loss, ep_runtime, r, db_results, best):
        '''
        Save model/results in dictionary and write results csv.
        ''' 
        if (self.args['tr_checkpoint'] == 'best_only'):
            filename = self.runname + '.tar'
        else:
            filename = self.runname + '__' + ('ep_{:03d}'.format(epoch+1)) + '.tar'
        run_output_dir = os.path.join(self.args['output_dir'], self.runname)
        model_path = os.path.join(run_output_dir, filename)
        results_path = os.path.join(run_output_dir, self.runname+'__results.csv')
        Path(run_output_dir).mkdir(parents=True, exist_ok=True)              
        
        results = {
            'runname': self.runname,
            'epoch': '{:05d}'.format(epoch+1),
            'filename': filename,
            'loss': loss,
            'ep_runtime': '{:0.2f}'.format(ep_runtime),
            **self.runinfos,
            **r,
            **self.args,
            }
        
        for key in results: 
            results[key] = str(results[key])                        

        if epoch==0:
            self.results_hist = pd.DataFrame(results, index=[0])
        else:
            self.results_hist.loc[epoch] = results
        self.results_hist.to_csv(results_path, index=False)


        if (self.args['tr_checkpoint'] == 'every_epoch') or (self.args['tr_checkpoint'] == 'best_only' and best):
      
            if hasattr(model, 'module'):
                state_dict = model.module.state_dict()
                model_name = model.module.name
            else:
                state_dict = model.state_dict()
                model_name = model.name
    
            torch_dict = {
                'runname': self.runname,
                'epoch': epoch+1,
                'model_args': model_args,
                'args': self.args,
                'model_state_dict': state_dict,
                'optimizer_state_dict': opt.state_dict(),
                'db_results': db_results,
                'results': results,
                'model_name': model_name,
                }
            
            torch.save(torch_dict, model_path)
            
        elif (self.args['tr_checkpoint']!='every_epoch') and (self.args['tr_checkpoint']!='best_only'):
            raise ValueError('selected tr_checkpoint option not available')

def calculate_audio_quality_scores(data):
    try:
        # Convert columns to numeric type
        cols_to_convert = ['mos_pred', 'noi_pred', 'dis_pred', 'col_pred', 'loud_pred', 'word_error_rate']
        for col in cols_to_convert:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Define the weights for each metric
        weights = {
            'mos_pred': 0.28,  
            'noi_pred': 0.08,
            'dis_pred': 0.08,
            'col_pred': 0.08,
            'loud_pred': 0.08,
            'word_error_rate': 0.40  
        }

        # Normalize and invert scores as necessary
        data['mos_pred'] = (data['mos_pred'] - 1) / 4  # 1 worst, 5 best
        data['loud_pred'] = (data['loud_pred'] - 1) / 4  # 1 worst, 5 best
        data['noi_pred'] = (data['noi_pred'] - 1) / 4  # 1 worst, 5 best
        data['dis_pred'] = (data['dis_pred'] - 1) / 4  # 1 worst, 5 best
        data['col_pred'] = (data['col_pred'] - 1) / 4  # 1 worst, 5 best
        data['word_error_rate'] = 1 - (data['word_error_rate']) / 100  # 0 best, 100 worst

        # Calculate composite score
        data['composite_score'] = 0
        for metric, weight in weights.items():
            data['composite_score'] += data[metric] * weight

        # Ensure the final score is within 0-1
        data['composite_score'] = data['composite_score'].clip(0, 1)

        # Round the composite score to 3 decimal places
        data['composite_score'] = data['composite_score'].round(3)

        return data['composite_score'][0]

    except Exception as e:
        print(f"An error occurred: {e}")
        # Handle the error or return a default value
        return None


def score(file, text) -> float:
    """
    Reward the miner response to the dummy request. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - float: The reward value for the miner.
    """
    # pick the filename and file path from the file arguement 
    # filename = file.split('/')[-1] 
    # filepath = os.path.abspath(file)
    # filepath = os.path.dirname(file)
    # bt.logging.info(f"_____________________________ File name _____________________________: {filename}")
    # bt.logging.info(f"_____________________________ File Path _____________________________: {filepath}")
    mode = 'predict_file'  # or 'predict_dir', 'predict_csv'
    pretrained_model = 'nisqa.tar'  # e.g., 'model.pth'
    deg = file  # e.g., 'test.wav'
    # data_dir = filepath  # e.g., 'data/'
    output_dir = './'  # e.g., 'results/'
    csv_file = 'results.csv'  # e.g., 'data.csv'
    csv_deg = 'column_in_csv_with_files_name'  # e.g., 'filename'
    num_workers = 5
    bs = 1
    ms_channel = None  # or the specific channel if stereo file
    # Create a dictionary with the parameters
    args = {
        'mode': mode,
        'pretrained_model': pretrained_model,
        'deg': deg,
        # 'data_dir': data_dir,
        'output_dir': output_dir,
        'csv_file': csv_file,
        'csv_deg': csv_deg,
        'num_workers': num_workers,
        'bs': bs,
        'ms_channel': ms_channel,
        'tr_bs_val': bs,
        'tr_num_workers': num_workers
    }
    # Word Error Rate prediction
    wer = SpeechToTextEvaluator()
    word_error_rate = wer.evaluate_wer(file, text)
    print("Word Error Rate is:  ", word_error_rate)
    # Instantiate the nisqaModel with hardcoded arguments
    nisqa = nisqaModel(args)
    # Print the device of the model
    print(" The parameters Device of the NISQA MODEL:  ", next(nisqa.model.parameters()).device)
    # Execute the prediction directly
    nisqa.predict()
    data = pd.read_csv('NISQA_results.csv')
    # Include WER in the 'NISQA_results.csv' file
    data['word_error_rate'] = word_error_rate
    data.to_csv('NISQA_results.csv', index=False)

    # Return the result from calculate_audio_quality_scores
    return calculate_audio_quality_scores(data)