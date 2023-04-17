import torch
import torch.nn as nn
import pandas as pd 
import numpy as np

class ARIMA_LSTM(nn.Module):
    #encoder，输入自变量和协变量的历史数据
    #input_size = covariate_size+1
    def __init__(self,
                 horizon_size:int,#预测步长
                 input_size:int,
                 output_size:int,
                 hidden_size:int,
                 dropout:int,
                 layer_size:int,
                 device):
        super(ARIMA_LSTM,self).__init__()
        self.horizon_size =horizon_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.dropout = dropout
        self.linear1 = nn.Linear(in_features= input_size,
                                 out_features= input_size*2)
        self.linear2 = nn.Linear(in_features= hidden_size+1,
                                 out_features= output_size*horizon_size)
        self.linear3 = nn.Linear(in_features= output_size*horizon_size,
                                 out_features= output_size*horizon_size)
        self.activation = nn.Tanh()                         
        self.LSTM = nn.LSTM(input_size= input_size*2,
                            hidden_size= hidden_size,
                            num_layers= layer_size,
                            dropout= dropout)
        self.linear1.double()
        self.linear2.double()
        self.linear3.double()
        self.activation.double()
        self.LSTM.double()
        self.apply(custom_weights_init)


    def forward(self,input,seasonal):
        input = input.permute(1,0,2)
        seasonal = seasonal.permute(1,0,2)
        encoder_input = self.linear1(input)
        encoder_input = self.activation(encoder_input)
        encoder_output,_ = self.LSTM(encoder_input)
        decoder_input = torch.concat([encoder_output[-1,:,:],seasonal[-1,:,:]],dim=1)
        decoder_output = self.linear2(decoder_input)
        decoder_output = self.activation(decoder_output)
        output = self.linear3(decoder_output)
        # output = self.activation(output)
        return output
    
    def predict(self,input,seasonal):
        with torch.no_grad():
            input = input.permute(1,0,2)
            seasonal = seasonal.permute(1,0,2)
            encoder_input = self.linear1(input)
            encoder_input = self.activation(encoder_input)
            encoder_output,_ = self.LSTM(encoder_input)
            decoder_input = torch.concat([encoder_output[-1,:,:],seasonal[-1,:,:]],dim=1)
            decoder_output = self.linear2(decoder_input)
            decoder_output = self.activation(decoder_output)
            output = self.linear3(decoder_output)
            # output = self.activation(output)
        return output
    
def custom_weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
