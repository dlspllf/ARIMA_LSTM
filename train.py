import torch
from torch import nn
from Model import ARIMA_LSTM
from torch.utils.data import DataLoader
from data import ARIMA_LSTM_dataset
import matplotlib.pyplot as plt
import numpy as np

def train(net:ARIMA_LSTM,
          dataset:ARIMA_LSTM_dataset,
          lr:float,
          batch_size:int,
          num_epochs:int,
          device):
    ARIMA_LSTM_optimizer = torch.optim.Adam(net.parameters(),lr=lr)

    data_iter = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,num_workers=0)
    Loss_FN = nn.MSELoss()
    LOSS = []
    for i in range(num_epochs):
        for (input_series,seasonal,real_vals) in data_iter:
            ARIMA_LSTM_optimizer.zero_grad()
            output = net(input_series, seasonal)
            loss = Loss_FN(output,real_vals)
            loss.backward()
            ARIMA_LSTM_optimizer.step()
        LOSS.append(loss.item())
        if (i+1)%5 == 0:
            print(f"epoch_num {i+1}, current loss is: {loss.item()}")
    LOSS = np.array(LOSS)
    plt.plot(np.arange(0,num_epochs),LOSS, color = '#1f77b4', label='LOSS')