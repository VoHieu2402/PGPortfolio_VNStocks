import torch
import torch.nn as nn

class ActorNetwork(nn.Module):
  def __init__(self, num_features: int=3, num_stocks: int=14, lags: int=30, out_cv1: int=2, out_cv2: int=30, kernel_size_cv1: int=3, dropout_rate: float=0.2):
    super().__init__()
    self.num_features = num_features
    self.num_stocks = num_stocks
    self.lags = lags
    self.conv1 = nn.Conv2d(
        in_channels=num_features, out_channels=out_cv1, kernel_size=(1,kernel_size_cv1)
    )
    self.conv2 = nn.Conv2d(
        in_channels=out_cv1, out_channels=out_cv2, kernel_size=(1,lags-kernel_size_cv1+1)
    )
    self.conv3 = nn.Conv2d(
        in_channels=out_cv2+1, out_channels=1, kernel_size=(1,1)
    )
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=2)
    self.batchnom1 = nn.BatchNorm2d(out_cv1)
    self.batchnom2 = nn.BatchNorm2d(out_cv2+1)
    self.dropout = nn.Dropout(dropout_rate)
    self.cash_bias = nn.Parameter(torch.zeros(1))


  def forward(self, x, prev_weight):
    outputs = self.conv1(x)
    outputs = self.relu(outputs)
    outputs = self.batchnom1(outputs)
    outputs = self.dropout(outputs)

    outputs = self.conv2(outputs)
    outputs = self.relu(outputs)
    outputs = torch.cat((outputs,prev_weight.unsqueeze(1)),1)
    outputs = self.batchnom2(outputs)
    outputs = self.dropout(outputs)

    outputs = self.conv3(outputs)
    cash_bias = self.cash_bias.expand(outputs.size(0), -1).unsqueeze(2).unsqueeze(3)
    outputs = torch.cat((outputs,cash_bias),2)
    outputs = self.softmax(outputs)

    return outputs.squeeze(1)[:,:-1,:], outputs.squeeze(1)

