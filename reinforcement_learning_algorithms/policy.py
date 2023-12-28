import torch.nn as nn
import os
import sys

# Add the project directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deep_learning_model.actor_network import ActorNetwork

class Policy(nn.Module):
    def __init__(self, num_features: int=3, num_stocks: int=5, lags: int=30, out_cv1: int=2, out_cv2: int=30, kernel_size_cv1: int=3, dropout_rate: float=0.2):
        super().__init__()
        self.actor_network = ActorNetwork(
            num_features = num_features,
            num_stocks = num_stocks,
            lags = lags,
            out_cv1 = out_cv1, out_cv2 = out_cv2,
            kernel_size_cv1 = kernel_size_cv1, dropout_rate = dropout_rate
        )
        self.stock_weights = None
        self.portfolio_weights = None

    def select_action(self, x1, x2):
        self.stock_weights, self.portfolio_weights = self.actor_network(x1, x2)
        return self.stock_weights, self.portfolio_weights
    
