import torch
import torch.nn as nn
import torch.optim as optim
from reinforcement_learning_algorithms.policy import Policy
from reinforcement_learning_algorithms.replay_buffer import ReplayBuffer

class Agent(nn.Module):
  """
      "state_portfolio" -> (batch_size, num_features, num_stocks, lags)
      "action_stocks" -> (batch_size, num_stocks, 1)
      "action_pf" -> (batch_size, num_stocks+1, 1)
      "reward" -> (batch_size, num_stocks, 1)
      "next_state_portfolio" -> (batch_size, num_features, num_stocks, lags)
      "state_benchmark" -> (batch_size, num_features, 1, lags)
      "next_state_benchmark" -> (batch_size, num_features, 1, lags)
      "prev_action_stocks" -> (batch_size, num_stocks, 1)
      "prev_pf" -> (batch_size, 1)
      "prev_pf_for_loss" -> (batch_size, lags_for_loss)
      "prev_action_pf" -> (batch_size, num_stocks+1, 1)
      "prev_bm" -> (batch_size, 1)
      "prev_bm_for_loss" -> (batch_size, lags_for_loss)
      "pre_each_asset" -> (batch_size, num_stocks+1)
  """
  def __init__(self, num_features: int=3, num_stocks: int=5, lags: int=30, out_cv1: int=2, out_cv2: int=30, kernel_size_cv1: int=3, dropout_rate: float=0.2, lr_actor: float=0.001, rb_capacity: int=100, batch_size: int=30, gamma: float=0.9, lags_for_loss: int=48, trans_cost: float=0.0025):
    super().__init__()
    self.num_features = num_features
    self.num_stocks = num_stocks
    self.lags = lags
    self.out_cv1 = out_cv1
    self.out_cv2 = out_cv2
    self.kernel_size_cv1 = kernel_size_cv1
    self.dropout_rate = dropout_rate
    self.rb_capacity = rb_capacity
    self.batch_size = batch_size
    self.lags_for_loss = lags_for_loss
    self.gamma = gamma
    self.trans_cost = trans_cost

    self.policy = Policy(
            num_features = num_features,
            num_stocks = num_stocks,
            lags = lags,
            out_cv1 = out_cv1, out_cv2 = out_cv2,
            kernel_size_cv1 = kernel_size_cv1, dropout_rate = dropout_rate,
    )
    self.actor_network = self.policy.actor_network
    self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=lr_actor)
    self.lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.actor_optimizer, gamma=gamma)
    self.replay_buffer = ReplayBuffer(rb_capacity)


  def train(self, update_lr=False, print_result=False):
    if len(self.replay_buffer) < self.batch_size + self.lags_for_loss:
          return
    transitions = self.replay_buffer.sample(self.batch_size, self.lags_for_loss)

    state_portfolio = torch.zeros(
        self.batch_size+self.lags_for_loss, self.num_features, self.num_stocks, self.lags, dtype=torch.float32
    )
    next_state_portfolio = torch.zeros(
        self.batch_size+self.lags_for_loss, self.num_features, self.num_stocks, self.lags, dtype=torch.float32
    )
    action = torch.zeros(
        self.batch_size+self.lags_for_loss, self.num_stocks, 1, dtype=torch.float32
    )
    reward = torch.zeros(
        self.batch_size+self.lags_for_loss, self.num_stocks, 1, dtype=torch.float32
    )
    prev_action = torch.zeros(
        self.batch_size+self.lags_for_loss, self.num_stocks, 1, dtype=torch.float32
    )
    state_benchmark = torch.zeros(
        self.batch_size+self.lags_for_loss, self.num_features, 1, self.lags, dtype=torch.float32
    )
    next_state_benchmark = torch.zeros(
        self.batch_size+self.lags_for_loss, self.num_features, 1, self.lags, dtype=torch.float32
    )
    prev_pf = torch.zeros(
        self.batch_size+self.lags_for_loss, 1, dtype=torch.float32
    )
    prev_pf_for_loss = torch.zeros(
        self.batch_size, self.lags_for_loss-1, dtype=torch.float32
    )
    prev_bm = torch.zeros(
        self.batch_size+self.lags_for_loss, 1, dtype=torch.float32
    )
    prev_bm_for_loss = torch.zeros(
        self.batch_size, self.lags_for_loss-1, dtype=torch.float32
    )
    pre_each_asset = torch.zeros(
        self.batch_size+self.lags_for_loss, self.num_stocks+1, dtype=torch.float32
    )

    for i in range(self.batch_size+self.lags_for_loss):
      state_portfolio[i] = transitions[i].state_portfolio
      next_state_portfolio[i] = transitions[i].next_state_portfolio
      action[i] = transitions[i].action
      reward[i] = transitions[i].reward
      prev_action[i] = transitions[i].prev_action
      state_benchmark[i] = transitions[i].state_benchmark
      next_state_benchmark[i] = transitions[i].next_state_benchmark
      prev_pf[i] = transitions[i].prev_pf
      prev_bm[i] = transitions[i].prev_bm
      pre_each_asset[i] = transitions[i].pre_each_asset

    state_portfolio = state_portfolio[self.lags_for_loss:].clone().detach().requires_grad_(False)
    next_state_portfolio = next_state_portfolio[self.lags_for_loss:].clone().detach().requires_grad_(False)
    action = action[self.lags_for_loss:].clone().detach().requires_grad_(False)
    reward = reward[self.lags_for_loss:].clone().detach().requires_grad_(False)
    prev_action = prev_action[self.lags_for_loss:].clone().detach().requires_grad_(False)
    state_benchmark = state_benchmark[self.lags_for_loss:].clone().detach().requires_grad_(False)
    next_state_benchmark = next_state_benchmark[self.lags_for_loss:].clone().detach().requires_grad_(False)
    pre_each_asset = pre_each_asset[self.lags_for_loss:].clone().detach().requires_grad_(False)
    

    # Calculate the historical ret for portfolio
    prev_pf_ = prev_pf[self.lags_for_loss:].clone().detach().requires_grad_(False)
    whole_prev_pf = prev_pf.clone().detach().requires_grad_(False)
    whole_prev_pf_yesterday = torch.roll(whole_prev_pf, shifts=1, dims=0)[2:]
    prev_return_pf = (whole_prev_pf[2:]-whole_prev_pf_yesterday) / whole_prev_pf_yesterday
    for n in range(self.batch_size):
      for l in range(self.lags_for_loss-1):
        prev_pf_for_loss[n,l] = prev_return_pf[n+l]
        
    # Calculate the historical ret for benchmark
    prev_bm_ = prev_bm[self.lags_for_loss:].clone().detach().requires_grad_(False)
    whole_prev_bm = prev_bm.clone().detach().requires_grad_(False)
    whole_prev_bm_yesterday = torch.roll(whole_prev_bm, shifts=1, dims=0)[2:]
    prev_return_bm = (whole_prev_bm[2:]-whole_prev_bm_yesterday) / whole_prev_bm_yesterday
    for n in range(self.batch_size):
      for l in range(self.lags_for_loss-1):
        prev_bm_for_loss[n,l] = prev_return_bm[n+l]


    # Calculate action
    action_stocks, action_portfolio = self.policy.select_action(state_portfolio, prev_action)


    # Re-calculate portfolio after new allocation
    post_each_asset = action_portfolio.squeeze(2) * prev_pf_
    transaction_amount = post_each_asset - pre_each_asset
    transaction_cost_for_each = torch.abs(transaction_amount) * 0.0025
    post_each_asset[:,-1] -= torch.sum(transaction_cost_for_each, 1) - transaction_cost_for_each[:,-1]
    new_prev_pf_ = torch.sum(post_each_asset, 1)
    prev_cash = post_each_asset[:,-1].unsqueeze(1)
    prev_stocks = new_prev_pf_ - prev_cash

    # Calculate the reward of the action - Daily return
    ret = 1 / next_state_portfolio[:,0,:,-2].unsqueeze(2)
    tot_ret = ret * post_each_asset[:,:self.num_stocks].unsqueeze(-1)
    new_pf = torch.sum(tot_ret.squeeze(-1),-1).unsqueeze(-1) + post_each_asset[:,-1].unsqueeze(-1)
    ret_pf = (new_pf - new_prev_pf_.unsqueeze(-1)) / new_prev_pf_.unsqueeze(-1)
    ret_bench = 1 / next_state_benchmark[:,0,:,-2] - 1    # Calculate the daily return of the benchmark

    # Calculate historical mean and std with the addition of new reward (used for Sharpe ratio calculation)
    complete_ret_pf = torch.cat((prev_pf_for_loss, ret_pf), 1)
    mean_ret_pf = torch.mean(complete_ret_pf, 1)
    std_ret_pf = torch.abs(torch.std(complete_ret_pf, 1))
    
    complete_ret_bm = torch.cat((prev_bm_for_loss, ret_bench), 1)
    mean_ret_bm = torch.mean(complete_ret_bm, 1)
    std_ret_bm  = torch.abs(torch.std(complete_ret_bm, 1))

    # Calculate the loss
    diff_sharpe = torch.mean((mean_ret_pf / std_ret_pf) - (mean_ret_bm / std_ret_bm))
    diff_ret = torch.log(torch.prod((ret_pf+1).squeeze(-1))) - torch.log(torch.prod((ret_bench+1).squeeze(-1)))
    actor_loss = -(0.3*diff_ret+0.7*diff_sharpe)

    # Print the actor loss if needed
    if print_result:
      print("Actor loss: ", actor_loss)

    # Update Networks
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()
    
    # Update learning rate if needed
    if update_lr:
      self.lr_scheduler.step()

    return actor_loss

