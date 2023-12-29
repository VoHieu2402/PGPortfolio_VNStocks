import torch
import numpy as np
import random
import pickle
from reinforcement_learning_algorithms.agent import Agent

state_tensor_pf_vnstocks_train = torch.load("data/torch_tensor_vnstocks/state_tensor_pf_vnstocks_train.pt")
state_tensor_VNI_train = torch.load("data/torch_tensor_vnstocks/state_tensor_VNI_train.pt")

agent = Agent(
    batch_size=30,
    lr_actor=0.00001,
    num_stocks=14,
    lags_for_loss=14,
    trans_cost=0.001,
    rb_capacity=100 # Must be greater than batch_size + lags_for_loss
)
lst_avg_action = []
lst_total_reward = []
lst_total_balance = []
max_reward = 0
best_model = None
num_win = 0
num_loss = 0

for episode in range(20):
  # idx_episode = random.randint(0,len(state_tensor_pf_vnstocks_train)-60)
  lst_actions = []
  # state_tensor_pf = state_tensor_pf_vnstocks_train[idx_episode:idx_episode+61]
  # state_tensor_bm = state_tensor_VNI_train[idx_episode:idx_episode+61]
  state_tensor_pf = state_tensor_pf_vnstocks_train
  state_tensor_bm = state_tensor_VNI_train
  total_reward = 1
  done = 0
  update_lr = False
  print_result = False
  balance = 1000
  lst_portfolio_returns = []
  lst_bm_returns = []
  prev_w = torch.zeros(1, state_tensor_pf.shape[2], 1, requires_grad=False, dtype=torch.float32)
  pre_each_value = torch.zeros(1, state_tensor_pf.shape[2]+1, requires_grad=False, dtype=torch.float32)
  pre_each_value[:,-1] = balance
  prev_pf = np.array([[balance]])
  prev_bm = torch.from_numpy(prev_pf).clone().detach().requires_grad_(False)
  prev_pf = torch.from_numpy(prev_pf).clone().detach().requires_grad_(False)
  agent.replay_buffer.reset()
  
  for i in range(len(state_tensor_pf)-1): 
    # print(i)
    state_pf = state_tensor_pf[i].unsqueeze(0)
    state_bm = state_tensor_bm[i].unsqueeze(0)
    action_stocks, action_pf = agent.policy.select_action(state_pf, prev_w)
    next_state_pf = state_tensor_pf[i+1].unsqueeze(0)
    next_state_bm = state_tensor_bm[i+1].unsqueeze(0)

    # Calculate prev portfolio based on action
    post_each_value = action_pf.squeeze(2) * prev_pf
    transaction_amount = post_each_value - pre_each_value
    transaction_cost_for_each = torch.abs(transaction_amount) * agent.trans_cost
    transaction_cost_for_each[:,-1] = 0
    total_transaction_cost = torch.zeros(1, state_tensor_pf.shape[2]+1, requires_grad=False, dtype=torch.float32)
    total_transaction_cost[:,-1] = torch.sum(transaction_cost_for_each, 1).item()
    post_each_value = post_each_value - total_transaction_cost
    prev_pf = torch.sum(post_each_value, 1)
    prev_cash = post_each_value[:,-1].unsqueeze(0)
    prev_stocks = prev_pf - prev_cash
    action_stocks = (post_each_value / prev_pf)[:,:state_tensor_pf.shape[2]].unsqueeze(-1)

    ret = 1 / next_state_pf[:,0,:,-2].unsqueeze(2)
    tot_ret = ret * post_each_value[:,:state_tensor_pf.shape[2]].unsqueeze(-1)
    new_pf = torch.sum(tot_ret.squeeze(-1),-1) + post_each_value[:,-1].unsqueeze(-1)
    ret_pf = (new_pf - prev_pf) / prev_pf

    ret_bench = 1 / next_state_bm[:,0,:,-2]
    new_bm = prev_bm * ret_bench
    reward = ret_pf+1 - ret_bench
    balance = new_pf.clone().detach().requires_grad_(False).squeeze(0).numpy()
    prev_pf = new_pf
    prev_bm = new_bm
    pre_each_value = post_each_value
    lst_portfolio_returns.append(ret_pf.item())
    lst_bm_returns.append(ret_bench.item()-1)

    total_reward *= (1+reward.item())
    if i == len(state_tensor_pf)-2:
      print_result=True
      update_lr=True
    loss = agent.train(update_lr=update_lr, print_result=print_result)
    agent.replay_buffer.push(
      state_pf, action_stocks, reward, next_state_pf,
      state_bm, next_state_bm,
      prev_w, new_pf.squeeze(0), new_bm.squeeze(0), pre_each_value.squeeze(0)
    )
    prev_w = action_stocks.clone().detach().requires_grad_(False)

  # lst_avg_action.append(lst_actions)
  if balance.item() >= max_reward:
    max_reward = balance.item()
    best_model = agent.actor_network.state_dict()

  risk_adj_portfolio = (np.mean(lst_portfolio_returns) / np.abs(np.std(lst_portfolio_returns)))
  risk_adj_benchmark = (np.mean(lst_bm_returns) / np.abs(np.std(lst_bm_returns)))
  lst_total_reward.append(risk_adj_portfolio - risk_adj_benchmark)
  lst_total_balance.append(balance)
  if risk_adj_portfolio - risk_adj_benchmark >= 0:
    num_win += 1
  else:
    num_loss += 1
  print(f"Episode {episode + 1}, Total Reward: {risk_adj_portfolio - risk_adj_benchmark}, Total Balance: {balance.item()}, Benchmark: {new_bm.item()}")
  print(f"Num of win: {num_win}, Num of loss: {num_loss}")
  print("------------------------------------------------------")

# After training, load the best model
agent.actor_network.load_state_dict(best_model)

print(max_reward)

# Save the agent to a file using pickle
with open("my_agent1.pkl", "wb") as file:
    pickle.dump(agent, file)
