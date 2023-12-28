# A DRL Framework for Portfolio Management: Application for Vietnamese stocks

This project draws inspiration from the deep reinforcement learning framework for portfolio management proposed by [Jiang et al. in 2017](https://arxiv.org/abs/1706.10059). I conduct further investigation into the original architecture introduced by Jiang et al, tailoring it for implementation in the Vietnamese stock market. The portfolio encompasses 14 distinct stocks, detailed in below table. The objective is to formulate a reward function that maximizes the risk-adjusted return of the portfolio relative to the benchmark (VNI - VN Index)

| Num | Ticker | Description
| --- | --- | --- |
| 1 | ACB | Asia Commercial Joint Stock Bank
| 2 | BID | Joint Stock Commercial Bank for Investment and Development of Vietnam (BIDV)
| 3 | BVH | BaoViet Holding
| 4 | CTG | Vietnam Joint Stock Commercial Bank for Industry and Trade (VietinBank)
| 5 | FPT | FPT Corporation
| 6 | HPG | Hoa Phat Group
| 7 | MBB | Military Commercial Joint Stock Bank
| 8 | MSN | Masan Group
| 9 | MWG | Mobile World Investment Corporation (The Gioi Di Dong)
| 10 | SSI | SSI Securities Corporation
| 11 | STB | Sai Gon Thuong Tin Joint Stock Commercial Bank (Sacombank)
| 12 | VCB | Joint Stock Commercial Bank for Foreign Trade of Vietnam (Vietcombank)
| 13 | VIC | Vingroup Joint Stock Company
| 14 | VNM | Vinamilk


## Usage Guideline

To begin, ensure the installation of the necessary packages outlined in the <i>requirements.txt</i> file. The focal point of my repository revolves around the training and testing processes. The training process is executed through the <i>train.py</i> file. If required, you have the flexibility to modify hyperparameters for the agent and training process; however, it is imperative to have a clear understanding of each parameter. No validation function is implemented to assess the suitability of hyperparameters, so errors may arise if the provided parameters are inappropriate. Subsequent to the training phase, the trained agent is saved as a pickle file. The <i>test.py</i> file is designed to load this pickle file for testing purposes, accompanied by visualizations of the performance.

### Quick Start
Start with following commands:
```
$ git clone https://github.com/VoHieu2402/PGPortfolio_VNStocks
$ cd PGPortfolio_VNStocks
$ python train.py
$ python test.py
```

### File Structure

- <b>data/torch_tensor_vn_stocks</b>: The training dataset spans from January 1, 2015, to June 1, 2023, while the testing dataset covers the period from June 1, 2023, to December 19, 2023. Historical Close-High-Low (CHL) data serves as the representation of the market state in our approach. The chosen lag is 30 previous time steps (equivalent to one month)
    - <b>state_tensor_pf_vnstocks_train.pt</b>: A PyTorch tensor that holds information about the state of 14 different stocks for training. It undergoes processing as described in the original paper. It has shape: <i>(batch_size, num_features, num_stocks, num_lags)</i>
    - <b>state_tensor_pf_vnstocks_test.pt</b>: A PyTorch tensor that holds information about the state of 14 different stocks for testing. It undergoes processing as described in the original paper. It has shape: <i>(batch_size, num_features, num_stocks, num_lags)</i>
    - <b>state_tensor_pf_VNI_train.pt</b>: A PyTorch tensor that holds information about the state of the benchmark (VN Index) for training. It undergoes processing as described in the original paper. It has shape: <i>(batch_size, num_features, 1, num_lags)</i>
    - <b>state_tensor_pf_VNI_test.pt</b>: A PyTorch tensor that holds information about the state of the benchmark (VN Index) for testing. It undergoes processing as described in the original paper. It has shape: <i>(batch_size, num_features, 1, num_lags)</i>
- <b>deep_learning_model</b>:
    - <b>actor_network.py</b>: The deep neural network that determines the allocation directly based on the state tensor. The architecture of the network is described in the original paper.
- <b>reinforcement_learning_algorithms</b>:
    - <b>replay_buffer.py</b>: The database used to stores experiences in the form of tuples <i>(state_portfolio, action, reward, next_state_portfolio, state_benchmark, next_state_benchmark, prev_action, prev_pf, prev_bm, pre_each_asset)</i>, representing the agent's interactions with the environment at different time steps.
    - <b>policy.py</b>: The policy that select actions using actor network.
    - <b>agent.py</b>: An agent that interacts with an environment with the goal of learning optimal actions to maximize cumulative rewards over time. It is responsible for making decisions, taking actions, and learning from the consequences of those actions.
- <b>train.py</b> During the training process, the agent undergoes 1000 episodes, each involving the management of a portfolio comprising 14 distinct stocks and cash to maximize the final risk-adjusted return. In each episode, there are 60 time steps, equivalent to a period of 2 months.
- <b>test.py</b> The tested agent evaluates the portfolio management strategy over a period comprising 112 time steps. Additionally, visualizations are generated in this file.


## New contributions to the original architecture

### Reward function

While the original framework calculates the reward as the explicit average of periodic logarithmic returns, my project defines the reward function as the disparity between the agent's risk-adjusted return and that of the benchmark. This addition of a risk element aims to enhance the stability of the portfolio management strategy, preventing excessive allocation to specific assets.

Moreover, employing risk-adjusted returns allows us to track the value of the portfolio over time. Integrating the portfolio value into the model enhances the agent's awareness of its position, facilitating more informed decision-making.

### Learning rate schedules

Instead of utilizing a constant learning rate, I utilize learning rate schedules to reduce the learning rate after each episode. This method is believed to improve the optimization process, particularly in the context of risk-adjusted returns, where daily returns are often very small.

## Performance and some discussion

The policy function is designed through a deep neural network which takes as input the input tensor (shape m x 50 x (3 or 4)) composed of :
- the m traded stocks 
- the 3/4 matrix columns (processed OHLC)
- 5O previous time steps

A first convolution is realized resulting in a smaller tensor. Then, a second convolution is made resulting in 20 vector of shape (m x 1 x 1). The previous output vector is stacked. 
The last layer is a terminate convolution resulting in a unique m vector. 
Then, a cash bias is added and a softmax applied. 

The output of the neural network is the vector of the actions the agent will take. 

Then, the environment can compute the new vector of weights, the new portfolio and instant reward.

![DLArchiteture](./print/DLArchiteture.png)

## Training and Testing of the agent

This part is still in progress as of today. Our thought is we are still not able to reproduce the paper's results. 
Indeed, even if the algorithm demonstrated the capacity to identify high-potential stocks which maximizes results. However, it has a little potential to change the position through the trading process. 

![results](./print/results.png)


## Understanding the problem & possible improvement

We tried many initial parameters such as low trading cost to produce incentive to change of position. 

The agent is 'training sensitive' but it is not 'input state sensitive'. 

In order to make the policy more dynamic, we think of using a discrete action space using pre-defined return thresholds. We'll turn the problem replacing the softmax by a tanh or by turning it into a classification task. 

![results2](./print/result2.png)

## Author

* **Vo Minh Hieu** [Hieu Vo](https://www.linkedin.com/in/hieu-vo-897a12158/)





