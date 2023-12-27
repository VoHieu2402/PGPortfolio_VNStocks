from collections import namedtuple
import random

# Define the replay buffer
Transition = namedtuple(
    'Transition',
    (
      "state_portfolio", "action", "reward", "next_state_portfolio",
      "state_benchmark", "next_state_benchmark",
      "prev_action", "prev_pf", "prev_bm", "pre_each_asset"
    )
)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def reset(self):
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, lag_length):
        # print(len(self.memory))
        idx = random.randint(0, len(self.memory) - batch_size - lag_length)
        return self.memory[idx : idx+batch_size+lag_length]

    def __len__(self):
        return len(self.memory)

