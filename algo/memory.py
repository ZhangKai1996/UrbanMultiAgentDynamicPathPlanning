import random
from collections import deque, namedtuple

import numpy as np

class SimpleReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class ScalableReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = {}
        self.counter = 0

    def push(self, *args, key=None):
        if key is None: key = 0

        if key not in self.buffer.keys():
            self.buffer[key] = deque(maxlen=self.capacity)
        self.buffer[key].append(args)
        self.counter += 1

    def sample(self, batch_size, num_keys=None):
        keys = list(self.buffer.keys())
        if num_keys is not None:
            keys = np.random.choice(keys, num_keys)

        batch_dict = {}
        for key in keys:
            value = self.buffer[key]
            if len(value) >= batch_size:
                batch_dict[key] = random.sample(value, batch_size)
        return batch_dict

    def sample_(self, batch_size, num_keys=None):
        batch, count = [], 0
        for key, value in self.buffer.items():
            if len(value) >= batch_size:
                batch += random.sample(self.buffer[key], batch_size)
                count += 1
            if count >= num_keys: break
        return batch

    def __len__(self):
        return self.counter


class RolloutBuffer:
    def __init__(self):
        self.states, self.masks = [], []
        self.actions, self.log_probs, self.values = [], [], []
        self.rewards, self.dones = [], []
        self.next_states, self.next_masks = [], []

    def store(self, state, mask,
              action, log_prob,
              reward, done, value,
              next_state=None, next_mask=None):
        self.states.append(state)
        self.masks.append(mask)

        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)

        self.rewards.append(reward)
        self.dones.append(done)

        if next_state is not None: self.next_states.append(next_state)
        if next_mask is not None: self.next_masks.append(next_mask)

    def clear(self):
        self.states, self.masks = [], []
        self.actions, self.log_probs, self.values = [], [], []
        self.rewards, self.dones = [], []
        self.next_states, self.next_masks = [], []


Experience = namedtuple('Experience', ('states', 'actions', 'next_states', 'rewards', 'dones'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)