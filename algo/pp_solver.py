import heapq

import numpy as np
import torch

from algo.model import DQNEmbedding as DQN


max_step = 100


class DijkstraPathPlanner:
    def __init__(self, graph):
        self.graph = graph

    def run(self, start_node, end_node, weight_key='dynamic_weight'):
        heap = [(0, start_node, [start_node])]
        visited = set()

        while heap:
            cost, current, path = heapq.heappop(heap)
            if current == end_node: return path, cost, True
            if current in visited: continue
            visited.add(current)

            for neighbor in self.graph.neighbors(current):
                if neighbor in visited: continue

                weight = self.graph[current][neighbor][0][weight_key]
                heapq.heappush(heap, (cost + weight, neighbor, path + [neighbor]))
        return [], float('inf'), False  # 如果没有路径


class RRTStarPathPlanner:
    pass


class RLStaticPathPlanner:
    def __init__(self, graph, p, num_action):
        self.p = p
        self.graph = graph
        self.nodes = list(self.graph.nodes)

        self.cur_node_idx = None
        self.goal_node_idx = None

        state_dim = len(self.nodes)
        self.pi_opp = DQN(input_dim=state_dim, output_dim=num_action)
        state_dict = torch.load('../trained/model_dpp_static.pth')
        self.pi_opp.load_state_dict(state_dict)

    def reset(self, u, v):
        self.cur_node_idx = self.nodes.index(u)
        self.goal_node_idx = self.nodes.index(v)
        return self.__get_state()

    def __get_state(self):
        return [self.cur_node_idx, self.goal_node_idx]

    def choose_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.int32).unsqueeze(0)
        q_values_tensor = self.pi_opp(state_tensor)
        q_values = q_values_tensor.squeeze(0).detach().numpy()
        indexes = np.argwhere(q_values == q_values.max()).squeeze(axis=1)
        return np.random.choice(indexes)

    def step(self, action):
        current_node = self.nodes[self.cur_node_idx]
        next_nodes = self.p[current_node]
        if action >= len(next_nodes):
            return self.__get_state(), -10.0, False

        next_node = next_nodes[action]
        next_node_idx = self.nodes.index(next_node)
        done = (next_node_idx == self.goal_node_idx)
        reward = -self.graph[current_node][next_node][0]['dynamic_weight']

        self.cur_node_idx = next_node_idx
        return self.__get_state(), reward, done

    def run(self, u, v):
        done, state = False, self.reset(u, v)
        cost, episode_step = 0.0, 0
        path = [u, ]
        while not done:
            action = self.choose_action(state)
            next_state, reward, done, *_ = self.step(action)
            path.append(self.nodes[self.cur_node_idx])
            state = next_state
            cost += reward
            episode_step += 1
            if episode_step >= max_step: break
        return path, -cost, done


class RLDynamicPathPlanner:
    def __init__(self, graph, p, num_action):
        self.p = p
        self.graph = graph
        self.nodes = list(self.graph.nodes)

        self.cur_node_idx = None
        self.goal_node_idx = None

        state_dim = len(self.nodes)
        self.pi_opp = DQN(input_dim=state_dim, output_dim=num_action)
        state_dict = torch.load('../trained/model_dpp_gcn.pth')
        self.pi_opp.load_state_dict(state_dict)

    def reset(self, u, v):
        self.cur_node_idx = self.nodes.index(u)
        self.goal_node_idx = self.nodes.index(v)
        print('\t Scenario Info: ', self.cur_node_idx, self.goal_node_idx)
        return self.__get_state()

    def __get_state(self):
        return [self.cur_node_idx, self.goal_node_idx]

    def choose_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.int32).unsqueeze(0)
        q_values_tensor = self.pi_opp(state_tensor)
        q_values = q_values_tensor.squeeze(0).detach().numpy()
        indexes = np.argwhere(q_values == q_values.max()).squeeze(axis=1)
        return np.random.choice(indexes)

    def step(self, action):
        current_node = self.nodes[self.cur_node_idx]
        next_nodes = self.p[current_node]
        if action >= len(next_nodes):
            return self.__get_state(), -10.0, False

        next_node = next_nodes[action]
        next_node_idx = self.nodes.index(next_node)
        done = (next_node_idx == self.goal_node_idx)
        reward = -self.graph[current_node][next_node][0]['dynamic_weight']

        self.cur_node_idx = next_node_idx
        return self.__get_state(), reward, done

    def run(self, u, v):
        done, state = False, self.reset(u, v)
        cost, episode_step = 0.0, 0
        path = [u, ]
        while not done:
            action = self.choose_action(state)
            next_state, reward, done, *_ = self.step(action)
            path.append(self.nodes[self.cur_node_idx])
            state = next_state
            cost += reward
            episode_step += 1
            if episode_step >= max_step: break
        return path, 1e2 - cost, done


