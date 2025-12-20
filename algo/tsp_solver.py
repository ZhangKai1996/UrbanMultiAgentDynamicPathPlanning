import random

import numpy as np
import torch

from algo.model import RNN

max_step = 100


class DPTSPSolver:
    def __init__(self, graph):
        self.nodes = list(graph.nodes)
        self.start = 0
        self.end = -1

        self.start_node_idx = None
        self.task_nodes_idx = None
        self.end_node_idx = None

    def reset(self, start, end, tasks):
        self.start_node_idx = start
        self.end_node_idx = end
        self.task_nodes_idx = tasks

    def get_all_nodes(self):
        return [self.start_node_idx, ] + self.task_nodes_idx + [self.end_node_idx, ]

    def run(self, matrix):
        start, end = self.start, self.end

        n = len(matrix)
        if end == -1: end = n - 1

        C = {}  # 状态集合：key=(mask, last_node), value=(cost, path)
        for k in range(n):
            if k == start or k == end:
                continue
            mask = (1 << start) | (1 << k)
            C[(mask, k)] = (matrix[start][k], [start, k])

        # 枚举子集大小
        for subset_size in range(3, n + 1):
            for mask in range(1 << n):
                if (mask & (1 << start)) == 0 or (mask & (1 << end)) != 0:
                    continue
                if bin(mask).count("1") != subset_size:
                    continue

                for u in range(n):
                    if u == start or u == end or not (mask & (1 << u)):
                        continue

                    prev_mask = mask ^ (1 << u)
                    min_cost = float('inf')
                    min_path = []

                    for v in range(n):
                        if v == start or v == end or v == u or not (prev_mask & (1 << v)):
                            continue
                        if (prev_mask, v) not in C:
                            continue
                        prev_cost, prev_path = C[(prev_mask, v)]
                        cost = prev_cost + matrix[v][u]
                        if cost < min_cost:
                            min_cost = cost
                            min_path = prev_path + [u]

                    if min_path:
                        C[(mask, u)] = (min_cost, min_path)

        # 枚举所有以某节点 u 结尾的路径，然后加上 u -> end 的边
        min_total_cost = float('inf')
        best_total_path = []
        final_mask = ((1 << n) - 1) ^ (1 << end)  # 所有点都访问，但 end 除外

        for u in range(n):
            if u == start or u == end:
                continue
            if (final_mask, u) not in C:
                continue
            prev_cost, prev_path = C[(final_mask, u)]
            total_cost = prev_cost + matrix[u][end]
            if total_cost < min_total_cost:
                min_total_cost = total_cost
                best_total_path = prev_path + [end]

        all_nodes = [self.nodes[idx] for idx in self.get_all_nodes()]
        path = [all_nodes[i] for i in best_total_path]
        return path, min_total_cost


class GeneticAlgorithmSolver:
    def __init__(self, dist_matrix, population_size=50, generations=100, mutation_rate=0.1):
        self.dist_matrix = dist_matrix          # 距离矩阵
        self.population_size = population_size  # 种群大小
        self.generations = generations          # 代数
        self.mutation_rate = mutation_rate      # 变异率
        self.num_nodes = len(dist_matrix)       # 节点数量

    def run(self, visiting_nodes, task_dict):
        print('\t Run the Generic Algorithm ...')
        n = self.num_nodes
        if n <= 3:
            best_individual = list(range(n))
            best_cost = sum(self.dist_matrix[n1][best_individual[i + 1]]
                            for i, n1 in enumerate(best_individual[:-1]))
            return best_individual, best_cost

        population = self.__generate_population()  # 初始种群

        best_individual = None
        best_fitness = float('-inf')
        for generation in range(self.generations):
            new_population = []
            # 精英策略：保存最好的个体
            for individual in population:
                fitness_value = self.__fitness(individual, visiting_nodes, task_dict)
                if fitness_value > best_fitness:
                    best_fitness = fitness_value
                    best_individual = individual
            # 生成新种群
            while len(new_population) < self.population_size:
                parent1, parent2 = self.__selection(population, visiting_nodes, task_dict)  # 选择两个父代
                child = self.__crossover(parent1, parent2)  # 交叉生成子代
                child = self.__mutate(child)  # 变异子代
                new_population.append(child)

            population = new_population  # 更新种群
        return best_individual, 1 / best_fitness  # 返回最优路径及其距离

    def __generate_population(self):
        n = self.num_nodes

        population = []
        for _ in range(self.population_size):
            individual = list(range(1, n-1))  # 初始路径从 0 到 n-1
            random.shuffle(individual)  # 随机打乱路径
            individual = [0, ] + individual + [n-1, ]
            population.append(individual)
        return population

    def __fitness(self, individual, visiting_nodes, task_dict):
        n = self.num_nodes

        new_visiting_nodes = [visiting_nodes[idx] for idx in individual]
        new_visiting_nodes = new_visiting_nodes[1:-1]

        is_available_individual = True
        for idx, node in enumerate(new_visiting_nodes):
            if node not in task_dict.keys(): continue

            task = task_dict[node]
            if task.is_picked(): continue
            p_idx = new_visiting_nodes.index(task.pickup_point)
            if p_idx > idx:
                is_available_individual = False

        times = 0.01 if is_available_individual else 1.0
        total_distance = 0
        for i in range(n-1):
            total_distance += self.dist_matrix[individual[i]][individual[i + 1]]
        return 1 / (total_distance * times) * 1000 # 适应度 = 1 / 路径总距离

    def __selection(self, population, visiting_nodes, task_dict):
        """
        选择操作：轮盘赌选择方法，根据适应度选择父代
        """
        fitness_values = [self.__fitness(ind, visiting_nodes, task_dict) for ind in population]
        total_fitness = sum(fitness_values)
        probabilities = [fit / total_fitness for fit in fitness_values]
        selected = random.choices(population, probabilities, k=2)  # 选择两个父代
        return selected

    def __crossover(self, parent1, parent2):
        """
        交叉操作：部分匹配交叉（PMX）生成新个体
        """
        n = self.num_nodes

        point1, point2 = sorted(random.sample(range(1, n-1), 2))  # 随机选择交叉点
        # 创建子代
        child = [None] * len(parent1)
        # 复制父代的交叉部分
        child[point1:point2 + 1] = parent1[point1:point2 + 1]
        # 填充其余部分
        for i in range(len(parent2)):
            if parent2[i] not in child:
                for j in range(len(child)):
                    if child[j] is None:
                        child[j] = parent2[i]
                        break
        return child

    def __mutate(self, individual):
        """
        变异操作：随机交换路径中的两个节点
        """
        n = self.num_nodes

        if random.random() < self.mutation_rate:
            i, j = random.sample(range(1, n-1), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual


class RLStaticTSPSolver:
    def __init__(self, graph):
        self.graph = graph
        self.nodes = list(self.graph.nodes)

        self.start_node_idx = None
        self.task_nodes_idx = None
        self.end_node_idx = None

        self.cur_node_idx = None
        self.matrix = None
        self.visited = None

        state_dim = 30
        self.pi_voo = RNN(input_dim=state_dim, output_dim=1)
        state_dict = torch.load('../trained/model_tsp_static.pth')
        self.pi_voo.load_state_dict(state_dict)

    @property
    def start_node(self): return self.nodes[self.start_node_idx]

    @property
    def end_node(self): return self.nodes[self.end_node_idx]

    def get_all_nodes(self):
        return [self.start_node_idx, ] + self.task_nodes_idx + [self.end_node_idx, ]

    def __get_state(self):
        cur_node_arr = list(format(self.cur_node_idx, '010b'))
        end_node_arr = list(format(self.end_node_idx, '010b'))

        state = []
        for i, task_node_idx in enumerate(self.task_nodes_idx):
            if self.visited[task_node_idx] == 1:
                continue

            arr1 = list(format(task_node_idx, '010b'))
            arr2 = cur_node_arr[:]
            arr3 = end_node_arr[:]
            state_ = np.concatenate([np.array(arr1, dtype=np.float32),
                                     np.array(arr2, dtype=np.float32),
                                     np.array(arr3, dtype=np.float32)])
            state.append(state_)
        if len(state) <= 0: state = [np.zeros(30, )]
        return np.vstack(state)

    def reset(self, start, end, tasks):
        self.start_node_idx = start
        self.end_node_idx = end
        self.task_nodes_idx = tasks

        self.visited = {node: 0 for node in self.task_nodes_idx}
        self.cur_node_idx = self.start_node_idx
        print('\t Scenario Info: ', self.start_node_idx, self.end_node_idx, self.task_nodes_idx)

    def choose_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.pi_voo(state_tensor).squeeze(0).detach().numpy()
        indexes = np.argwhere(q_values == q_values.max()).squeeze(axis=1)
        return np.random.choice(indexes)

    def step(self, action):
        unvisited_task_nodes = [v for v in self.task_nodes_idx if self.visited[v] == 0]
        next_node_idx = unvisited_task_nodes[action]

        all_nodes = self.get_all_nodes()
        i1 = all_nodes.index(self.cur_node_idx)
        i2 = all_nodes.index(next_node_idx)
        reward = -self.matrix[i1][i2]

        self.cur_node_idx = next_node_idx
        self.visited[next_node_idx] = 1
        done = sum(self.visited.values()) == len(self.task_nodes_idx)
        if done:
            i3 = all_nodes.index(self.end_node_idx)
            reward -= self.matrix[i2][i3]
        return self.__get_state(), reward, done

    def run(self, matrix=None):
        self.matrix = matrix

        done, state = False, self.__get_state()
        cost, episode_step = 0.0, 0
        path = [self.start_node, ]
        while not done:
            action = self.choose_action(state)
            next_state, reward, done, *_ = self.step(action)
            path.append(self.nodes[self.cur_node_idx])
            state = next_state
            cost += reward
            episode_step += 1
            if episode_step >= max_step: break
        path.append(self.end_node)
        return path, -cost, done


class RLDynamicTSPSolver:
    def __init__(self, graph):
        self.graph = graph
        self.nodes = list(self.graph.nodes)

        self.start_node_idx = None
        self.task_nodes_idx = None
        self.end_node_idx = None

        self.cur_node_idx = None
        self.matrix = None
        self.visited = None

        state_dim = 32
        self.pi_voo = RNN(input_dim=state_dim, output_dim=1)
        state_dict = torch.load('../trained/model_tsp_rnn.pth')
        self.pi_voo.load_state_dict(state_dict)

    @property
    def start_node(self): return self.nodes[self.start_node_idx]

    @property
    def end_node(self): return self.nodes[self.end_node_idx]

    def get_all_nodes(self):
        return [self.start_node_idx, ] + self.task_nodes_idx + [self.end_node_idx, ]

    def __get_state(self):
        all_nodes = self.get_all_nodes()
        cur_node_arr = list(format(self.cur_node_idx, '010b'))
        end_node_arr = list(format(self.end_node_idx, '010b'))
        i1 = all_nodes.index(self.cur_node_idx)

        state = []
        for i, task_node_idx in enumerate(self.task_nodes_idx):
            if self.visited[task_node_idx] == 1:
                continue

            arr1 = list(format(task_node_idx, '010b'))
            arr2 = cur_node_arr[:] + [self.matrix[i1][i+1], ]
            arr3 = end_node_arr[:] + [self.matrix[i+1][-1], ]
            state_ = np.concatenate([np.array(arr1, dtype=np.float32),
                                     np.array(arr2, dtype=np.float32),
                                     np.array(arr3, dtype=np.float32)])
            state.append(state_)
        if len(state) <= 0: state = [np.zeros(32, )]
        return np.vstack(state)

    def reset(self, start, end, tasks):
        self.start_node_idx = start
        self.end_node_idx = end
        self.task_nodes_idx = tasks

        self.visited = {node: 0 for node in self.task_nodes_idx}
        self.cur_node_idx = self.start_node_idx
        print('\t Scenario Info: ', self.start_node_idx, self.end_node_idx, self.task_nodes_idx)

    def choose_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.pi_voo(state_tensor).squeeze(0).detach().numpy()
        indexes = np.argwhere(q_values == q_values.max()).squeeze(axis=1)
        return np.random.choice(indexes)

    def step(self, action):
        unvisited_task_nodes = [v for v in self.task_nodes_idx if self.visited[v] == 0]
        next_node_idx = unvisited_task_nodes[action]

        all_nodes = self.get_all_nodes()
        i1 = all_nodes.index(self.cur_node_idx)
        i2 = all_nodes.index(next_node_idx)
        reward = -self.matrix[i1][i2]

        self.cur_node_idx = next_node_idx
        self.visited[next_node_idx] = 1
        done = sum(self.visited.values()) == len(self.task_nodes_idx)
        if done:
            i3 = all_nodes.index(self.end_node_idx)
            reward -= self.matrix[i2][i3]
        return self.__get_state(), reward, done

    def run(self, matrix=None):
        self.matrix = matrix

        done, state = False, self.__get_state()
        cost, episode_step = 0.0, 0
        path = [self.start_node, ]
        while not done:
            action = self.choose_action(state)
            next_state, reward, done, *_ = self.step(action)
            path.append(self.nodes[self.cur_node_idx])
            state = next_state
            cost += reward
            episode_step += 1
            if episode_step >= max_step: break
        path.append(self.end_node)
        return path, -cost, done