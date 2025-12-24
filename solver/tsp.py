import random
import heapq

import numpy as np


class TSPSolverGA:
    def __init__(self, dist):
        self.dist = dist

    def solve(self, start_node, task_nodes, end_node,
              population_size=20,
              generations=30,
              mutation_rate=0.2):

        if len(task_nodes) <= 0:
            return [], self.dist[start_node][end_node]
        if len(task_nodes) == 1:
            cost = self.dist[start_node][task_nodes[0]]
            cost += self.dist[task_nodes[0]][end_node]
            return task_nodes, cost

        def path_cost(order):
            cost_ = self.dist[start_node][order[0]]
            for i in range(len(order) - 1):
                cost_ += self.dist[order[i]][order[i + 1]]
            cost_ += self.dist[order[-1]][end_node]
            return cost_

        def random_individual():
            perm = task_nodes[:]
            random.shuffle(perm)
            return perm

        def crossover(p1_, p2_):
            n = len(p1_)
            a, b = sorted(random.sample(range(n), 2))
            child_ = [None] * n
            child_[a:b] = p1_[a:b]

            ptr = b
            for x in p2_:
                if x not in child_:
                    if ptr >= n: ptr = 0
                    child_[ptr] = x
                    ptr += 1
            return child_

        def mutate(ind):
            if random.random() < mutation_rate:
                i, j = random.sample(range(len(ind)), 2)
                ind[i], ind[j] = ind[j], ind[i]

        population = [random_individual() for _ in range(population_size)]

        for _ in range(generations):
            population.sort(key=path_cost)
            elites = population[: population_size // 3]

            new_pop = elites[:]
            while len(new_pop) < population_size:
                p1, p2 = random.sample(elites, 2)
                child = crossover(p1, p2)
                mutate(child)
                new_pop.append(child)
            population = new_pop

        best = min(population, key=path_cost)
        return best, path_cost(best)


class TSPSolverDP:
    def __init__(self, dist):
        self.dist = dist

    def solve(self, start_node, task_nodes, end_node):
        if len(task_nodes) <= 0:
            return [], self.dist[start_node][end_node]
        if len(task_nodes) == 1:
            cost = self.dist[start_node][task_nodes[0]]
            cost += self.dist[task_nodes[0]][end_node]
            return task_nodes, cost

        # 构建 DP 节点列表
        nodes = [start_node] + task_nodes + [end_node]
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        n = len(nodes)

        # 转换成矩阵索引的距离
        dist_matrix = np.zeros((n, n))
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if i == j:
                    dist_matrix[i][j] = 0
                else:
                    dist_matrix[i][j] = self.dist[u][v]

        # DP 初始化
        dp = np.inf * np.ones((1 << n, n))
        dp[1 << 0][0] = 0  # 从 start_node 开始

        # 遍历所有子集
        for mask in range(1 << n):
            for u in range(n):
                if (mask & (1 << u)) == 0:
                    continue
                for v in range(n):
                    if mask & (1 << v):
                        continue
                    dp[mask | (1 << v)][v] = min(dp[mask | (1 << v)][v],
                                                 dp[mask][u] + dist_matrix[u][v])

        # 最终成本
        full_mask = (1 << n) - 1
        min_cost = dp[full_mask][n - 1]  # 到达终点

        # 回溯最优路径（节点索引）
        path_idx = [n - 1]
        mask = full_mask
        last = n - 1
        while last != 0:
            for u in range(n):
                if mask & (1 << u) and u != last:
                    if dp[mask][last] == dp[mask ^ (1 << last)][u] + dist_matrix[u][last]:
                        path_idx.append(u)
                        mask ^= (1 << last)
                        last = u
                        break
        path_idx = path_idx[::-1]

        # 只返回 task_nodes 顺序
        best_order = [nodes[i] for i in path_idx if nodes[i] in task_nodes]
        return best_order, min_cost



class TSPSolverBnB:
    def __init__(self, dist):
        self.dist = dist

    @staticmethod
    def calculate_bound(dist_matrix, visited):
        """
        计算当前状态的界限（估算最短剩余路径）
        """
        n = len(dist_matrix)
        bound = 0.0
        # 行最小值
        for i in range(n):
            if visited[i]: continue

            row_min = float('inf')
            for j in range(n):
                if not visited[j]: row_min = min(row_min, dist_matrix[i][j])
            if row_min != float('inf'): bound += row_min

        # 列最小值
        for j in range(n):
            if visited[j]: continue

            col_min = float('inf')
            for i in range(n):
                if not visited[i]: col_min = min(col_min, dist_matrix[i][j])
            if col_min != float('inf'): bound += col_min
        return bound

    def solve(self, start_node, task_nodes, end_node):
        if len(task_nodes) <= 0:
            return [], self.dist[start_node][end_node]
        if len(task_nodes) == 1:
            cost = self.dist[start_node][task_nodes[0]]
            cost += self.dist[task_nodes[0]][end_node]
            return task_nodes, cost

        # 构建节点列表: start + task_nodes + end
        nodes = [start_node] + task_nodes + [end_node]
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        n = len(nodes)

        # 距离矩阵索引化
        dist_matrix = np.zeros((n, n))
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                dist_matrix[i][j] = 0.0 if i==j else self.dist[u][v]

        # 最优解初始化
        best_cost = float('inf')
        best_path = []

        # 优先队列 (cost_so_far, path, visited, bound)
        pq = []
        visited = [False] * n
        visited[0] = True  # start_node 已访问
        initial_bound = self.calculate_bound(dist_matrix, visited)
        heapq.heappush(pq, (0.0, [0], visited, initial_bound))

        while pq:
            current_cost, current_path, current_visited, current_bound = heapq.heappop(pq)
            if current_cost + current_bound >= best_cost: continue  # 剪枝

            if len(current_path) == n:
                if current_path[-1] == n - 1:
                    best_cost = current_cost
                    best_path = current_path
                continue

            # 扩展节点
            for i in range(n):
                if not current_visited[i]:
                    new_cost = current_cost + dist_matrix[current_path[-1]][i]
                    new_path = current_path + [i]
                    new_visited = current_visited[:]
                    new_visited[i] = True
                    new_bound = self.calculate_bound(dist_matrix, new_visited)
                    if new_cost + new_bound < best_cost:
                        heapq.heappush(pq, (new_cost, new_path, new_visited, new_bound))

        # 转换回节点 ID，并只返回任务节点顺序
        best_order = [nodes[i] for i in best_path if nodes[i] in task_nodes]
        return best_order, best_cost
