import random
import math

import osmnx as ox

from common.geo import load_graph


class RRTStarPathPlanner:
    def __init__(self, graph, max_iter=1000, radius=3):
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.max_iter = max_iter
        self.radius = radius  # 重连接半径

    def distance(self, node1, node2):
        # 使用欧几里得距离或者边长
        p1 = self.graph.nodes[node1]
        p2 = self.graph.nodes[node2]
        return math.hypot(p1['x'] - p2['x'], p1['y'] - p2['y'])

    def get_edge_weight(self, u, v, weight_key):
        if not self.graph.has_edge(u, v):
            return float('inf')
        edge_data = self.graph.get_edge_data(u, v)
        if isinstance(edge_data, dict):
            edge_data = edge_data[0] if 0 in edge_data else list(edge_data.values())[0]
        return edge_data.get(weight_key, float('inf'))

    def run(self, start_node, end_node, weight_key='dynamic_weight'):
        tree = {start_node: None}
        cost = {start_node: 0.0}

        for _ in range(self.max_iter):
            # 带目标偏置的随机节点采样
            if random.random() < 0.2:
                rand_node = end_node
            else:
                rand_node = random.choice(self.nodes)

            # 找已有节点中与rand_node相连的最近的 parent
            nearby = [n for n in tree if self.graph.has_edge(n, rand_node)]
            if not nearby:
                continue

            # 选择 cost 最小的 parent 作为连接点
            min_parent = min(
                nearby,
                key=lambda n: cost[n] + self.get_edge_weight(n, rand_node, weight_key)
            )
            min_cost = cost[min_parent] + self.get_edge_weight(min_parent, rand_node, weight_key)

            # 添加到树中
            if rand_node not in tree:
                tree[rand_node] = min_parent
                cost[rand_node] = min_cost

            # ====== RRT*：重连优化周围节点 ======
            for neighbor in nearby:
                if neighbor == rand_node:
                    continue
                new_cost = cost[rand_node] + self.get_edge_weight(rand_node, neighbor, weight_key)
                if new_cost < cost.get(neighbor, float('inf')):
                    tree[neighbor] = rand_node
                    cost[neighbor] = new_cost

            if rand_node == end_node:
                break

        # 回溯路径
        if end_node not in tree:
            return [], float('inf'), False

        path = []
        node = end_node
        total_cost = 0.0
        while node is not None:
            path.append(node)
            parent = tree[node]
            if parent is not None:
                total_cost += self.get_edge_weight(parent, node, weight_key)
            node = parent

        return path[::-1], total_cost, True


def main():
    # 初始化图和动态权重
    place = "南京航空航天大学(将军路校区)"
    graph, p, num_action = load_graph(place_name=place)

    # 创建 RRT 路径规划器
    planner = RRTStarPathPlanner(graph, max_iter=2500, radius=3)

    # 随机起点终点
    nodes = list(graph.nodes)
    start = random.choice(nodes)
    end = random.choice(nodes)
    print(start, end)

    # 运行
    path, cost, success = planner.run(start, end)
    if success:
        print(f"Found path with cost: {cost}")
        ox.plot_graph_route(graph, path, route_linewidth=4, node_size=0)
    else:
        print("No path found.")


if __name__ == '__main__':
    main()

