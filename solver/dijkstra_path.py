import heapq


class DijkstraSolver:
    def __init__(self, graph, weight_key="length"):
        self.graph = graph
        self.weight_key = weight_key

    def solve(self, start, goal):
        pq = [(0.0, start, [])]
        visited = set()

        while pq:
            cost, u, path = heapq.heappop(pq)
            if u in visited: continue
            visited.add(u)
            path = path + [u]
            if u == goal: return path, cost

            for v, data in self.graph[u].items():
                w = data.get(self.weight_key, 1.0)
                heapq.heappush(pq, (cost + w, v, path))

        return None, float("inf")

    def precompute_distances(self, nodes):
        """
        nodes: iterable of node ids
        return: dist[u][v]
        """
        dist = {u: {} for u in nodes}
        for u in nodes:
            for v in nodes:
                if u == v: continue
                dist[u][v] = self.solve(u, v)[1]
        return dist
