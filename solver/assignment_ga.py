import random

from solver.tsp import TSPSolverGA


class MultiAgentAssignmentGA:
    def __init__(self, dist, agents, init_nodes):
        self.dist = dist
        self.agents = agents
        self.start_node = init_nodes[0]
        self.end_node = init_nodes[1]

        self.tsp_solver = TSPSolverGA(dist)

    def solve(self, tasks, population_size=30, generations=40):
        task_ids = [t["id"] for t in tasks]
        task_nodes = {t["id"]: t["node"] for t in tasks}
        num_agents = len(self.agents)

        def random_individual():
            return {t_id: random.randint(0, num_agents - 1) for t_id in task_ids}

        def fitness(ind):
            orders = [[] for _ in range(num_agents)]
            for t_id, a_id in ind.items(): orders[a_id].append(task_nodes[t_id])

            cost = 0.0
            for order in orders:
                _, c = self.tsp_solver.solve(self.start_node, order, self.end_node)
                cost += c
            return cost

        population = [random_individual() for _ in range(population_size)]

        for _ in range(generations):
            population.sort(key=fitness)
            population = population[: population_size // 2]

            while len(population) < population_size:
                p1, p2 = random.sample(population, 2)
                child = {}
                for tid in task_ids:
                    child[tid] = p1[tid] if random.random() < 0.5 else p2[tid]
                    if random.random() < 0.1:
                        child[tid] = random.randint(0, num_agents - 1)
                population.append(child)

        best = min(population, key=fitness)

        assignment = {i: [] for i in range(num_agents)}
        for tid, aid in best.items():
            assignment[aid].append(tid)
        return assignment
