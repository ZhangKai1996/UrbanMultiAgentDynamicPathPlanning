import numpy as np

from env.agent import Agent
from env.rendering import StaticRender


class MultiAgentTaskEnv:
    def __init__(self, graph, num_agents, num_tasks, init_nodes):
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.num_agents = num_agents
        self.num_tasks = num_tasks

        self.start_node, self.end_node = init_nodes
        self.agents = [Agent(i, self.start_node, self.end_node)
                       for i in range(num_agents)]

        self.all_tasks = None
        self.pending_tasks = None
        self.cv_render = None

    def reset(self, reuse=False, render=False):
        if self.all_tasks is None: reuse = False
        if not reuse:
            nodes = self.nodes.copy()
            nodes.remove(self.start_node)
            nodes.remove(self.end_node)
            np.random.shuffle(nodes)
            self.all_tasks = [{"id": i, "node": nodes[i]} for i in range(self.num_tasks)]

        self.pending_tasks = self.all_tasks.copy()
        [agent.reset() for agent in self.agents]

        if render and self.cv_render is None:
            self.cv_render = StaticRender(self.graph)

    def step(self):
        for agent in self.agents:
            if agent.is_over(): continue

            agent.step()

    def render(self, **kwargs):
        if self.cv_render is None: return
        self.cv_render.draw(**kwargs)

    def close(self):
        if self.cv_render is None: return
        self.cv_render.close()
