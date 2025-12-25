import numpy as np

from env.core import Agent, Task
from env.rendering import StaticRender


class DynamicUrbanEnv:
    def __init__(self,
                 graph,
                 num_agents=3,
                 num_tasks=5,
                 start_node=None,
                 end_node=None,
                 dynamics=None):
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.num_agents = num_agents
        self.num_tasks = num_tasks
        print('-------Environment-----------')
        print('\t Number of agents:', num_agents)
        print('\t Number of tasks:', num_tasks)

        self.dynamics = dynamics
        self.start_node = start_node
        self.end_node = end_node
        self.empty_nodes = None
        self.agents = [Agent(i, self.start_node, self.end_node)
                       for i in range(num_agents)]

        self.clock = 0
        self.all_tasks = None
        self.pending_tasks = None
        self.cv_render = None

    def relevant_nodes(self):
        nodes = []
        for task in self.pending_tasks:
            nodes.append(task.node)
        for agent in self.agents:
            if agent.is_over(): continue
            nodes.append(agent.current_node)
            nodes.append(agent.end_node)
        return list(set(nodes))

    def relevant_agent(self):
        return [agent for agent in self.agents if not agent.is_over()]

    def reset(self, reuse=False, render=False):
        if self.all_tasks is None: reuse = False
        if not reuse:
            self.empty_nodes = self.nodes.copy()
            self.empty_nodes.remove(self.start_node)
            self.empty_nodes.remove(self.end_node)
            np.random.shuffle(self.empty_nodes)
            self.all_tasks = [Task(i, self.empty_nodes.pop(0)) for i in range(self.num_tasks)]

        self.dynamics.reset()
        self.pending_tasks = self.all_tasks[:]
        for agent in self.agents: agent.reset()

        if render and self.cv_render is None:
            self.cv_render = StaticRender(self)

    def step(self, interval=10, weight_key='dynamic_weight'):
        self.clock += 1
        for agent in self.agents:
            if agent.is_over(): continue
            agent.step(clock=self.clock)
            u = agent.last_node
            v = agent.current_node
            agent.cost += self.graph[u][v].get(weight_key, 1.0)

        has_dynamics = False
        if self.clock <= 400 and self.clock % interval == 0:
            i = self.all_tasks[-1].name
            new_task = Task(i+1, self.empty_nodes.pop(0))
            self.all_tasks.append(new_task)
            has_dynamics = True

        # Update tasks
        self.pending_tasks = [t for t in self.all_tasks if not t.is_completed()]

        # Update environment dynamics
        if self.clock % interval == 0:
            self.dynamics.next()
            has_dynamics = True
        return has_dynamics

    def render(self, **kwargs):
        if self.cv_render is None: return
        self.cv_render.draw(**kwargs)

    def close(self):
        if self.cv_render is None: return
        self.cv_render.close()
