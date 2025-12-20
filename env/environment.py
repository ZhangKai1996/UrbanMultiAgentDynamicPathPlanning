import osmnx as ox
import numpy as np

from env.core import Vehicle, Task
from env.rendering import EnvRender


class CityDeliveryEnv:
    def __init__(self, args):
        self.num_tasks = args.num_tasks

        self.__get_city_map(args.place)
        self.vehicle = None
        self.cv_render = None

        self.clock = 0
        self.time = 0.0

    def dot(self):
        return sum([int(tasks.is_delivered()) for tasks in self.tasks])

    def __get_city_map(self, place):
        print('[Environment] Parsing the map ...')
        graph = ox.graph_from_place(place, network_type='drive')
        print(f"Initial number of nodes: {len(graph.nodes)}")
        while True:
            # 删除孤立节点：没有连接任何其他节点的节点
            isolated_nodes = []
            ret, max_number = {}, 0
            for node1 in graph.nodes:
                tmp = []
                for node2, data in graph[node1].items():
                    if node1 == node2: continue
                    if data.get('length', 1) > 0: tmp.append(node2)
                if len(tmp) <= 0:
                    isolated_nodes.append(node1)
                else:
                    ret[node1] = tmp
                    max_number = max(max_number, len(tmp))

            if len(isolated_nodes) <= 0: break
            print(isolated_nodes)
            # 从图中移除孤立节点
            graph.remove_nodes_from(isolated_nodes)
        print(f"Number of nodes after removing isolated nodes: {len(graph.nodes)}")

        self.city_map = graph
        self.p = ret
        self.num_action = max_number
        print('\t Number of city nodes: ', len(graph.nodes))
        print('\t Number of city edges: ', len(graph.edges))

    def __generate_tasks(self, num_tasks):
        print('\t Generating the tasks ...')
        all_nodes = list(self.city_map.nodes)[:]
        np.random.shuffle(all_nodes)
        pickup_nodes = all_nodes[:num_tasks]
        delivery_nodes = all_nodes[num_tasks:num_tasks*2]

        others = all_nodes[num_tasks*2:]
        [self.start_node, self.end_node] = others[:2]
        self.other_nodes = others[2:]

        print('\t Start node: ', self.start_node)
        print('\t End node: ', self.end_node)
        print('\t Number of task: ', num_tasks)
        self.tasks = [Task(i + 1, pickup_nodes[i], delivery_nodes[i]) for i in range(num_tasks)]
        for i in range(num_tasks):
            print('\t\t Task {}'.format(i+1),  pickup_nodes[i], delivery_nodes[i])

    def __add_env_dynamics(self, dynamic=False):
        for u, v, data in self.city_map.edges(data=True):
            length = data.get('length', 1)
            times = np.random.uniform(0.2, 1.0) if dynamic else 0.6
            traffic_time = length / (self.vehicle.speed * times + 1e-6)
            data['times'] = times
            data['dynamic_weight'] = traffic_time

    def __add_task_dynamics(self):
        num_new_tasks = np.random.randint(4)
        for i in range(num_new_tasks):
            p_node = self.other_nodes.pop(0)
            d_node = self.other_nodes.pop(0)
            new_task = Task(len(self.tasks)+1, p_node, d_node)
            self.tasks.append(new_task)

    def __update_task(self):
        for task in self.tasks:
            if task.is_recovered: continue

            if not task.is_picked():
                if self.vehicle.cur_node == task.pickup_node:
                    task.complete_pickup(clock=self.time)
            elif not task.is_delivered():
                if self.vehicle.cur_node == task.delivery_node:
                    task.complete_delivery(clock=self.time)
            else:
                # Task nodes are recovered after the task is finished.
                self.other_nodes.append(task.pickup_node)
                self.other_nodes.append(task.delivery_node)
                task.is_recovered = True

    def __observation_tsp(self):

        for task in self.tasks:
            if task.is_delivered(): continue

    def __observation_dpp(self):
        return

    def observation(self, label='tsp'):
        if label == 'tsp': return self.__observation_tsp()
        if label == 'dpp': return self.__observation_dpp()
        raise NotImplementedError

    def reset(self, label='tsp', render=False):
        print('[Environment] Initialize environment')
        if label == 'tsp':
            self.clock = 0
            self.time = 0.0
            self.__generate_tasks(self.num_tasks)
            self.vehicle = Vehicle(start_node=self.start_node)
            self.__add_env_dynamics(dynamic=False)

            if render and self.cv_render is None:
                self.cv_render = EnvRender(self)
        elif label == 'dpp':
            pass
        else:
            raise NotImplementedError

    def step(self, action, interval=120, label='tsp'):
        if label == 'tsp':
            self.vehicle.set_end_node()
        # Increment the clock each step
        self.clock += 1
        self.__update_task()

        # Update weights every step
        # if self.clock % interval == 0:
        #     print('[Environment] Edge weights change at {}'.format(self.clock))
        #     self.__add_env_dynamics(dynamic=True)
        #     self.__add_task_dynamics()

    def render(self, **kwargs):
        if self.cv_render is None: return
        self.cv_render.draw(**kwargs)

    def close(self):
        if self.cv_render is None: return
        self.cv_render.close()
