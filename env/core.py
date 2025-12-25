
class Task:
    def __init__(self, name, node):
        self.name = name
        self.node = node
        self.time = None

    def is_completed(self):
        return self.time is not None

    def set_time(self, clock):
        self.time = clock

    def clear(self):
        self.node = None
        self.time = None


class Agent:
    def __init__(self, agent_id, start_node, end_node):
        self.id = agent_id

        self.start_node = start_node
        self.end_node = end_node
        self.current_node = start_node
        self.last_node = None

        self.assigned_tasks = []
        self.completed_tasks = []
        self.planned_path = []
        self.cost = 0.0

    def is_over(self):
        return self.current_node == self.end_node

    def reset(self):
        self.current_node = self.start_node
        self.last_node = None
        self.assigned_tasks = []
        self.completed_tasks = []
        self.planned_path = []

    def current_task(self):
        if len(self.assigned_tasks) <= 0: return None
        return self.assigned_tasks[0]

    def step(self, clock):
        if len(self.planned_path) <= 0: return

        self.last_node = self.current_node
        self.current_node = self.planned_path.pop(0)

        cur_task = self.current_task()
        if cur_task is not None:
            if self.current_node == cur_task.node:
                cur_task.set_time(clock=clock)
                task = self.assigned_tasks.pop(0)
                self.completed_tasks.append(task)

    def assign_tasks(self, task_seq):
        self.assigned_tasks = task_seq[:]

    def assign_path(self, path):
        self.planned_path = path[:]
