
class Agent:
    def __init__(self, agent_id, start_node, end_node):
        self.id = agent_id
        self.start_node = start_node
        self.end_node = end_node

        self.current_node = start_node
        self.task_seq = []
        self.path = []

    def is_over(self):
        return self.current_node == self.end_node

    def reset(self):
        self.current_node = self.start_node
        self.task_seq = []
        self.path = []

    def step(self):
        if len(self.path) <= 0: return
        self.current_node = self.path.pop(0)

    def assign_tasks(self, task_ids):
        self.task_seq = task_ids[:]

    def assign_path(self, path):
        self.path = path[:]
