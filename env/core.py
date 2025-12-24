
class Task:
    def __init__(self, name, task_node_idx):
        self.name = name
        self.task_node_idx = task_node_idx
        self.over_time = None

    def is_over(self):
        return self.over_time is not None

    def complete_task(self, clock):
        self.over_time = clock

    def clear(self):
        self.task_node_idx = None
        self.over_time = None


class Vehicle:
    def __init__(self, start_node_idx=None):
        self.cur_node_idx = start_node_idx
        self.start_node_idx = start_node_idx
        self.end_node_idx = None

        self.trajectory = []
        self.previous_paths = []  # Store historical paths (up to 3 paths)

        self.distance_cost = 0.0
        self.time_cost = 0.0

    def set_start_node(self, node_idx):
        self.start_node_idx = node_idx

    def set_end_node(self, node_idx):
        self.end_node_idx = node_idx

    def clear(self):
        self.cur_node_idx = self.start_node_idx
        self.end_node_idx = None
