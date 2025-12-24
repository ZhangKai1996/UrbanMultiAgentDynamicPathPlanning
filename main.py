import time
from common.geo import load_graph

from env.environment import MultiAgentTaskEnv
from solver.assignment_ga import MultiAgentAssignmentGA
from solver.dijkstra_path import DijkstraSolver


def main():
    from parameters import radius, env_kwargs

    num_agents = 5
    num_tasks = 30
    args =  load_graph(radius=radius, **env_kwargs)
    graph, p, num_action, center_node, ranked_nodes, *_ = args

    init_nodes = [center_node, list(graph.nodes)[ranked_nodes[-1]]]
    env = MultiAgentTaskEnv(graph, num_agents, num_tasks, init_nodes)
    env.reset(render=True)

    pp_solver = DijkstraSolver(graph, weight_key="length")
    nodes_for_dist = ([t["node"] for t in env.all_tasks] + init_nodes)
    dist = pp_solver.precompute_distances(nodes_for_dist)
    ta_solver = MultiAgentAssignmentGA(dist=dist,
                                       agents=env.agents,
                                       init_nodes=init_nodes)

    t0 = time.time()
    assignment = ta_solver.solve(env.pending_tasks)
    print("Assignment time:", time.time() - t0)

    # ===== Path construction =====
    paths = {}
    for aid, task_ids in assignment.items():
        agent = env.agents[aid]
        agent.assign_tasks(task_ids)

        current = agent.start_node
        full_path = [current, ]
        for tid in task_ids:
            node = env.all_tasks[tid]["node"]
            path, _ = pp_solver.solve(current, node)
            full_path += path[1:]
            current = node

        path, _ = pp_solver.solve(current, agent.end_node)
        full_path += path[1:]
        agent.assign_path(full_path)

        cost = sum(graph[u][v].get("length", 1.0)
                   for u, v in zip(full_path[:-1], full_path[1:]))
        paths[aid] = [task_ids, cost, full_path]

    env.render(paths=paths,
               init_nodes=init_nodes,
               tasks=env.all_tasks,
               name="multi_agent_paths",
               show=True)


if __name__ == "__main__":
    main()
