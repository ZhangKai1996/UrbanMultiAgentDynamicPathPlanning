import time
from common.geo import load_graph
from env.dynamics import DynamicsIterator

from env.environment import DynamicUrbanEnv
from solver.assignment_ga import MultiAgentAssignmentGA
from solver.dijkstra_path import DijkstraSolver


def main():
    from parameters import radius, env_kwargs, num_dynamics, env_id

    num_agents = 5
    num_tasks = 10
    weight_key = 'dynamic_weight'
    render = True

    args = load_graph(radius=radius, **env_kwargs)
    graph, p, num_action, center_node, ranked_nodes, *_ = args

    dynamics_iterator = DynamicsIterator(graph, radius, num_dynamics, env_id)
    env = DynamicUrbanEnv(graph,
                          num_agents=num_agents,
                          num_tasks=num_tasks,
                          start_node=center_node,
                          end_node=list(graph.nodes)[ranked_nodes[-1]],
                          dynamics=dynamics_iterator)
    env.reset(render=render)

    # Dijkstra for Path Planning
    pp_solver = DijkstraSolver(graph, weight_key=weight_key)
    has_dynamics = True

    step = 0
    while True:
        print(f'Step {step} ({has_dynamics}):')
        tasks = env.pending_tasks
        agents = env.relevant_agent()
        if len(agents) <= 0: break

        print('\t>>> Agent state ({}): '.format(len(agents)))
        for agent in env.agents:
            print('\t', agent.id, agent.current_node, agent.start_node, agent.end_node)
        print('\t>>> Task state ({}): '.format(len(tasks)))
        for task in env.all_tasks:
            print('\t', task.name, task.node, task.time, task.is_completed())

        if has_dynamics:
            dist = pp_solver.precompute_distances(nodes=env.relevant_nodes())
            ta_solver = MultiAgentAssignmentGA(dist=dist, agents=agents)

            t0 = time.time()
            assignment = ta_solver.solve(tasks=tasks)
            print("\t>>> Assignment time:", time.time() - t0)

            # ===== Path construction =====
            for aid, task_ids in assignment.items():
                assigned_tasks = [tasks[tid] for tid in task_ids]
                print('\t>>>', aid, task_ids)
                agent = agents[aid]
                agent.assign_tasks(assigned_tasks)

                current = agent.current_node
                full_path = [current, ]
                for task in assigned_tasks:
                    node = task.node
                    path, _ = pp_solver.solve(current, node)
                    full_path += path[1:]
                    current = node

                path, _ = pp_solver.solve(current, agent.end_node)
                full_path += path[1:]

                eta = [graph[u][v].get(weight_key, 1.0) for u, v in zip(full_path[:-1], full_path[1:])]
                agent.assign_path(full_path[1:])

        has_dynamics = env.step(weight_key=weight_key)
        if render: env.render(show=False, has_dynamics=has_dynamics)
        step += 1


if __name__ == "__main__":
    main()
