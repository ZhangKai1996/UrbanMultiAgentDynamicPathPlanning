from common.geo import load_graph

num_episodes = 1 * 2
# num_episodes = int(1e3)
place_name = "南京航空航天大学(将军路校区)"
args = load_graph(place_name)
kwargs = {
    'has_writer': False,
}


def main():
    from train_dpp_static import DPPStatic
    planner = DPPStatic(*args, **kwargs)

    # from train_dpp_dynamic import DPPDynamic
    # planner = DPPDynamic(*args, **kwargs)

    # from train_dpp_pa import DPPDynamicPA
    # planner = DPPDynamicPA(*args, **kwargs)

    # from train_dpp_encoder import DPPWithEncoder
    # planner = DPPWithEncoder(*args, **kwargs)

    # from train_pp_dqn import PathPlannerDQN
    # planner = PathPlannerDQN(*args, **kwargs)

    # from train_tsp_static import StaticTSPSolverRNN
    # planner = StaticTSPSolverRNN(*args, **kwargs)

    planner.test(num_episodes=num_episodes)


if __name__ == '__main__':
    main()
