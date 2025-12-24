max_step = 100

c = 100.0
alpha = 0.001
num_agents = 5
num_tasks = 10

env_kwargs = {'place_name': None,
              'network_type': 'drive',
              'center_idx': 12000,
              'remove': True,
              'render': False}

kwargs = {'num_episodes': int(5e4),
          'epsilon_decay': 0.5,
          'epsilon_end': 0.1,
          'batch_size': 256,
          'update_rate': 100,
          'save_rate': 100,
          'test_rate': 100}

env_id = 1
if env_id == 1:
    radius: float = 8e3
    env_kwargs['place_name'] = "Nanjing, China"
    env_kwargs['center_idx'] = 12000
    # kwargs['num_episodes'] = int(1165 * 1e2)
if env_id == 2:
    radius: float = 1e6
    env_kwargs['place_name'] = "Chaoyang, Beijing, China"
    env_kwargs['center_idx'] = 10200
    # kwargs['num_episodes'] = int(356 * 1e2)
if env_id == 3:
    radius: float = 1e6
    env_kwargs['place_name'] = "Pudong, Shanghai, China"
    env_kwargs['center_idx'] = 12500
    # kwargs['num_episodes'] = int(370 * 1e2)
