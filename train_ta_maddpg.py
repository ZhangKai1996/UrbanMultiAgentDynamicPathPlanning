import os
import csv
import time
import shutil

import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from algo.trainer import Trainer
from algo.baselines import dijkstra_path
from common.geo import load_graph
from env.environment import CityDeliveryEnv


def train(env,
          graph, p, num_action,
          max_step=100,
          batch_size=32,
          save_rate=100,
          test_rate=None,
          update_rate=10,
          epsilon_end=0.1,
          epsilon_decay=0.1,
          num_episodes=int(1e3),
          num_episodes_test=int(1e3),
          folder='exp_ta/'):
    np.random.seed(1234)

    trainer = Trainer(graph, p, num_action,
                      batch_size=batch_size,
                      epsilon_decay=int(num_episodes * epsilon_decay),
                      epsilon_end=epsilon_end)
    if test_rate is None: test_rate = len(graph.nodes)

    log_folder = folder + 'logs'
    if os.path.exists(log_folder):
        shutil.rmtree(log_folder)
        print('Removing all previous files!')
    writer = SummaryWriter(log_dir=log_folder)

    print('-------------Train kwargs------------')
    print('\t Number of episodes:', num_episodes)
    print('\t Epsilon decay:', epsilon_decay)
    print('\t Epsilon end:', epsilon_end)
    print('\t Batch size:', batch_size)
    print('\t Update rate:', update_rate)
    print('\t Save rate:', save_rate)
    print('\t Test rate:', test_rate)
    print('\t Folder:', folder)
    print('-------------------------------------')

    rew_stats, sr_stats = [], []
    loss_stats, step_stats = [], []
    start_time = time.time()

    step = 0
    for episode in tqdm(range(1, num_episodes + 1), desc='Training ...'):
        done, states = False, env.reset()

        total_reward, episode_step = 0.0, 0
        while not done:
            actions = trainer.choose_action(states, episode=episode)
            next_states, rewards, dones, *_ = env.step(actions)
            experience = (states, actions, rewards, next_states, float(dones))
            trainer.add_experience(*experience)

            loss = trainer.update(step=step, update_rate=update_rate)
            if loss is not None: loss_stats.append(loss)

            states = next_states
            total_reward += rewards
            episode_step += 1
            step += 1
            if episode_step >= max_step: break

        step_stats.append(episode_step)
        rew_stats.append(total_reward)
        sr_stats.append(int(done))
        if episode % save_rate == 0:
            end_time = time.time()
            mean_rew = np.mean(rew_stats)
            mean_step = np.mean(step_stats)
            sr = np.mean(sr_stats)
            eps = trainer.epsilon(episode)

            print("Episode: {}".format(episode), end=', ')
            print("Step: {}".format(step), end=', ')
            print("Mean Step: {:>6.2f}".format(mean_step), end=', ')
            print("Mean Rew: {:>7.2f}".format(mean_rew), end=', ')
            print("SR: {:>6.4f}".format(sr), end=', ')
            print("Epsilon: {:>6.4f}".format(eps), end=', ')
            print("Time: {:>5.2f}".format(end_time - start_time))

            if len(loss_stats) > 0:
                mean_loss = np.mean(loss_stats)
                writer.add_scalar('Loss', mean_loss, episode)
            writer.add_scalar('Epsilon', eps, step)
            writer.add_scalar('Step', mean_step, episode)
            writer.add_scalar('Reward', mean_rew, episode)
            writer.add_scalar('Success Rate', sr, episode)
            writer.add_scalars('critic_loss',
                               {'agent_{}'.format(i + 1): v for i, v in enumerate(np.mean(self.c_losses, axis=0))},
                               episode)
            writer.add_scalars('actor_loss',
                               {'agent_{}'.format(i + 1): v for i, v in enumerate(np.mean(self.a_losses, axis=0))},
                               episode)

            rew_stats, sr_stats = [], []
            loss_stats, step_stats = [], []
            start_time = end_time

            trainer.save_model(folder)

        if episode % test_rate == 0:
            epoch = episode // save_rate
            test(env, trainer,
                 epoch=epoch,
                 max_step=max_step,
                 render=False,
                 num_episodes=num_episodes_test,
                 test_path=folder + 'result.csv')


def test(env, trainer,
         num_episodes=int(1e3),
         epoch=0,
         max_step=100,
         render=False,
         test_path=None):

    sr_stats, gap_stats = [], []
    for _ in range(num_episodes):
        done_rl, (state, mask) = False, env.reset(render=render, is_test=False)

        start_node, end_node = env.start_node, env.end_node
        cost_di, path_di = dijkstra_path(env.graph, start_node, end_node, weight_key='length')
        cost_di *= env.alpha
        done_di = path_di[0] == start_node and path_di[-1] == end_node

        cost_rl, episode_step = 0.0, 0
        path_rl = [env.cur_node, ]
        while not done_rl:
            action = trainer.choose_action(state, mask)
            (next_state, next_mask), reward, done_rl, *_ = env.step(action, is_test=True)
            path_rl.append(env.cur_node)
            cost_rl += -reward
            state, mask = next_state, next_mask
            episode_step += 1

            if episode_step >= max_step: break

        if render:
            # paths = {'RL': [end_node, cost_rl, path_rl, done_rl],
            #          'DI': [end_node, cost_di, path_di, done_di]}
            paths = {'RL': [end_node, cost_rl, path_rl, done_rl]}
            env.render(paths=paths, name=f'{epoch}', show=False)

        if done_rl and done_di:
            gap = (cost_rl - cost_di) / cost_di
            gap_stats.append([gap, int(gap == 0)])
        sr_stats.append([int(done_rl), int(done_di)])

    result = [epoch, ] + [0.0 for _ in range(5)]
    if len(gap_stats) > 0:
        gap_stats_arr = np.array(gap_stats)
        result = [epoch, gap_stats_arr.max(0)[0], ]
        result += list(gap_stats_arr.mean(0))
        result += list(np.array(sr_stats).mean(0))

    if test_path is not None:
        title = None
        if not os.path.exists(test_path):
            title = ['Epoch', 'Max Gap', 'Mean Gap', 'ZR', 'SR(RL)', 'SR(DI)']

        with open(test_path, 'a+', newline='') as f:
            f_writer = csv.writer(f)
            if title is not None: f_writer.writerow(title)
            f_writer.writerow(result)


def main():
    from parameters import num_tasks, num_agents
    from parameters import c, alpha, radius, env_id
    from parameters import max_step, env_kwargs, kwargs

    args = load_graph(radius=radius, **env_kwargs)
    graph, p, num_action, center_node, ranked_nodes, *_ = args

    env = CityDeliveryEnv(graph, p, num_action,
                          c=c,
                          alpha=alpha,
                          num_tasks=num_tasks,
                          num_agents=num_agents)

    root = 'trained/exp_ta_{}_{}/'.format(env_id, int(radius))
    folder = root + '{}_{}/'.format(c, alpha)

    train(env,
          graph, p,
          num_action,
          folder=folder,
          max_step=max_step,
          num_episodes_test=1,
          **kwargs)
    print('Training process is over !')
    print('-------------------------------------')


if __name__ == '__main__':
    main()
