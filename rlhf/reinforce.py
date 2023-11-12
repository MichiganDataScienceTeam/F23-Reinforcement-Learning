import argparse
from functools import reduce
import gymnasium as gym
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np
from itertools import count
from collections import deque
from discriminator import compare
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from typing import List, Tuple

from reward_estimator import Rewards


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=129, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('LunarLander-v2', render_mode='rgb_array')
env.reset(seed=args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(env.observation_space.shape[0], 64)
        self.dropout = nn.Dropout(p=0.2)
        self.affine2 = nn.Linear(64, 4)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x: torch.Tensor):
        assert not torch.isnan(x).any()
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-4)
eps = np.finfo(np.float32).eps.item()
rewards_device = 'mps' if torch.backends.mps.is_available() else 'cpu'
rewards = Rewards(env, 1e-3, torch.device(rewards_device))


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def policy_training_step(policy_losses: list[list[torch.Tensor]]):
    optimizer.zero_grad()
    loss_sums = [torch.cat(policy_loss).sum() for policy_loss in policy_losses]
    batch_loss = reduce(lambda a, b: a + b, loss_sums).mean() # maybe sum?
    # mean: each trajectory has the same weight regardless of its length
    print("Policy loss sum", batch_loss.item())
    batch_loss.backward()
    optimizer.step()


def discounted_policy_loss(i_episode, r_hat: torch.Tensor, plot) -> list[torch.Tensor]:
    # pad_sequence https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html
    R = 0
    policy_loss = []
    returns = deque()
    # r_hat is a sequence of r_hat estimates from our reward estimator
    for r in r_hat.detach().cpu().numpy()[::-1]:
        R = r + args.gamma * R
        returns.appendleft(R)
    returns = torch.tensor(np.array(returns))
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    # print("Policy losses", [p.item() for p in policy_loss])
    if plot is not None:
        plot.plot([p.item() for p in policy_loss])
    return policy_loss


# Returns List[Tuple[ObsType, ActType, float]]
def generate_trajectory(state, record=False):
    trajectory = []

    venv = VideoRecorder(env, base_path='figures/video') if record else None
    for _ in range(1, 500):  # Don't infinite loop while learning
        action = select_action(state)
        obs, reward, done, _, _ = env.step(action)
        if venv is not None:
            venv.capture_frame()
        trajectory.append((
            obs,
            action,
            reward,
        ))
        if args.render:
            env.render()
        if done:
            break
    if venv is not None:
        venv.close()
    return trajectory

def main():
    running_reward = 10
    database = []
    for i_episode in count(1):
        if i_episode == 1000:
            global env
            env = gym.make("LunarLander-v2", render_mode='human')
        state, _ = env.reset()
        ep_reward = 0

        # TODO: pass trajectories to discriminator
        NUM_PAIRS = 80 if i_episode < 2 else 80 # i guess this is a hyperparameter
        for _ in range(NUM_PAIRS):
            state, _ = env.reset()
            traj1 = generate_trajectory(state)
            state, _ = env.reset()
            traj2 = generate_trajectory(state)
            feedback = compare(traj1, traj2)
            database.append((traj1, traj2, feedback))
        print(" Database size", len(database))
        # database type: List[Tuple[Trajectory, Trajectory, Tuple[float, float]]]

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        state, _ = env.reset()
        NUM_REINFORCE_TRAJ = 20 # hyperparameter: how many trajectories per batch?
        policy_losses: list[list[torch.Tensor]] = []
        plt.clf()
        fig, axes = plt.subplots(1, 2)
        for i in range(NUM_REINFORCE_TRAJ):
            state, _ = env.reset()
            training_traj = generate_trajectory(state, record=i == 0)
            policy_loss = discounted_policy_loss(i_episode,
                                                 rewards.estimate_reward(training_traj),
                                                 plot=axes[0] if i % 5 == 0 else None)
            discounted_policy_loss(i_episode,
                                   torch.Tensor([s[2] for s in training_traj]),
                                   plot=axes[1] if i % 5 == 0 else None)
            del policy.rewards[:]
            del policy.saved_log_probs[:]
            policy_losses.append(policy_loss)
            print('mean reward', sum([reward for _, _, reward in training_traj]) / (len(training_traj) + 0.1))
        policy_training_step(policy_losses)
        fig.suptitle(f'Policy loss (episode {i_episode})')
        axes[0].set_title('Predicted')
        axes[1].set_title('Actual')
        plt.savefig(f'figures/policy-losses-{i_episode}.png', dpi=200)

        rewards.update(rewards.loss(database))

        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break
    
    env = gym.make("LunarLander-v2", render_mode='human')
    state, _ = env.reset()
    for t in range(1, 10000):  # Don't infinite loop while learning
        action = select_action(state)
        state, _, done, _, _ = env.step(action)
        if args.render:
            env.render()
        if done:
            break


if __name__ == '__main__':
    main()