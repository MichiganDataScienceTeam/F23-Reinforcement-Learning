import argparse
import gymnasium as gym
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

from reward_estimator import Rewards


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.97, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=129, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('LunarLander-v2')
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

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()
rewards_device = 'mps' if torch.backends.mps.is_available() else 'cpu'
rewards = Rewards(env, 1e-2, torch.device(rewards_device))


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode(i_episode, r_hat: torch.Tensor):
    R = 0
    policy_loss = []
    returns = deque()
    # r_hat is a sequence of r_hat estimates from our reward estimator
    for r in r_hat.numpy()[::-1]:
        R = r + args.gamma * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    if i_episode > 300 and i_episode % 10 == 0:
        plt.clf()
        plt.plot(returns.detach().numpy())
        plt.savefig(str(i_episode) + '.png', dpi=200)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


# Returns List[Tuple[ObsType, ActType, float]]
def generate_trajectory(state):
    trajectory = [state]
    for _ in range(1, 10000):  # Don't infinite loop while learning
        action = select_action(state)
        obs, reward, done, _, _ = env.step(action)
        trajectory.append((
            obs,
            action,
            reward,
        ))
        if args.render:
            env.render()
        if done:
            break
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
        NUM_PAIRS = 10 # i guess this is a hyperparameter
        for _ in range(NUM_PAIRS):
            state, _ = env.reset()
            traj1 = generate_trajectory(state)
            state, _ = env.reset()
            traj2 = generate_trajectory(state)
            feedback = compare(traj2, traj2)
            database.append((traj1, traj2, feedback))
        # database type: List[Tuple[Trajectory, Trajectory, Tuple[float, float]]]

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        finish_episode(i_episode, rewards.estimate_reward(generate_trajectory(state)))

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
        state, reward, done, _, _ = env.step(action)
        if args.render:
            env.render()
        if done:
            break


if __name__ == '__main__':
    main()