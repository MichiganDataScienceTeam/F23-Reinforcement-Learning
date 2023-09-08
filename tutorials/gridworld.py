"""Simple Gridworld Environment for learning RL."""

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np


class GridWorld():
    """
    Simple Gridworld MDP for learning RL.
    Navigate from a "." to the Goal space "G".
    Passing through "X" spaces is not allowed

    Map:
    . . . .
    . . X .
    . . G X
    . X . .
    
    Actions:
        Any of cardinal directions, when possible
    
    Reward:
        -1 on all steps
    """
    metadata = {"render_modes":["human"]}

    def __init__(self, render_mode=None, random_start=True, max_steps=100):
        """Initialize environment."""

        self.render_mode = render_mode
        self.random_start = random_start

        # Environment Parameters
        self.action_space = np.arange(4)
        self.position_change = {0:-4, 1:1, 2:4, 3:-1}
        self.action_decoder = {0:"North", 1:"East", 2:"South", 3:"West"}
        self.action_encoder = {"North":0, "East":1, "South":2, "West":3}

        self.valid_spaces = np.array(
                            [0, 1, 2, 3,
                             4, 5,    7,
                             8, 9,    
                             12,   14,15])
        self.blocked_states = np.array([6, 11, 13])
        self.terminal_state = 10

        self.transition_dynamics = np.array([
            [0,1,1,0], [0,1,1,1], [0,1,0,1], [0,0,1,1],
            [1,1,1,0], [1,0,1,1], [0,0,0,0], [1,0,0,0],
            [1,1,1,0], [1,1,0,1], [0,0,0,0], [0,0,0,0],
            [1,0,0,0], [0,0,0,0], [1,1,0,0], [0,0,0,1]])
        
        self.reward_map = np.full(shape=16, fill_value=-1)

        self.current_state = None
        self.max_steps = max_steps
        self.current_step = 0
    
    def reset(self, start_location=None):
        """Reset environment."""
        if start_location is not None:
            assert start_location not in self.valid_spaces, f"starting location must be one of {self.valid_spaces}"
        
        if self.random_start:
            self.current_state = np.random.choice(self.valid_spaces)
        elif start_location:
            self.current_state = start_location
        else:
            self.current_state = 0
        
        obs = self._get_obs()

        return obs
    
    def step(self, action: int):
        """Take step in environment."""
        assert self.action_space.__contains__(action), f"{action} is an invalid action."
        
        terminated, truncated = (False, False)
        self.current_step += 1
        
        self.current_state = self._update_state(action)
        
        obs = self._get_obs()
        reward = self.reward_map[self.current_state]
        
        if self.current_state == self.terminal_state:
            terminated = True
        elif self.current_step == self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated
    
    def get_neighbors(self, state):
        """
        Get a list of the neighbors to the passed state for each direction
        """
       
        if state == self.terminal_state or state in self.blocked_states:
            return []
        north_n = state - 4 if state - 4 >= 0 else state
        east_n = state + 1 if (state + 1) % 4 > state % 4 else state
        south_n = state + 4 if state + 4 < 16 else state
        west_n = state - 1 if (state - 1) % 4 < state % 4 else state
        res = [north_n,east_n,south_n,west_n]
        
        for i in range(4):
            if res[i] in self.blocked_states:
                res[i] = state
        return res
        
    
    def _get_obs(self):
        """Get the current observation."""
        return self.current_state
    
    def _update_state(self, action):
        """Return the new state."""
        new_pos = self.get_neighbors(self.current_state)[action]

        return new_pos


###
if __name__=="__main__":
    env = GridWorld()

    obs = env.reset()
    term, trunc = False, False
    steps = 0
    while not term and not trunc:
        action = np.random.choice(env.action_space)
        obs, reward, term, trunc = env.step(action)
        steps += 1
    print(f"{obs} found after {steps} steps")
