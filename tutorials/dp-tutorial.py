"""
Value Iteration Lab

Given a very simple MDP, evaluate the optimal value function v* using any method you like.

HINT: The easiest is the closed-form, linear algebra solution


MDP: Gridworld

. . . .
. . X .
. . G X
. X . .

"""
import numpy as np

class GridWorld():

    def __init__(self, gamma=0.9):
        """
        Init the Gridworld
        """
        assert(gamma <= 1 and gamma >= 0)
        self.gamma = gamma

        self.num_states = 16
        self.num_actions = 4

        self.terminal_state = 10
        self.blocked_states = [6,11,13]


    def get_MDP(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Get the 5-tuple for the MRP
        
        Return:
            states:   list of all states      shape: (16,)
            actions:  list of all actions    shape: (4,)
                      0:North, 1:East, 2:South, 3:West
            dynamics: transition dynamics   shape: (16,4,16)
                      S x A -> list of Probs to go to each state
            rewards:  reward map            shape: (16,4)
                      Simply return -1 for all
            gamma: discount factor
        """

        states = np.array(list(range(self.num_states)))
        actions = np.array(list(range(self.num_actions)))

        dynamics = np.full((self.num_states,self.num_actions,self.num_states), 0)
        for s in range(self.num_states):
            neighs = self.get_neighbors(s)
            if neighs:
                for i in range(self.num_actions):
                    dynamics[s,i,neighs[i]] = 1

        rewards = np.full((self.num_states, self.num_actions),-1)
        rewards[self.terminal_state,:] = np.full(self.num_actions, 0)
        for b in self.blocked_states:
            rewards[b,:] = np.full(self.num_actions, 0)

        return (states, actions, dynamics, rewards, self.gamma)


    def get_neighbors(self, state: int) -> np.ndarray:
        """
        Get legal neighbors of passed in state.

        The dynamics for the GridWorld are hard-coded here
        """
        assert(state >= 0 and state <= 15)

        if state == self.terminal_state or state in self.blocked_states:
            return []

        sz = 4
        north_n = state - sz if state - sz >= 0 else state
        east_n = state + 1 if (state + 1) % sz > state % sz else state
        south_n = state + sz if state + sz < self.num_states else state
        west_n = state - 1 if (state - 1) % sz < state % sz else state
        res = [north_n,east_n,south_n,west_n]
        
        for i in range(4):
            if res[i] in self.blocked_states:
                res[i] = state
        return res


###

def init_random_policy(num_states, num_actions):
    """
    Initialize a random policy based on MDP's possible states and actions
    
    Arguments:
        num_states: number of states in MDP
        num_actions: number of action in MDP
    
    Returns:
        policy: ndarray of arbitray probs   shape: (num_states, num_actions)
                S -> list of probs for taking each action
    """
    policy = np.random.rand(num_states, num_actions)
    return policy/policy.sum(axis=1)[:,None]


def print_values(values: np.ndarray):
    """
    Print values.
    """
    assert(values.shape == (16,))
    for i in range(4):
        for j in range(4):
            v = i * 4 + j
            print(v_star[v], end="  ")
        print()

def print_max_dir(values: np.ndarray, grid: GridWorld):
    """
    Print direction of maximum value.
    """
    act_to_dir = {0:"^",1:">",2:"v",3:"<"}
    assert(values.shape == (16,))
    for i in range(4):
        for j in range(4):
            v = i * 4 + j
            if v in grid.blocked_states:
                print("X", end=" ")
            elif v == grid.terminal_state:
                print("G", end=" ")
            else:
                neighs = grid.get_neighbors(v)
                neigh_vals = [values[n] for n in neighs]
                print(act_to_dir[np.argmax(neigh_vals)], end=" ")
        print()

###

env = GridWorld()
states, actions, dynamics, rewards, gamma = env.get_MDP()

policy = init_random_policy(states.shape[0], actions.shape[0])


"""
Below, compute V* using the information provided above.

Assuming you make no temp variables (why), this is possible in 3 steps

HINT: some useful functions
np.diagonal
np.matmul
np.linalg.inverse
np.identity
np.transpose

"""

r_pi = np.array([0,0,0])

p_pi = np.array([0,0,0])


v_star = np.full(16, fill_value=0)
v_star = np.round(v_star,3)


# Print Values
print_values(v_star)

print("---")

# Print in increasing direction of value (the policy?!?!?!?!)
print_max_dir(v_star, env)
