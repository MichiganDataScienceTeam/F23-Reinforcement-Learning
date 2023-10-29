"""Module that implements comparison of two trajectories using gymnasium."""

import gymnasium
from typing import TypeVar

ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')

def compare(
    trajectory1: list[tuple[ObsType, ActType, float]],
    trajectory2: list[tuple[ObsType, ActType, float]],
) -> tuple[float, float] | None:
    """Return a distribution mu over {1, 2} representing
    the preference toward trajectory1 or trajectory2.
    
    If trajectory1 is better than trajectory2, return (1, 0).
    If they are the same, return (0.5, 0.5).
    If trajectory2 is better than trajectory1, return (0, 1).
    If the trajectories are not comparable, return None.
    """
    # mean instead of sum?
    reward_1 = sum(reward for _, _, reward in trajectory1)
    reward_2 = sum(reward for _, _, reward in trajectory2)

    if reward_2 < reward_1:
        return (1, 0)
    elif reward_1 < reward_2:
        return (0, 1)
    else:
        return (0.5, 0.5)
