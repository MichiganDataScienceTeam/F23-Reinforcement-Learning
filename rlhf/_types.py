from typing import TypeVar, List, Tuple

ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')

Trajectory = List[Tuple[ObsType, ActType, float]]
FeedbackRow = Tuple[Trajectory, Trajectory, Tuple[float, float]]
Database = List[FeedbackRow]