import numpy as np
from fdp.env.rlbench.env import RlbenchEnv
from typing import List, Optional


MT_TASKS = {
    # the paper's RLBench tasks (RLBench 1.2.0); task_id = position in this list
    'mt7': [
        'close_box',
        'close_drawer',
        'close_fridge',
        'close_microwave',
        'toilet_seat_down',
        'take_umbrella_out_of_umbrella_stand',
        'reach_target',
    ],
}


def is_multitask(task_name: str) -> bool:
    return task_name in MT_TASKS


def get_subtasks(task_name: str) -> List[str]:
    if is_multitask(task_name):
        return MT_TASKS[task_name]
    else:
        return [task_name]


def take_a_glance(task_names: List[str], 
    camera_name: Optional[str]) -> List[np.ndarray]:
    env = RlbenchEnv(image_size=512)
    obs = []
    for task_name in task_names:
        env.set_task(task_name)
        if camera_name is None:
            obs.append(env.render())
        else:
            obs.append(env.reset()[0][camera_name + '_rgb'])
    env.close()
    return obs
