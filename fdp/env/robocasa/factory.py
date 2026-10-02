from typing import List


MT_TASKS = {
    # the 7 evaluated tasks (task ids come from env._ATOMICS_TASKS)
    'atomics': [
        'CloseMicrowave',
        'SlideDishwasherRack',
        'CloseOven',
        'OpenMicrowave',
        'OpenDishwasher',
        'TurnOnMicrowave',
        'TurnOnStove',
    ],
}


def is_multitask(task_name: str) -> bool:
    return task_name in MT_TASKS


def get_subtasks(task_name: str) -> List[str]:
    if is_multitask(task_name):
        return MT_TASKS[task_name]
    else:
        return [task_name]
