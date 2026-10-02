import string
import gymnasium
import numpy as np
import robocasa
from robocasa.utils.env_utils import create_env
from robocasa.models.fixtures import FixtureType
from typing import List, Optional, Dict

# Fail fast if we accidentally import the legacy v0.1 robocasa instead of
# RoboCasa365. v0.1 has no __version__ attribute; v365 declares "1.0.0".
assert getattr(robocasa, "__version__", "0").startswith("1."), (
    "Expected RoboCasa365 (>=1.0). Got "
    f"version={getattr(robocasa, '__version__', 'unknown')}. "
    "Activate the `robocasa365` conda env."
)

# hard_mode: fixture the robot spawns at, away from the task's target
_HARD_DISTRACTOR = {
    'OpenDrawer':         FixtureType.SINK,
    'CloseDrawer':        FixtureType.SINK,
    'TurnOnSinkFaucet':   FixtureType.STOVE,
    'TurnOffSinkFaucet':  FixtureType.STOVE,
    'TurnSinkSpout':      FixtureType.STOVE,
    'StartCoffeeMachine': FixtureType.SINK,
    'TurnOnMicrowave':    FixtureType.SINK,
    'TurnOffMicrowave':   FixtureType.SINK,
}

# Task-id space of the training zarr (10 tasks, ids 0..9; do not reorder).
# The evaluated tasks are factory.MT_TASKS['atomics'].
_ATOMICS_TASKS = [
    'OpenMicrowave',                  # 0
    'CloseMicrowave',                 # 1
    'TurnOnMicrowave',                # 2
    'OpenDishwasher',                 # 3
    'SlideDishwasherRack',            # 4
    'CloseOven',                      # 5
    'SlideOvenRack',                  # 6
    'TurnOnStove',                    # 7
    'PickPlaceCounterToStove',        # 8
    'PickPlaceCounterToOven',         # 9
]
_TASK_ID_MAPS = {
    'atomics':    {n: i for i, n in enumerate(_ATOMICS_TASKS)},
}


class RobocasaEnv(gymnasium.Env):
    metadata = {
        "render_modes": ['rgb_array'],
        "render_fps": 20
    }

    def __init__(self,
        task_name: str,
        image_size: int = 128,
        seed: int = 42,
        camera_names: List[str] = [
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        # the v365 LeRobot datasets only have base-relative eef pose and gripper qpos
        state_ports: List[str] = [
            'robot0_base_to_eef_pos',
            'robot0_base_to_eef_quat',
            'robot0_gripper_qpos',
        ],
        video_camera: str = "robot0_agentview_right",
        video_resolution: int = 512,
        max_episode_steps: int = 500,
        enable_render: bool = True,
        hard_mode: bool = False,
        hard_distractor: Optional[FixtureType] = None,
        hard_layout_ids: Optional[int] = None,
        split: Optional[str] = None,
        task_group: str = 'atomics',
    ):
        super().__init__()
        if task_group not in _TASK_ID_MAPS:
            raise ValueError(
                f"task_group={task_group!r} not in {list(_TASK_ID_MAPS)}")
        self._task_id_map = _TASK_ID_MAPS[task_group]
        self._task_group = task_group

        # hard_mode: island layouts, robot spawned at a distractor fixture away from the target
        layout_ids = None
        if hard_mode:
            layout_ids = hard_layout_ids if hard_layout_ids is not None else -5  # ISLAND
            if hard_distractor is None:
                hard_distractor = _HARD_DISTRACTOR.get(task_name)
                if hard_distractor is None:
                    raise ValueError(
                        f"hard_mode=True but no default distractor fixture is "
                        f"registered for task_name='{task_name}'. Pass "
                        f"hard_distractor=FixtureType.<X> explicitly, or add "
                        f"an entry to _HARD_DISTRACTOR in env.py.")

        # create_env sets the renderer flags itself (from render_onscreen)
        if not enable_render:
            print("[RobocasaEnv] WARN: enable_render=False is ignored under "
                  "RoboCasa365; offscreen rendering is always on.")
        env = create_env(
            env_name=task_name,
            robots="PandaOmron",
            camera_names=list(set(list(camera_names) + [video_camera])),
            camera_widths=image_size,
            camera_heights=image_size,
            seed=seed,
            render_onscreen=False,
            layout_ids=layout_ids,
            split=split,
        )

        if hard_mode:
            # re-point init_robot_base_ref to the distractor on every reset
            _orig_setup = env._setup_kitchen_references
            _distractor = hard_distractor

            def _patched_setup():
                _orig_setup()
                env.init_robot_base_ref = env.get_fixture(_distractor)

            env._setup_kitchen_references = _patched_setup
            # takes effect from the next reset()

        self.env = env
        self.task_name = task_name
        # v365: reset once before get_ep_meta() (it needs layout_id)
        env.reset()
        self.task_prompt = env.get_ep_meta()['lang']
        self.state_ports = state_ports
        self.camera_names = camera_names
        self.video_camera = video_camera
        self.video_resolution = video_resolution
        self.max_episode_steps = max_episode_steps
        self.done = False

        # setup gym spaces
        obs_dict = env._get_observations()
        observation_space = gymnasium.spaces.Dict({})
        for port in state_ports:
            observation_space.spaces[port] = gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=obs_dict[port].shape, dtype=np.float32
            )
        for cam_name in camera_names:
            observation_space.spaces[f"{cam_name}_rgb"] = gymnasium.spaces.Box(
                low=0.0, high=1.0,
                shape=(3, image_size, image_size), dtype=np.float32
            )
        observation_space.spaces['prompt'] = gymnasium.spaces.Text(
            min_length=0, max_length=512,
            charset=string.printable
        )
        observation_space.spaces['task_id'] = gymnasium.spaces.Box(
            low=0, high=max(0, len(self._task_id_map) - 1),
            shape=(1,), dtype=np.float32
        )
        self.observation_space = observation_space
        self.action_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(env.action_dim,), dtype=np.float32
        )
    
    def _extract_obs(self, 
        raw_obs: Optional[Dict[str, np.ndarray]]=None
    ) -> Dict[str, np.ndarray]:
        if raw_obs is None:
            raw_obs = self.env._get_observations()

        obs_dict = {}

        # robot state
        for port in self.state_ports:
            obs_dict[port] = raw_obs[port].astype(np.float32)

        # rgb: CHW float in [0, 1], as in training
        for cam_name in self.camera_names:
            img = np.flip(raw_obs[f"{cam_name}_image"], axis=0)  # HWC uint8
            obs_dict[f"{cam_name}_rgb"] = (
                img.transpose(2, 0, 1).astype(np.float32) / 255.0
            )
            
        # prompt
        obs_dict['prompt'] = self.task_prompt

        # integer task id; range and meaning depend on task_group (see __init__)
        task_id = self._task_id_map.get(self.task_name, -1)
        obs_dict['task_id'] = np.array([task_id], dtype=np.float32)

        return obs_dict

    def step(self, action: np.ndarray):
        obs, reward, terminated, info = self.env.step(action)
        if self.env._check_success():
            reward = 1.0
        else:
            reward = 0.0
        self.done = self.done or terminated or (reward >= 1) \
            or (self.env.timestep >= self.max_episode_steps)
        return self._extract_obs(obs), reward, self.done, False, info

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        obs_dict = self._extract_obs(obs)
        self.done = False
        return obs_dict, {'prompt': self.task_prompt}

    def render(self, mode='rgb_array'):
        assert mode == 'rgb_array'
        frame = np.flip(self.env.sim.render(
            height=self.video_resolution, width=self.video_resolution, 
            camera_name=self.video_camera
        ), axis=0).astype(np.uint8)
        return frame

    def close(self):
        self.env.close()

    def seed(self, *args, **kwargs):
        pass
