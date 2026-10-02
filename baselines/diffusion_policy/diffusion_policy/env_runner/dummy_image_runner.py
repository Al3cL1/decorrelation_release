"""No-op env_runner, so the workspace builds without simulator dependencies; rollouts run separately."""
from typing import Dict

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class DummyImageRunner(BaseImageRunner):
    def __init__(self, output_dir, **kwargs):
        super().__init__(output_dir)
        # ignore everything else (shape_meta, task_name, n_test, etc.)

    def run(self, policy: BaseImagePolicy) -> Dict:
        return {}
