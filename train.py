"""Train an FDP policy. Hydra: any key of fdp/config/train_factorpolicy.yaml can be overridden.

Usage:
    python train.py task=libero/atomics25
    accelerate launch train.py task=rlbench/mt7_tid          # multi-GPU
    python train.py task=rlbench/mt7_tid policy.ortho_coef=0.0 policy.affine_weights=false   # plain FDP
"""
import pathlib
import sys

FDP_ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(FDP_ROOT))

import hydra
from omegaconf import OmegaConf
from fdp.workspace.train_policy import TrainPolicyWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(FDP_ROOT / "fdp" / "config"),
    config_name="train_factorpolicy",
)
def main(cfg):
    print(f"\n=== TRAINING  name={cfg.name}  task={cfg.task_name} ===\n", flush=True)
    workspace = TrainPolicyWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
