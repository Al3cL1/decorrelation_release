# Third-party components

| Component | How it is used | Version the paper used | License |
|---|---|---|---|
| Factorized Diffusion Policy (Liu et al., RA-L 2026) | `fdp/` is a fork of [Chaoqi-LIU/fdp](https://github.com/Chaoqi-LIU/fdp); redistributed with the authors' permission | upstream `6c7426f` | MIT (see `LICENSE`) |
| Diffusion Policy (Chi et al.) | `baselines/diffusion_policy/` is an overlay of new and patched files for [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy); the DP baselines | `5ba07ac` | MIT (`baselines/diffusion_policy/LICENSE`) |
| Sparse Diffusion Policy (SDP) | `baselines/diffusion_policy/diffusion_policy/model/moe/task_moe.py` and `.../model/diffusion/transformer_for_diffusion_moe.py` taken verbatim from [AnthonyHuo/SDP](https://github.com/AnthonyHuo/SDP) | — | MIT (same notice as Diffusion Policy) |
| LeRobot | dependency of `fdp/env/robocasa/lerobot_to_zarr.py`, whose `_PatchedLeRobotDataset.__getitem__` adapts `LeRobotDataset.__getitem__` | `0.3.3` | Apache-2.0 |
| robosuite | dependency | `ARISE-Initiative/robosuite@aaa8b9b2` (RoboCasa/RLBench env), `1.4.0` (LIBERO env) | MIT |
| RoboCasa | dependency | `robocasa/robocasa@56e355c` | MIT |
| LIBERO | dependency | `Lifelong-Robot-Learning/LIBERO@8f1084e` | MIT |
| RLBench (fork) | dependency | `Chaoqi-LIU/RLBench@b80e51f` | see upstream RLBench |
| PyRep, CoppeliaSim | RLBench backend, installed by the user | CoppeliaSim 4.1.0 Edu | see their licenses |
| robomimic | dependency (vision encoder) | `0.2.0` | MIT |
