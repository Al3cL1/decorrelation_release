# Baselines: Diffusion Policy and SDP

The DP rows of Table 2 and the DP / SDP numbers of §4.5 come from
[Diffusion Policy](https://github.com/real-stanford/diffusion_policy) (Chi et al.) at commit
`5ba07ac`, with the files in `diffusion_policy/` here laid over it:

| file | what |
|---|---|
| `config/task/{rlbench_mt7,robocasa_atomics_H16_12D_raw,libero_atomics25}_tid.yaml` | DP task configs that read the same zarrs as the FDP configs |
| `dataset/robocasa_zarr_dataset.py` | generic zarr dataset for those configs (HWC → CHW on read) |
| `env_runner/dummy_image_runner.py` | no-op runner; rollouts run through FDP's runners (below) |
| `policy/sdp_transformer_hybrid_image_policy.py` | SDP baseline |
| `model/moe/task_moe.py`, `model/diffusion/transformer_for_diffusion_moe.py` | taken verbatim from [SDP](https://github.com/AnthonyHuo/SDP) (MIT) |
| `common/{normalize_util,replay_buffer}.py`, `model/common/lr_scheduler.py` | upstream files with zarr 3 / current diffusers compatibility fixes |

`eval_dp_{rlbench,robocasa,libero}.py` roll a DP checkpoint out through FDP's own runners, so
DP and FDP are scored by identical rollout code.

## Setup

Use the same environments as FDP (`decorr` for RLBench / RoboCasa, `decorr-libero` for LIBERO).

```bash
git clone https://github.com/real-stanford/diffusion_policy.git && cd diffusion_policy
git checkout 5ba07ac
cp -r <this repo>/baselines/diffusion_policy/diffusion_policy/. diffusion_policy/
export PYTHONPATH=$PWD:$PYTHONPATH      # upstream has no __init__.py, so use the path
```

## Training (run from the Diffusion Policy clone)

All runs use the upstream transformer-hybrid workspace with in-training rollouts off and
top-k checkpoints on validation loss:

```bash
COMMON="--config-name=train_diffusion_transformer_hybrid_workspace horizon=16 n_action_steps=8 \
  training.rollout_every=99999 checkpoint.topk.monitor_key=val_loss checkpoint.topk.mode=min"
FMT="checkpoint.topk.format_str='ep-{epoch:04d}_vl-{val_loss:.4f}.ckpt'"

# Table 2, RLBench DP row
python train.py $COMMON "$FMT" task=rlbench_mt7_tid policy.n_layer=10 \
  dataloader.batch_size=256 val_dataloader.batch_size=256 training.num_epochs=1001 training.checkpoint_every=25

# Table 2, RoboCasa DP row
python train.py $COMMON "$FMT" task=robocasa_atomics_H16_12D_raw_tid policy.n_layer=10 \
  dataloader.batch_size=128 val_dataloader.batch_size=128 training.num_epochs=500 training.checkpoint_every=25

# §4.5, LIBERO DP
python train.py $COMMON "$FMT" task=libero_atomics25_tid policy.n_layer=8 n_obs_steps=2 \
  dataloader.batch_size=128 val_dataloader.batch_size=128 training.num_epochs=200 \
  training.checkpoint_every=50 training.val_every=10

# §4.5, LIBERO SDP
python train.py $COMMON "$FMT" task=libero_atomics25_tid policy.n_layer=8 n_obs_steps=2 \
  policy._target_=diffusion_policy.policy.sdp_transformer_hybrid_image_policy.SDPTransformerHybridImagePolicy \
  +policy.n_tasks=1 +policy.w_MI=0.0 \
  dataloader.batch_size=128 val_dataloader.batch_size=128 training.num_epochs=200 \
  training.checkpoint_every=50 training.val_every=10
```

`policy.n_layer=10` matches the trunk parameters of FDP at K=4.

## Evaluation (run from this repo, with the clone on `PYTHONPATH`)

```bash
python baselines/eval_dp_rlbench.py  -c <run>/checkpoints/latest.ckpt -o <out_dir> --test_start_seed 7
python baselines/eval_dp_robocasa.py -c <run>/checkpoints/latest.ckpt -o <out_dir> --test_start_seed 7
python baselines/eval_dp_libero.py   -c <run>/checkpoints/<lowest ep-*_vl-*.ckpt> -o <out_dir> --test_start_seed 7
python baselines/eval_dp_libero.py   -c <sdp run>/checkpoints/<lowest ep-*_vl-*.ckpt> -o <out_dir> --test_start_seed 7 --ddim_steps 10
```

As for FDP, the RLBench / RoboCasa rows use the final checkpoint and the LIBERO numbers the
best-validation one; SDP is sampled with 10 DDIM steps, the sampler FDP uses. RoboCasa and
LIBERO render headless with `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`.
