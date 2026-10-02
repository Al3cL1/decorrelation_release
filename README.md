# Learning to be Different: Component Specialization for Compositional Diffusion Policies

<!-- abstract -->

## Installation

```bash
# RLBench + RoboCasa (install CoppeliaSim 4.1.0 first: see the top of environment.yml)
conda env create -f environment.yml && conda activate decorr
pip install --no-deps "robocasa @ git+https://github.com/robocasa/robocasa@56e355ccc64389dfc1b8a61a33b9127b975ba681" lerobot==0.3.3
pip install tianshou==0.4.10 pygame hidapi pynput datasets==3.6.0 jsonlines torchcodec==0.7.0
pip install -e .

# LIBERO (then install LIBERO itself: see the top of environment-libero.yml)
conda env create -f environment-libero.yml && conda activate decorr-libero
pip install -e .
```

## Data

```bash
export DATA_ROOT=/path/to/data

# LIBERO
python fdp/env/libero/build_dataset.py --download --hdf5_dir $DATA_ROOT/libero/hdf5 --out_dir $DATA_ROOT/libero/zarr

# RoboCasa
python -m robocasa.scripts.download_datasets --source mimicgen --split pretrain --tasks \
    OpenMicrowave CloseMicrowave TurnOnMicrowave OpenDishwasher SlideDishwasherRack \
    CloseOven SlideOvenRack TurnOnStove PickPlaceCounterToStove PickPlaceCounterToOven
python fdp/env/robocasa/lerobot_to_zarr.py --n-eps 100 --out $DATA_ROOT/robocasa/zarr/atomics_v365_a12_raw_N1000.zarr

# RLBench
python fdp/env/rlbench/gen_data.py -t mt7 -c 350 -s $DATA_ROOT/rlbench/mt7_N350_tid.zarr
```

The LIBERO and RLBench writers need `zarr<3`.

## Usage

```bash
python train.py task=libero/atomics25                     # task configs in fdp/config/task/
python train.py task=rlbench/mt7_tid policy.ortho_coef=0.0 policy.affine_weights=false   # FDP baseline

python eval_libero.py -c <ckpt> -o <out_dir>              # also eval_robocasa.py, eval.py (RLBench)
python diagnostics.py -c <ckpt> -o <out_dir>
```

Toy task:

```bash
cd toy
python train.py --loss_mode flat_cos2_envelope --lambda_decorr 0.045 --use_router --affine_weights --seed 10 --output_dir <ckpt_dir>
python eval.py --ckpt_dir <ckpt_dir> --lambdas 0.045 --seed 10
```

Baselines (DP, SDP): see `baselines/README.md`.

## Citation

<!-- arXiv link -->

```bibtex
@inproceedings{laprevotte2026different,
  title     = {Learning to be Different: Component Specialization for Compositional Diffusion Policies},
  author    = {Laprevotte, Alec and Chen, Haonan and Han, Xiaoshen and Du, Yilun},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```

MIT license (`LICENSE`); third-party code and licenses in `THIRD_PARTY.md`.
