#!/usr/bin/env bash
# The training runs behind the GeoRayMap rows of the paper, with the flags that were used.
#
#   DATA=/path/to/RadioMapSeer/ bash scripts/reproduce_paper.sh                    # print the commands
#   DATA=/path/to/RadioMapSeer/ LAUNCH=python bash scripts/reproduce_paper.sh      # run them in turn
#   DATA=/path/to/RadioMapSeer/ LAUNCH="sbatch scripts/slurm_job.sh" bash scripts/reproduce_paper.sh
#
# Each run takes about 10 h on one 80 GB H100 and ends with its test-set evaluation at batch
# size 1 (appended to results/local_runs.jsonl). The paper's numbers (in the comments) come from
# single runs without a fixed seed, so a rerun will differ slightly.
set -euo pipefail
: "${DATA:?set DATA=/path/to/RadioMapSeer/}"
LAUNCH=${LAUNCH:-echo python}
run() { $LAUNCH train.py --data_dir "$DATA" "$@"; }

# Table 1, and Table 2 top block: ConvNeXt U-Net on DPM targets (these runs used float16 autocast)
run --n_samples 16 --precision fp16                    # GeoRayMap, N = 16                  38.12 dB
run --n_samples 0  --precision fp16                    # without ray channels           36.51 dB
run --n_samples 64 --precision fp16                    # N = 64                         38.11 dB
run --n_samples 64 --precision fp16 --ssim_weight 0    # N = 64 without the SSIM term   38.01 dB
run --n_samples 64 --precision fp16 --learned_atten    # N = 64 + learned attenuation   37.93 dB

# Table 2 bottom block: IRT2 (ray-traced) targets, bfloat16
run --simulation IRT2 --n_samples 16 --precision bf16  # ray channels                   33.74 dB
run --simulation IRT2 --n_samples 0  --precision bf16  # without ray channels           32.17 dB

# Table 2 caption: the N = 64 model in bfloat16 (38.05 dB, 0.06 dB from float16)
run --n_samples 64 --precision bf16

# Table 2 middle block (RadioUNet backbone with and without ray channels): radiounet_ray/README.md
