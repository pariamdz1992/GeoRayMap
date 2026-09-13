# Results

- `paper_rows.csv`: every row of Tables 1 and 2 of the paper.
- `georaymap_runs.jsonl`: the GeoRayMap rows exactly as the evaluation code wrote them (full precision,
  including MSE and MAE).

**Protocol (all rows).** RadioMapSeer test split: the maps at shuffled positions 601–699, with
80 transmitters each, give 7,920 maps. Targets are DPM unless marked IRT2, after RadioUNet's
threshold transform: gains below 0.2 are clipped, then rescaled to [0, 1]. Metrics are averaged
over maps (batch size 1), and latency is per map at batch size 1 on one NVIDIA H100.

Run names that start with `raynet_` come from GeoRayMap's development code name.

| Run in `georaymap_runs.jsonl` | Row in the paper | Released weights |
|---|---|---|
| `raynet_srm_n16_bs1` | Table 1 GeoRayMap; Table 2 ray marching, N = 16 (ours) | `georaymap_srm_n16.pt` |
| `raynet_srm_n0_bs1` | Table 2 w/o ray marching (DPM) | `georaymap_srm_noray.pt` |
| `raynet_srm_bs1` | Table 2 ray marching, N = 64 | `georaymap_srm_n64.pt` |
| `raynet_srm_ssim0.0_bs1` | Table 2 N = 64 without SSIM term | `georaymap_srm_n64_nossim.pt` |
| `raynet_srm_learned_bs1` | Table 2 N = 64 + learned attenuation | `georaymap_srm_n64_learned.pt` |
| `raynet_srm_n16_irt2_bf16_bs1` | Table 2 IRT2, ray marching, N = 16 | `georaymap_irt2_n16.pt` |
| `raynet_srm_n0_irt2_bf16_bs1` | Table 2 IRT2, w/o ray marching | `georaymap_irt2_noray.pt` |
| `raynet_srm_bf16_bs1` | not in the paper: N = 64 in bfloat16, 0.06 dB from float16 (control for the bfloat16 IRT2 rows) | – |
| `raynet_srm_ssim0.0_edge0.0_bf16_bs1` | not in the paper: N = 64 with the L1 loss only, bfloat16 | – |

The two RadioUNet rows (with and without ray channels) were written by `radiounet_ray/test.py`;
their weights are `radiounet.pt` and `radiounet_ray_n16.pt`. The other baselines of Table 1
(RME-GAN, UVM-Net, and our reproductions of RadioDiff and RadioFlow) were trained and measured
with the same protocol, using code that is not part of this repository.
