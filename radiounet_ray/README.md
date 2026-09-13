# RadioUNet + ray channels (Table 2, middle block)

This folder tests whether the ray channels also help a second backbone trained with its own
recipe. `train.py` is a script version of RadioUNet's official notebook `RadioWNet_c_DPM_Thr2.ipynb`
([RonLevie/RadioUNet](https://github.com/RonLevie/RadioUNet)): RadioWNet, batch 15, Adam
(lr 1e-4), StepLR(30, 0.1), MSE loss, 50 epochs for the first U-Net and then 50 for the second.
With `RU_RAY=1`, the four GeoRayMap channels are appended to RadioWNet's inputs (6 instead of 2)
and nothing else changes.

| Model | NMSE (10⁻³) ↓ | PSNR (dB) ↑ | SSIM ↑ | ms ↓ | Params (M) |
|---|---|---|---|---|---|
| RadioUNet | 10.68 | 34.45 | 0.9240 | 2.56 | 13.3 |
| RadioUNet + ray channels (N = 16) | **8.03** | **35.77** | **0.9363** | 3.30 | 13.3 |

Same protocol as the rest of the paper: full test split, thresholded targets, batch size 1, one
H100. `test.py` times the model call with CUDA events.

## Run (from this folder)

```bash
export RADIOMAPSEER_DIR=/path/to/RadioMapSeer/   # the trailing slash is required
python train.py                  # RadioUNet             -> Results/RadioUNet_c_DPM_Thr2/
RU_RAY=1 python train.py         # + ray channels        -> Results/RadioUNet_c_DPM_Thr2_ray/
python test.py                   # evaluate the second U-Net; appends a row to a CSV file
RU_RAY=1 python test.py
```

Other settings (`RU_EPOCHS`, `RU_BATCH`, `RU_STAGES`, `RU_CKPT`, ...) are environment variables,
listed at the top of `train.py` and `test.py`. To evaluate the released weights:
`RU_RAY=1 RU_CKPT=radiounet_ray_n16.pt python test.py` and `RU_CKPT=radiounet.pt python test.py`.

`lib/modules.py` and `lib/loaders.py` are RadioUNet's files, unchanged (MIT License, see
`LICENSE-RadioUNet`); `lib/ray_wnet.py` adds the ray channels from `georaymap/rays.py`. These files
need `pandas` and `matplotlib` in addition to the main requirements.
