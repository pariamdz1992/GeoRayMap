"""RadioMapSeer loader: a port of RadioUNet_c from RadioUNet, with its standard split.

Adapted from https://github.com/RonLevie/RadioUNet (lib/loaders.py, class RadioUNet_c),
MIT License, Copyright (c) 2019 Ron Levie (see radiounet_ray/LICENSE-RadioUNet). Kept: the
seed-42 map shuffle and the split (train = shuffled positions 0-500, validation = 501-600,
test = 601-699, 80 transmitters per map), the file layout, the threshold transform of the targets
and the input scaling. Dropped: the random-simulation and missing-building variants, which the
paper does not use. Each item is (inputs, target, "<map>_<tx>").
"""
import os

import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms


class RadioMapSeer(Dataset):
    SPLITS = {"train": (0, 500), "val": (501, 600), "test": (601, 699)}

    def __init__(self, data_dir, phase="train", simulation="DPM", cars=False, thresh=0.2,
                 num_tx=80):
        """
        Args:
            data_dir: RadioMapSeer root, the folder that holds png/ and gain/.
            phase: "train", "val" or "test".
            simulation: target simulator, "DPM" (dominant path) or "IRT2" (ray tracing).
            cars: use the maps with cars (targets in gain/cars<simulation>/, and a third input
                channel with the cars).
            thresh: threshold of the target transform (0.2, RadioUNet's noise floor).
            num_tx: transmitters per map (80 in RadioMapSeer).
        """
        if phase not in self.SPLITS:
            raise ValueError(f"phase must be one of {list(self.SPLITS)}, got {phase!r}")
        if simulation not in ("DPM", "IRT2"):
            raise ValueError(f"simulation must be 'DPM' or 'IRT2', got {simulation!r}")
        # The same map order as RadioUNet_c's np.random.seed(42); np.random.shuffle(...),
        # without changing numpy's global random state.
        self.maps_inds = np.arange(0, 700, 1, dtype=np.int16)
        np.random.RandomState(42).shuffle(self.maps_inds)
        self.ind1, self.ind2 = self.SPLITS[phase]
        self.num_tx = num_tx
        self.thresh = thresh
        self.cars = cars
        self.dir_gain = os.path.join(data_dir, "gain", ("cars" if cars else "") + simulation)
        self.dir_buildings = os.path.join(data_dir, "png", "buildings_complete")
        self.dir_tx = os.path.join(data_dir, "png", "antennas")
        self.dir_cars = os.path.join(data_dir, "png", "cars")
        self.to_tensor = transforms.ToTensor()
        if not os.path.isdir(self.dir_gain):
            raise FileNotFoundError(f"{self.dir_gain} not found: is {data_dir} the RadioMapSeer root?")

    def __len__(self):
        return (self.ind2 - self.ind1 + 1) * self.num_tx

    def sample_name(self, idx):
        """'<map>_<tx>' of item idx, the file-name stem used by RadioMapSeer."""
        map_pos, tx = divmod(idx, self.num_tx)
        return f"{int(self.maps_inds[map_pos + self.ind1]) + 1}_{tx}"

    def __getitem__(self, idx):
        map_pos, tx = divmod(idx, self.num_tx)
        map_id = int(self.maps_inds[map_pos + self.ind1]) + 1
        name_map, name = f"{map_id}.png", f"{map_id}_{tx}.png"

        buildings = np.asarray(io.imread(os.path.join(self.dir_buildings, name_map)))
        tx_img = np.asarray(io.imread(os.path.join(self.dir_tx, name)))
        gain = np.expand_dims(np.asarray(io.imread(os.path.join(self.dir_gain, name))), axis=2) / 255

        # Threshold transform: gains below the threshold are clipped to it, then rescaled to [0, 1].
        if self.thresh > 0:
            mask = gain < self.thresh
            gain[mask] = self.thresh
            gain = gain - self.thresh * np.ones(np.shape(gain))
            gain = gain / (1 - self.thresh)

        if not self.cars:
            inputs = np.stack([buildings, tx_img], axis=2)   # uint8: ToTensor scales it to [0, 1]
        else:
            cars = np.asarray(io.imread(os.path.join(self.dir_cars, name_map))) / 256
            inputs = np.stack([buildings / 256, tx_img / 256, cars], axis=2)

        inputs = self.to_tensor(inputs).type(torch.float32)
        gain = self.to_tensor(gain).type(torch.float32)
        return inputs, gain, name[:-4]
