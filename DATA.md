## Dataset and Pretrained Weight Download and Storage Layout

The current training and validation configs expect the following datasets:

| Dataset / Weight | Used in | Download / Placement |
| --- | --- | --- |
| Hypersim | training | [Download Link](https://huggingface.co/datasets/ritianyu/Hypersim) |
| Real-World Benchmark (KITTI, ETH3D, NYU, ScanNet, DIODE) | validation | [Download Link](https://huggingface.co/datasets/ritianyu/Depth_Eval_Datasets) |
| Synthetic Benchmark (CyberPunk, DeadIsland, Spiderman2, SpidermanMM, WatchDogLegion) | validation | [Download Link](https://huggingface.co/datasets/ritianyu/game_4k_data) |
| DINOv3 `vitl16` checkpoint | train infinidepth from scratch | place at `${commonspace}/pretrained_models/dinov3/dinov3_vitl16_pretrain_lvd1689m.pth` |

After downloading and unpacking them, place them under `${commonspace}` like this:

```text
${commonspace}/
├── datasets/
│   ├── Kitti/
│   ├── ETH3D/
│   ├── nyu/
│   ├── scannet/
│   ├── DIODE/
│   ├── cyberpunk/
│   ├── deadisland/
│   ├── spiderman2/
│   ├── spidermanmm/
│   └── watchdoglegion/
├── processed_datasets/
│   ├── hypersim/
│   │   └── train.txt
│   │   └── val.txt
│   ├── Kitti/
│   │   └── val.txt
│   ├── ETH3D/
│   │   └── val.txt
│   ├── nyu/
│   │   └── val.txt
│   ├── scannet/
│   │   └── val.txt
│   ├── DIODE/
│   │   └── val.txt
│   ├── cyberpunk/
│   │   └── val.txt
│   ├── deadisland/
│   │   └── val.txt
│   ├── spiderman2/
│   │   └── val.txt
│   ├── spidermanMM/
│   │   └── val.txt
│   └── watchdoglegion/
│       └── val.txt
└── pretrained_models/
    └── dinov3/
        └── dinov3_vitl16_pretrain_lvd1689m.pth
```

**Current config paths**

- data root: `${commonspace}/datasets/***`
- meta file: `${commonspace}/processed_datasets/***/train.txt`, `${commonspace}/processed_datasets/***/val.txt`
- DINOv3 backbone weight: `${commonspace}/pretrained_models/dinov3/dinov3_vitl16_pretrain_lvd1689m.pth`

**Meta file format**

Each line in a dataset meta file is interpreted relative to the dataset `data_root` and should be one of:

```text
rgb_rel_path depth_rel_path
rgb_rel_path depth_rel_path prompt_depth_rel_path
```

**Note**

If you want to train/val on other datasets, you can prepare the data and meta file in the same format above, then modify the training/validation config to point to your new meta file and data root. You can also merge multiple meta files together and use `--include` to load them simultaneously.


</details>
