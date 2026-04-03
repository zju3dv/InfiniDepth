model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384], "layer_idxs": [2, 5, 8, 11]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768], "layer_idxs": [2, 5, 8, 11]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024], "layer_idxs": [4, 11, 17, 23]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536], "layer_idxs": [9, 19, 29, 39]},
}

dinov3_model_configs = {
    "vitl16":{'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'layer_idxs': [4, 11, 17, 23]},
    "vith16plus": {'encoder': 'vith', 'features': 384, 'out_channels': [1280, 1280, 1280, 1280], 'layer_idxs': [7, 15, 23, 31]},
}
