import os
from os.path import join
import cv2
import h5py
import hydra
import numpy as np
from omegaconf import DictConfig
from training.utils.logger import Log
from torchvision.transforms import Compose

EPS = 1e-6


class Dataset:
    def __init__(self, **kwargs):
        super().__init__()
        self.cfg = DictConfig(kwargs)
        self.dataset_name = self.cfg.get("dataset_name", "unknown")
        self.num_files = self.cfg.get("num_files", 0)
        self.split = self.cfg.get("split", "train")
        self.fov_ratio = self.cfg.get("fov_ratio", 1.5)
        self.num_pts = self.cfg.get("num_pts", None)
        self.params = None
        self.build_metas()
        self.build_transforms()
        log_str = f"{self.cfg.split} split of {self.dataset_name} dataset: {len(self.rgb_files)} frames in total."
        if self.num_files > 0:
            log_str += f" (Sampling num_files: {self.num_files})"

        Log.info(log_str)

    def update_params(self, params):
        self.params = params

    def build_metas(self):
        """
        Prepare dataset file paths including RGB images, depth maps, and prompt depth maps.

        This method reads a metadata file that contains paths to the dataset files,
        validates their existence, and stores them for later use.
        """
        # Get metadata file path from config
        meta_path = self.cfg.get("meta_path")

        # Read all lines from the metadata file
        with open(meta_path) as f:
            lines = f.readlines()

        # Initialize file path lists
        self.rgb_files = []
        self.depth_files = []
        self.prompt_files = []

        # Flag to check file existence only once for performance
        file_validated = False

        # Process each line in the metadata file
        for line in lines:
            line = line.strip()
            splits = line.split(" ")

            # Parse file paths based on format (2 or 3 columns)
            if len(splits) == 2:
                rgb_path, depth_path = splits
                # If prompt path not specified, use depth path
                prompt_depth_path = depth_path
                self.same_depth_as_prompt = True
            elif len(splits) == 3:
                rgb_path, depth_path, prompt_depth_path = splits
                self.same_depth_as_prompt = False
            else:
                raise ValueError(f"Invalid metadata format: {line}")

            # Construct full file paths
            data_root = self.cfg.get("data_root")
            rgb_file = join(data_root, rgb_path)
            depth_file = join(data_root, depth_path)
            prompt_depth_file = join(data_root, prompt_depth_path)

            # Validate file existence (only for the first set)
            if not file_validated:
                assert os.path.exists(rgb_file), f"RGB file does not exist: {rgb_file}"
                assert os.path.exists(depth_file), f"Depth file does not exist: {depth_file}"
                assert os.path.exists(prompt_depth_file), f"Prompt depth file does not exist: {prompt_depth_file}"
                file_validated = True

            # Store file paths
            self.rgb_files.append(rgb_file)
            self.depth_files.append(depth_file)
            self.prompt_files.append(prompt_depth_file)

        frames = self.cfg.get("frames", None)
        if frames is not None:
            start, end, skip = frames
            end = end if end != -1 else len(self.rgb_files)
            self.rgb_files = self.rgb_files[start:end:skip]
            self.depth_files = self.depth_files[start:end:skip]
            self.prompt_files = self.prompt_files[start:end:skip]

    def build_transforms(self):
        """
        Build the transformation pipeline for data augmentation.
        If no transforms are specified in config, use identity function.
        Otherwise, compose all transforms and log the transformation layers.
        """
        # Get transforms from config or use empty list as default
        transforms = self.cfg.get("transforms", [])

        # If no transforms specified, use identity function
        if len(transforms) == 0:
            self.transform = lambda x: x
            return

        # Prepare log message for transform layers
        log_str = f"{self.dataset_name} transform layers: \n"
        compose_transforms = []
        for idx, transform in enumerate(transforms):
            # Add newline after each transform except the last one
            log_str += (str(transform) + "\n") if idx != len(transforms) - 1 else str(transform)
            if isinstance(transform, dict) or isinstance(transform, DictConfig):
                transform = hydra.utils.instantiate(transform)
            compose_transforms.append(transform)

        # Log the transform layers
        Log.info(log_str)

        # Create the transform pipeline using torchvision's Compose
        self.transform = Compose(compose_transforms)

    def read_rgb(self, index):
        """
        Read the RGB image from the file path.
        """
        img_path = self.rgb_files[index]
        rgb = cv2.imread(img_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return np.asarray(rgb / 255.0).astype(np.float32)

    def read_rgb_name(self, index):
        """
        Read the RGB image name from the file path.
        """
        return self.dataset_name + "__".join(self.rgb_files[index].split("/")[-4:])

    def read_depth(self, index, depth=None):
        """
        Read the depth map from the file path.
        return depth, depth_mask, disparity, disparity_mask
        """
        if depth is None and hasattr(self, "depth_files") is False:
            return None, None, None, None

        # reading depth values
        if depth is not None:
            pass
        elif self.depth_files[index].endswith(".png"):
            depth_path = self.depth_files[index]
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 1000.0
        elif self.depth_files[index].endswith(".npz"):
            depth = np.load(self.depth_files[index])["data"]
        elif self.depth_files[index].endswith(".hdf5"):
            depth = h5py.File(self.depth_files[index])["dataset"]
            depth = np.asarray(depth)
        elif self.depth_files[index].endswith(".npy"):
            depth = np.load(self.depth_files[index])
        else:
            raise ValueError(f"Invalid depth file: {self.depth_files[index]}")
        # print(self.depth_files[index], depth.min(), depth.max())

        # parse depth
        if len(depth.shape) == 2:
            pass
        elif len(depth.shape) == 3 and depth.shape[2] == 1:
            depth = depth[:, :, 0]
        else:
            raise ValueError(f"Invalid depth file: {self.depth_files[index]}")

        # Some datasets store depth in float16 to save space. Promote here so
        # downstream quantile/stat computations stay numerically stable.
        depth = np.asarray(depth, dtype=np.float32)

        # valid mask
        depth_mask = np.logical_and(np.logical_and(depth > EPS, ~np.isnan(depth)), (~np.isinf(depth)))

        if depth_mask.sum() == 0:
            Log.warn(f"No valid mask in the depth map of {self.depth_files[index]}")
        if depth_mask.sum() != 0 and np.isnan(depth).sum() != 0:
            depth[np.isnan(depth)] = depth[depth_mask].max()
        if depth_mask.sum() != 0 and np.isinf(depth).sum() != 0:
            depth[np.isinf(depth)] = depth[depth_mask].max()

        # disparity
        disparity_mask = np.logical_and(depth_mask, depth > 0.01)
        disparity = np.zeros_like(depth)
        disparity[disparity_mask] = 1.0 / depth[disparity_mask]
        return depth, depth_mask.astype(np.uint8), disparity, disparity_mask.astype(np.uint8)

    def check_shape(self, rgb, dpt):
        assert rgb.shape[:2] == dpt.shape[:2], f"rgb.shape: {rgb.shape}, dpt.shape: {dpt.shape}"
        assert len(rgb.shape) == 3, f"rgb.shape: {rgb.shape}"
        assert len(dpt.shape) == 2, f"dpt.shape: {dpt.shape}"

    def read_prompt_depth(self, index, depth, depth_mask, disparity, disparity_mask):
        if hasattr(self, "prompt_files") and self.same_depth_as_prompt is False:
            prompt_depth = np.asarray(cv2.imread(self.prompt_files[index], cv2.IMREAD_ANYDEPTH) / 1000.0).astype(
                np.float32
            )
            depth, depth_mask, disparity, disparity_mask = self.read_depth(index=index, depth=prompt_depth)
        prompt_mask = np.logical_and(
            np.logical_and(np.logical_and(depth_mask == 1, disparity_mask == 1), depth > 0.0), disparity > 0.0
        ).astype(np.uint8)
        prompt_depth = depth.copy()
        prompt_disparity = disparity.copy()
        prompt_depth[prompt_mask == 0] = 0.0
        prompt_disparity[prompt_mask == 0] = 0.0
        return prompt_depth, prompt_disparity, prompt_mask

    def read_focal(self, index, **kwargs):
        # for future use
        return 300.0

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index, params = index
        else:
            params = None
        if self.num_files > 0:
            base_index = int(index / (self.num_files / len(self.rgb_files)))
            index = (np.random.randint(0, len(self.rgb_files)) + base_index) % len(self.rgb_files)
        else:
            index = index % len(self.rgb_files)
        repeat_num = 0 
        while True:
            rgb = self.read_rgb(index)
            out = self.read_depth(index)
            if not isinstance(out, (list, tuple)):
                out = (out,)
            dpt, msk, disp, disp_msk, highfreq_mask = (list(out) + [None] * 5)[:5]
            if dpt is not None:
                self.check_shape(rgb, dpt)
            sample = {
                "image": rgb,
            }
            if dpt is not None:
                sample["depth"] = dpt
                sample["mask"] = msk
                sample["disparity"] = disp
                sample["disparity_mask"] = disp_msk

            if highfreq_mask is not None:
                sample["highfreq_mask"] = highfreq_mask
                
            if hasattr(self, "prompt_files"):
                prompt_depth, prompt_disp, prompt_mask = self.read_prompt_depth(
                    index=index, depth=dpt, depth_mask=msk, disparity=disp, disparity_mask=disp_msk
                )
                sample["prompt_mask"] = prompt_mask
                sample["prompt_disparity"] = prompt_disp
                sample["prompt_depth"] = prompt_depth

            if hasattr(self, "focal"):
                sample["focal"] = self.read_focal(index=index)

            sample["dataset_name"] = self.dataset_name
            sample["image_name"] = self.read_rgb_name(index)
            sample["image_path"] = self.rgb_files[index]
            # Not use fov_ratio yet !
            # fov_ratio = max(1.0, np.random.uniform(2 - self.fov_ratio, self.fov_ratio))
            # sample["fov_ratio"] = fov_ratio
            if params is not None:
                sample["height"] = params["height"]
                sample["width"] = params["width"]

            # # for debug
            # if np.random.rand() < 0.1:
            #     sample["mask"] = np.zeros_like(sample["mask"])
            #     sample["disparity_mask"] = np.zeros_like(sample["disparity_mask"])

            sample = self.transform(sample)

            # for implicitpda
            if "reference_meta" in sample:
                if sample["reference_meta"]["depth_min"] == None or sample["reference_meta"]["disparity_min"] == None:
                    repeat_num += 1
                    index = int(np.random.randint(0, len(self.rgb_files)))
                    image_name = self.rgb_files[index]
                    if repeat_num >= 5:
                        Log.warn(f"No valid mask in the depth map of {image_name}.")
                    elif repeat_num > 10:
                        raise ValueError(f"No valid mask in the depth map of {image_name}.")
                else:
                    break
            else:
                # for pdav2/dav2
                if ("mask" not in sample or sample["mask"].sum() >= 10) and (
                    "disparity_mask" not in sample or sample["disparity_mask"].sum() >= 10
                ):
                    break
                else:
                    repeat_num += 1
                    index = int(np.random.randint(0, len(self.rgb_files)))
                    image_name = self.rgb_files[index]
                    if repeat_num >= 5:
                        Log.warn(f"No valid mask in the depth map of {image_name}.")
                    elif repeat_num > 10:
                        raise ValueError(f"No valid mask in the depth map of {image_name}.")

        return sample

    def __len__(self):
        return self.num_files if self.num_files > 0 else len(self.rgb_files)
