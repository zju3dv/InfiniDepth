import bisect
from copy import deepcopy
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from torch.utils.data import ConcatDataset, DataLoader


class MyConcatDataset(ConcatDataset):
    def __getitem__(self, index):
        if isinstance(index, tuple):
            idx, params = index
            if idx < 0:
                idx += len(self)
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            ds = self.datasets[dataset_idx]
            return ds[(sample_idx, params)]
        return super().__getitem__(index)


class GeneralDataModule(pl.LightningDataModule):
    default_train_loader_opts = DictConfig(
        {
            "batch_size": 1,
            "num_workers": 4,
            "shuffle": True,
            "pin_memory": True,
            "drop_last": True,
            "persistent_workers": True,
        }
    )
    default_val_loader_opts = DictConfig(
        {
            "batch_size": 1,
            "num_workers": 1,
            "shuffle": False,
            "pin_memory": False,
            "drop_last": False,
            "persistent_workers": True,
        }
    )

    def __init__(
        self,
        train_dataset: DictConfig = None,
        val_dataset: DictConfig = None,
        test_dataset: DictConfig = None,
        train_loader_opts: DictConfig = None,
        val_loader_opts: DictConfig = None,
        **kwargs,
    ):
        """
        Initialize the GeneralDataModule with datasets and loader options.

        This is a general datamodule that can be used for any dataset.
        Train uses ConcatDataset. Val and Test use CombinedLoader, sequentially
        consuming each iterable and returning a triplet (data, idx, iterable_idx).

        Args:
            train_dataset (DictConfig): Configuration for the training dataset.
            val_dataset (DictConfig): Configuration for the validation dataset.
            test_dataset (DictConfig): Configuration for the test dataset.
            train_loader_opts (DictConfig): Options for the training data loader.
            val_loader_opts (DictConfig): Options for the validation data loader.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_loader_opts = self.default_train_loader_opts
        self.val_loader_opts = self.default_val_loader_opts

        if train_loader_opts is not None:
            self.train_loader_opts.update(train_loader_opts)
        if val_loader_opts is not None:
            self.val_loader_opts.update(val_loader_opts)
        self.train_loader_opts.persistent_workers = True if self.train_loader_opts.num_workers > 0 else False
        self.val_loader_opts.persistent_workers = True if self.val_loader_opts.num_workers > 0 else False

    def val_dataloader(self):
        """
        Create and return the validation data loader.

        Returns:
            CombinedLoader or DataLoader: The validation data loader.
        """
        return GeneralDataModule._parse_loaders(self.val_dataset, self.val_loader_opts)

    def test_dataloader(self):
        """
        Create and return the test data loader.

        Returns:
            CombinedLoader or DataLoader: The test data loader.
        """
        return GeneralDataModule._parse_loaders(self.val_dataset, self.val_loader_opts)

    def train_dataloader(self):
        """
        Create and return the training data loader.

        Returns:
            DataLoader: The training data loader.
        """
        return GeneralDataModule._parse_train_dataloader(self.train_dataset, self.train_loader_opts)

    @staticmethod
    def _parse_train_dataloader(config, loader_opts):
        """
        Parse and create the training data loader from the configuration.

        Args:
            config (DictConfig): Configuration for the dataset.
            loader_opts (DictConfig): Options for the data loader.

        Returns:
            DataLoader or CombinedLoader: The training data loader.
        """
        datasets = GeneralDataModule._parse_datasets(config)
        dataset = MyConcatDataset(datasets)
        if "loader_opts" in config:
            loader_opts = deepcopy(loader_opts)
            loader_opts.update(config.loader_opts)
        if "sampler_opts" in config:
            loader_opts = deepcopy(loader_opts)
            sampler_opt = config.sampler_opts
            batch_sampler = instantiate(
                sampler_opt,
                dataset=dataset,
                batch_size=loader_opts.batch_size,
                shuffle=loader_opts.get("shuffle", True),
                drop_last=loader_opts.get("drop_last", True),
            )
            del loader_opts["batch_size"]
            del loader_opts["shuffle"]
            del loader_opts["drop_last"]
            return DataLoader(dataset, batch_sampler=batch_sampler, **loader_opts)
        else:
            return DataLoader(dataset, **loader_opts)

    @staticmethod
    def _parse_datasets(config):
        """
        Parse and instantiate datasets from the configuration.

        Args:
            config (DictConfig): Configuration for the datasets.

        Returns:
            list: A list of instantiated datasets.
        """
        datasets = []
        if isinstance(config.dataset_opts, DictConfig):
            for key in config.dataset_opts:
                dataset_opt = config.dataset_opts[key]
                dataset = instantiate(dataset_opt)
                datasets.append(dataset)
        elif isinstance(config.dataset_opts, ListConfig):
            for dataset_opt in config.dataset_opts:
                dataset = instantiate(dataset_opt)
                datasets.append(dataset)
        return datasets

    @staticmethod
    def _parse_loaders(config, loader_opts):
        """
        Parse and create data loaders from the configuration.

        Args:
            config (DictConfig): Configuration for the datasets.
            loader_opts (DictConfig): Options for the data loaders.

        Returns:
            DataLoader or list: A single DataLoader or a list of DataLoaders.
        """
        if isinstance(config.dataset_opts, DictConfig):
            dataloaders = {}
            for key in config.dataset_opts:
                dataset_opt = config.dataset_opts[key]
                dataset = instantiate(dataset_opt)
                if "loader_opts" in config:
                    loader_opt = deepcopy(loader_opts)
                    loader_opt.update(config.loader_opts)
                dataloaders[key] = DataLoader(dataset, **loader_opts)
            return dataloaders
        elif isinstance(config.dataset_opts, ListConfig):
            dataloaders = []
            for idx, dataset_opt in enumerate(config.dataset_opts):
                if isinstance(dataset_opt, ListConfig):
                    datasets = [instantiate(opt) for opt in dataset_opt]
                    dataset = ConcatDataset(datasets)
                else:
                    dataset = instantiate(dataset_opt)
                if "loader_opts" in config:
                    loader_opt = deepcopy(loader_opts)
                    if isinstance(config.loader_opts, ListConfig):
                        loader_opt.update(config.loader_opts[idx])
                    else:
                        loader_opt.update(config.loader_opts)
                dataloaders.append(DataLoader(dataset, **loader_opts))
            return dataloaders
