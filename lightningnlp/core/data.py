import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import pytorch_lightning as pl
from datasets import Dataset, DatasetDict, Version, load_dataset
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers import PreTrainedTokenizerBase

from lightningnlp.core.iterable import IterableDataLoader


class TransformerDataModule(pl.LightningDataModule):
    """Base ``LightningDataModule`` for HuggingFace Datasets. Provides helper functions and boilerplate logic to
    load/process datasets.
    Args:
        tokenizer: ``PreTrainedTokenizerBase`` for tokenizing data.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        train_batch_size: int = 32,
        validation_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 0,
        dataset_name: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        revision: Optional[Union[str, Version]] = None,
        train_val_split: Optional[int] = None,
        train_file: Optional[str] = None,
        test_file: Optional[str] = None,
        validation_file: Optional[str] = None,
        max_length: int = 128,
        preprocessing_num_workers: int = 1,
        load_from_cache_file: bool = True,
        cache_dir: Optional[Union[Path, str]] = None,
        limit_train_samples: Optional[int] = None,
        limit_val_samples: Optional[int] = None,
        limit_test_samples: Optional[int] = None,
        streaming: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.train_batch_size = train_batch_size
        self.validation_batch_size = validation_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.revision = revision
        self.train_val_split = train_val_split
        self.train_file = train_file
        self.test_file = test_file
        self.validation_file = validation_file

        self.max_length = max_length
        self.preprocessing_num_workers = preprocessing_num_workers
        self.load_from_cache_file = load_from_cache_file
        self.cache_dir = cache_dir

        self.limit_train_samples = limit_train_samples
        self.limit_val_samples = limit_val_samples
        self.limit_test_samples = limit_test_samples

        self.streaming = streaming
        os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"  # TODO: smarter handling of this env variable

    def setup(self, stage: Optional[str] = None):
        dataset = self.load_dataset()
        dataset = self.split_dataset(dataset)
        dataset = self.process_data(dataset, stage=stage)
        self.ds = dataset

    def process_data(
        self, dataset: Union[Dataset, DatasetDict], stage: Optional[str] = None
    ) -> Union[Dataset, DatasetDict]:
        return dataset

    def load_dataset(self) -> Dataset:
        # Allow custom data files when loading the datasets
        data_files = {}
        if self.train_file is not None:
            data_files["train"] = self.train_file
        if self.validation_file is not None:
            data_files["validation"] = self.validation_file
        if self.test_file is not None:
            data_files["test"] = self.test_file

        data_files = data_files or None
        if self.dataset_name is not None:
            # Download and load the Huggingface datasets.
            dataset = load_dataset(
                path=self.dataset_name,
                name=self.dataset_config_name,
                cache_dir=self.cache_dir,
                data_files=data_files,
                revision=self.revision,
                streaming=self.streaming,
            )

        # Load straight from data files
        elif data_files:
            extension = self.train_file.split(".")[-1]
            dataset = load_dataset(extension, data_files=data_files)

        else:
            raise MisconfigurationException(
                "You have not specified a datasets name nor a custom train and validation file"
            )

        return dataset

    def split_dataset(self, dataset: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
        if self.train_val_split is not None:
            split = dataset["train"].train_test_split(self.train_val_split)
            dataset["train"] = split["train"]
            dataset["validation"] = split["test"]
        dataset = self._select_samples(dataset)
        return dataset

    def _select_samples(self, dataset: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
        samples = (
            ("train", self.limit_train_samples),
            ("validation", self.limit_val_samples),
            ("test", self.limit_test_samples),
        )
        for column_name, n_samples in samples:
            if n_samples is not None and column_name in dataset:
                indices = range(min(len(dataset[column_name]), n_samples))
                dataset[column_name] = dataset[column_name].select(indices)
        return dataset

    def state_dict(self) -> Dict[str, Any]:
        return {"tokenizer": self.tokenizer}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.tokenizer = state_dict["tokenizer"]

    def train_dataloader(self) -> DataLoader:
        if self.streaming:
            return IterableDataLoader(
                self.ds["train"],
                batch_size=self.train_batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )

        return DataLoader(
            self.ds["train"],
            batch_size=self.train_batch_size,
            sampler=RandomSampler(self.ds["train"]),
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        if self.streaming:
            return IterableDataLoader(
                self.ds["validation"],
                batch_size=self.validation_batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )

        return DataLoader(
            self.ds["validation"],
            batch_size=self.validation_batch_size,
            sampler=SequentialSampler(self.ds["validation"]),
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if "test" in self.ds:
            if self.streaming:
                return IterableDataLoader(
                    self.ds["test"],
                    batch_size=self.test_batch_size,
                    num_workers=self.num_workers,
                    collate_fn=self.collate_fn,
                )

            return DataLoader(
                self.ds["test"],
                batch_size=self.test_batch_size,
                sampler=SequentialSampler(self.ds["test"]),
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )

    @property
    def collate_fn(self) -> Optional[Callable]:
        return None
