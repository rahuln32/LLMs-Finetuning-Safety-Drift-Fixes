# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/alpaca_dataset/alpaca_data_no_safety.json"

@dataclass
class saferpaca_dataset:
    dataset: str = "saferpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/saferpaca_dataset/saferpaca_Instruction_500.json"

@dataclass
class safe_only_dataset:
    dataset: str = "safe_only_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/safe_only_dataset/safety_only_data_Instructions.json"