from datasets import load_dataset, Dataset
import mmengine
import os
from typing import Union
from .record_log import Logger


def load_fcall_datasets(task, dataset):
    dataset_name = os.path.basename(dataset)
    Logger.info_logger(f"task {task},loading dataset {dataset}")
    dataset = mmengine.load(dataset)
    total_num = len(dataset)
    Logger.info_logger(f"dataset: {dataset_name} total samples: {total_num}")
    return dataset, total_num


def load_dsl_datasets(task, dataset):
    return load_fcall_datasets(task, dataset)


def load_summary_datasets(task, dataset) -> dict:
    Logger.info_logger(f"task {task},loading dataset {dataset}")
    if os.path.isfile(dataset):
        try:
            dataset = load_dataset("json", data_files=dataset, split="train")
        except:
            raise ValueError("olny support json file")
        dataset_name = os.path.basename(dataset)
        Logger.info_logger(f"dataset: {dataset_name} total samples: {dataset.num_rows}")
        return {dataset_name: dataset}
    else:
        dataset_path = os.path.join("data", dataset)
        dataset_list = []
        dataset_name = []
        for name in os.listdir(dataset_path):
            dataset_name.append(os.path.basename(name))
            if name.split(".")[-1] == "json":
                dataset = load_dataset("json", data_files=os.path.join(dataset_path, name), split="train")
                dataset_list.append(dataset)

        sample_num = 0
        for dataset in dataset_list:
            sample_num += dataset.num_rows
        out_dataset_name = " ".join(dataset_name)
        Logger.info_logger(f"dataset: {out_dataset_name}, total samples: {sample_num}")
        dataset_result = {name: dataset for name, dataset in zip(dataset_name, dataset_list)}
        return dataset_result