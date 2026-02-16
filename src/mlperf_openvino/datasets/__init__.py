"""Datasets for MLPerf OpenVINO Benchmark."""

from .base import BaseDataset, QuerySampleLibrary
from .imagenet import ImageNetDataset, ImageNetQSL
from .librispeech import LibriSpeechDataset, LibriSpeechQSL
from .coco_prompts import COCOPromptsDataset, COCOPromptsQSL
from .cnn_dailymail import CnnDailyMailDataset, CnnDailyMailQSL

__all__ = [
    "BaseDataset",
    "QuerySampleLibrary",
    "ImageNetDataset",
    "ImageNetQSL",
    "LibriSpeechDataset",
    "LibriSpeechQSL",
    "COCOPromptsDataset",
    "COCOPromptsQSL",
    "CnnDailyMailDataset",
    "CnnDailyMailQSL",
]
