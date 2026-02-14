"""OpenORCA dataset for MLPerf LLM benchmarks (Llama 3.1 8B / Llama 2 70B).

Follows the MLCommons Inference reference implementation:
  https://github.com/mlcommons/inference/tree/master/language/llama2-70b

Dataset: Open-Orca/OpenOrca (processed subset with 24576 samples).
Each sample contains an input prompt and a reference output for ROUGE evaluation.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseDataset, QuerySampleLibrary

logger = logging.getLogger(__name__)


class OpenOrcaDataset(BaseDataset):
    """OpenORCA dataset for text generation benchmarks.

    Loads the MLCommons-processed pickle file containing pre-tokenized
    prompts and reference outputs for ROUGE-based accuracy evaluation.
    """

    def __init__(
        self,
        data_path: str,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        count: Optional[int] = None,
        max_seq_length: int = 1024,
        **kwargs,
    ):
        super().__init__(data_path, count, **kwargs)
        self.model_name = model_name
        self.max_seq_length = max_seq_length

        self._input_texts: List[str] = []
        self._reference_outputs: List[str] = []
        self._input_ids_cache: Dict[int, np.ndarray] = {}
        self._attention_mask_cache: Dict[int, np.ndarray] = {}
        self._input_lens: List[int] = []

        self._tokenizer = None

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left",
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def load(self) -> None:
        data_path = Path(self.data_path)

        pkl_candidates = [
            data_path / "open_orca_gpt4_tokenized_llama.sampled_24576.pkl",
            data_path / "processed_data.pkl",
        ]
        pkl_file = None
        for candidate in pkl_candidates:
            if candidate.exists():
                pkl_file = candidate
                break

        if pkl_file is not None:
            self._load_from_pickle(pkl_file)
        else:
            dataset_file = data_path / "dataset.json"
            if dataset_file.exists():
                self._load_from_json(dataset_file)
            else:
                raise FileNotFoundError(
                    f"No dataset found in {data_path}. Expected one of: "
                    f"{[c.name for c in pkl_candidates]} or dataset.json. "
                    f"Run: mlperf-ov download-dataset --dataset open-orca"
                )

        if self.count is not None and self.count < len(self._input_texts):
            self._input_texts = self._input_texts[: self.count]
            self._reference_outputs = self._reference_outputs[: self.count]

        self._items = list(range(len(self._input_texts)))
        self._labels = self._reference_outputs
        self._loaded = True

        logger.info(
            f"OpenORCA dataset loaded: {len(self._input_texts)} samples "
            f"(max_seq_length={self.max_seq_length})"
        )

    def _load_from_pickle(self, pkl_path: Path) -> None:
        """Load MLCommons-format pickle (list of dicts with 'input' and 'output')."""
        logger.info(f"Loading dataset from {pkl_path}...")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    self._input_texts.append(str(entry.get("input", "")))
                    self._reference_outputs.append(str(entry.get("output", "")))
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    self._input_texts.append(str(entry[0]))
                    self._reference_outputs.append(str(entry[1]))
        elif isinstance(data, dict):
            inputs = data.get("input", data.get("inputs", []))
            outputs = data.get("output", data.get("outputs", []))
            self._input_texts = [str(x) for x in inputs]
            self._reference_outputs = [str(x) for x in outputs]

        logger.info(f"Loaded {len(self._input_texts)} samples from pickle")

    def _load_from_json(self, json_path: Path) -> None:
        """Load from JSON format."""
        import json

        logger.info(f"Loading dataset from {json_path}...")

        with open(json_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            for entry in data:
                self._input_texts.append(str(entry.get("input", "")))
                self._reference_outputs.append(str(entry.get("output", "")))
        elif isinstance(data, dict):
            inputs = data.get("input", data.get("inputs", []))
            outputs = data.get("output", data.get("outputs", []))
            self._input_texts = [str(x) for x in inputs]
            self._reference_outputs = [str(x) for x in outputs]

        logger.info(f"Loaded {len(self._input_texts)} samples from JSON")

    def tokenize_sample(self, index: int) -> Dict[str, np.ndarray]:
        """Tokenize a single sample, with caching."""
        if index in self._input_ids_cache:
            return {
                "input_ids": self._input_ids_cache[index],
                "attention_mask": self._attention_mask_cache[index],
            }

        tokenizer = self._get_tokenizer()
        text = self._input_texts[index]

        encoded = tokenizer(
            text,
            return_tensors="np",
            padding=False,
            truncation=True,
            max_length=self.max_seq_length,
        )

        input_ids = encoded["input_ids"].astype(np.int64)
        attention_mask = encoded["attention_mask"].astype(np.int64)

        self._input_ids_cache[index] = input_ids
        self._attention_mask_cache[index] = attention_mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def get_input_text(self, index: int) -> str:
        return self._input_texts[index]

    def get_reference_output(self, index: int) -> str:
        return self._reference_outputs[index]

    def get_sample(self, index: int) -> Tuple[np.ndarray, Any]:
        features = self.tokenize_sample(index)
        return features["input_ids"], self._reference_outputs[index]

    def get_samples(self, indices: List[int]) -> Tuple[np.ndarray, List[Any]]:
        inputs = []
        labels = []
        for idx in indices:
            inp, label = self.get_sample(idx)
            inputs.append(inp)
            labels.append(label)
        return inputs, labels

    def postprocess(self, results: np.ndarray, indices: List[int]) -> List[Any]:
        """Decode token IDs to text."""
        tokenizer = self._get_tokenizer()
        if isinstance(results, np.ndarray) and results.ndim >= 2:
            return tokenizer.batch_decode(results, skip_special_tokens=True)
        return [str(results)]

    def compute_accuracy(
        self, predictions: List[str], labels: List[str]
    ) -> Dict[str, float]:
        """Compute ROUGE scores per MLCommons specification."""
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            raise ImportError(
                "rouge-score is required for LLM accuracy. "
                "Install with: pip install rouge-score"
            )

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=False
        )

        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        gen_lengths = []

        for pred, ref in zip(predictions, labels):
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores["rouge1"].fmeasure * 100)
            rouge2_scores.append(scores["rouge2"].fmeasure * 100)
            rougeL_scores.append(scores["rougeL"].fmeasure * 100)

            tokenizer = self._get_tokenizer()
            gen_len = len(tokenizer.encode(pred))
            gen_lengths.append(gen_len)

        num_samples = len(predictions)
        avg_rouge1 = sum(rouge1_scores) / num_samples if num_samples > 0 else 0.0
        avg_rouge2 = sum(rouge2_scores) / num_samples if num_samples > 0 else 0.0
        avg_rougeL = sum(rougeL_scores) / num_samples if num_samples > 0 else 0.0
        avg_tokens = sum(gen_lengths) / num_samples if num_samples > 0 else 0.0

        return {
            "rouge1": avg_rouge1,
            "rouge2": avg_rouge2,
            "rougeL": avg_rougeL,
            "tokens_per_sample": avg_tokens,
            "num_samples": num_samples,
        }


class OpenOrcaQSL(QuerySampleLibrary):
    """MLPerf Query Sample Library for OpenORCA dataset."""

    def __init__(
        self,
        data_path: str,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        count: Optional[int] = None,
        performance_sample_count: int = 24576,
        max_seq_length: int = 1024,
    ):
        self.dataset = OpenOrcaDataset(
            data_path=data_path,
            model_name=model_name,
            count=count,
            max_seq_length=max_seq_length,
        )
        self._performance_sample_count = performance_sample_count
        self._loaded_samples: Dict[int, Dict[str, np.ndarray]] = {}

    def load(self) -> None:
        self.dataset.load()

    def load_query_samples(self, sample_list: List[int]) -> None:
        for sample_id in sample_list:
            self._loaded_samples[sample_id] = self.dataset.tokenize_sample(sample_id)

    def unload_query_samples(self, sample_list: List[int]) -> None:
        for sample_id in sample_list:
            self._loaded_samples.pop(sample_id, None)

    def get_features(self, sample_id: int) -> Dict[str, np.ndarray]:
        if sample_id in self._loaded_samples:
            return self._loaded_samples[sample_id]
        return self.dataset.tokenize_sample(sample_id)

    def get_input_text(self, sample_id: int) -> str:
        return self.dataset.get_input_text(sample_id)

    def get_label(self, sample_id: int) -> str:
        return self.dataset.get_reference_output(sample_id)

    @property
    def total_sample_count(self) -> int:
        return self.dataset.sample_count

    @property
    def performance_sample_count(self) -> int:
        return min(self._performance_sample_count, self.total_sample_count)
