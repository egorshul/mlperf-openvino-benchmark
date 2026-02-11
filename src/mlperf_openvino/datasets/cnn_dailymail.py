"""CNN-DailyMail dataset for MLPerf LLM benchmarks (Llama 3.1 8B).

Follows the MLCommons Inference v5.1 reference implementation:
  https://github.com/mlcommons/inference/tree/master/language/llama3.1-8b

Dataset: CNN-DailyMail v3.0.0 — text summarization task.
  - Datacenter: 13,368 samples (cnn_eval.json)
  - Edge: 5,000 samples (sample_cnn_eval_5000.json)

Each sample contains an input prompt (article + summarization instruction)
and a reference output (highlights) for ROUGE evaluation.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseDataset, QuerySampleLibrary

logger = logging.getLogger(__name__)

# Llama 3.1 Instruct chat template for summarization
_SUMMARIZATION_PROMPT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "Briefly summarize this news article:\n\n{article}"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)


class CnnDailyMailDataset(BaseDataset):
    """CNN-DailyMail dataset for text summarization benchmarks.

    Loads the MLCommons-processed JSON file containing pre-formatted
    prompts and reference summaries for ROUGE-based accuracy evaluation.
    """

    def __init__(
        self,
        data_path: str,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        count: Optional[int] = None,
        max_seq_length: int = 2048,
        **kwargs,
    ):
        super().__init__(data_path, count, **kwargs)
        self.model_name = model_name
        self.max_seq_length = max_seq_length

        self._input_texts: List[str] = []
        self._reference_outputs: List[str] = []
        self._input_ids_cache: Dict[int, np.ndarray] = {}
        self._attention_mask_cache: Dict[int, np.ndarray] = {}

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

        # MLCommons reference filenames (in order of priority)
        json_candidates = [
            data_path / "cnn_eval.json",
            data_path / "sample_cnn_eval_5000.json",
        ]
        pkl_candidates = [
            data_path / "cnn_eval.pkl",
        ]

        loaded = False
        for candidate in json_candidates:
            if candidate.exists():
                self._load_from_json(candidate)
                loaded = True
                break

        if not loaded:
            for candidate in pkl_candidates:
                if candidate.exists():
                    self._load_from_pickle(candidate)
                    loaded = True
                    break

        if not loaded:
            raise FileNotFoundError(
                f"No CNN-DailyMail dataset found in {data_path}. "
                f"Expected: cnn_eval.json or sample_cnn_eval_5000.json. "
                f"Run: mlperf-ov download-dataset --dataset cnn-dailymail"
            )

        if self.count is not None and self.count < len(self._input_texts):
            self._input_texts = self._input_texts[: self.count]
            self._reference_outputs = self._reference_outputs[: self.count]

        self._items = list(range(len(self._input_texts)))
        self._labels = self._reference_outputs
        self._loaded = True

        logger.info(
            f"CNN-DailyMail dataset loaded: {len(self._input_texts)} samples "
            f"(max_seq_length={self.max_seq_length})"
        )

    def _load_from_json(self, json_path: Path) -> None:
        """Load MLCommons-format JSON (list of dicts with 'input' and 'output')."""
        logger.info(f"Loading dataset from {json_path}...")

        with open(json_path, "r", encoding="utf-8") as f:
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

    def _load_from_pickle(self, pkl_path: Path) -> None:
        """Load from pickle format (fallback)."""
        logger.info(f"Loading dataset from {pkl_path}...")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, list):
            for entry in data:
                if isinstance(entry, dict):
                    self._input_texts.append(str(entry.get("input", "")))
                    self._reference_outputs.append(str(entry.get("output", "")))
        elif hasattr(data, "to_dict"):
            # pandas DataFrame
            records = data.to_dict("records")
            for entry in records:
                self._input_texts.append(str(entry.get("input", "")))
                self._reference_outputs.append(str(entry.get("output", "")))

        logger.info(f"Loaded {len(self._input_texts)} samples from pickle")

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

    def clear_cache(self) -> None:
        """Free the tokenization cache memory."""
        self._input_ids_cache.clear()
        self._attention_mask_cache.clear()

    def get_input_text(self, index: int) -> str:
        return self._input_texts[index]

    def get_reference_output(self, index: int) -> str:
        return self._reference_outputs[index]

    def get_sample(self, index: int) -> Tuple[np.ndarray, Any]:
        features = self.tokenize_sample(index)
        return features["input_ids"], self._reference_outputs[index]

    def get_samples(self, indices: List[int]) -> Tuple[List[np.ndarray], List[Any]]:
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
        """Compute ROUGE scores per MLCommons specification.

        Uses rouge_scorer with use_stemmer=True matching the MLPerf reference
        evaluate-accuracy.py for Llama 3.1 8B.
        """
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            raise ImportError(
                "rouge-score is required for LLM accuracy. "
                "Install with: pip install rouge-score"
            )

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        gen_lengths = []

        tokenizer = self._get_tokenizer()

        for pred, ref in zip(predictions, labels):
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores["rouge1"].fmeasure * 100)
            rouge2_scores.append(scores["rouge2"].fmeasure * 100)
            rougeL_scores.append(scores["rougeL"].fmeasure * 100)

            gen_len = len(tokenizer.encode(pred))
            gen_lengths.append(gen_len)

        num_samples = len(predictions)
        avg_rouge1 = sum(rouge1_scores) / num_samples if num_samples > 0 else 0.0
        avg_rouge2 = sum(rouge2_scores) / num_samples if num_samples > 0 else 0.0
        avg_rougeL = sum(rougeL_scores) / num_samples if num_samples > 0 else 0.0
        total_gen_len = sum(gen_lengths)
        avg_tokens = total_gen_len / num_samples if num_samples > 0 else 0.0

        return {
            "rouge1": avg_rouge1,
            "rouge2": avg_rouge2,
            "rougeL": avg_rougeL,
            "tokens_per_sample": avg_tokens,
            "gen_len": total_gen_len,
            "num_samples": num_samples,
        }


class CnnDailyMailQSL(QuerySampleLibrary):
    """MLPerf Query Sample Library for CNN-DailyMail dataset.

    Follows MLPerf Inference v5.1 spec:
      - performance_sample_count = 13368 (datacenter) or 5000 (edge)
    """

    def __init__(
        self,
        data_path: str,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        count: Optional[int] = None,
        performance_sample_count: int = 13368,
        max_seq_length: int = 2048,
    ):
        self.dataset = CnnDailyMailDataset(
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
        self.dataset.clear_cache()

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
