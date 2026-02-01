"""SQuAD dataset for BERT Question Answering."""

import json
import logging
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import BaseDataset, QuerySampleLibrary

logger = logging.getLogger(__name__)

# BERT tokenization constants
MAX_SEQ_LENGTH = 384
MAX_QUERY_LENGTH = 64
DOC_STRIDE = 128
MAX_ANSWER_LENGTH = 30
N_BEST_SIZE = 20


def _get_best_indexes(logits: np.ndarray, n_best: int = N_BEST_SIZE) -> List[int]:
    """Get the n-best logit indices sorted by score."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in index_and_score[:n_best]]


def normalize_answer(s: str) -> str:
    """Normalize answer: lowercase, remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


class BertTokenizer:
    """
    Simple BERT tokenizer for SQuAD.

    For production use, consider using transformers library.
    This is a simplified version for basic functionality.
    """

    def __init__(self, vocab_file: Optional[str] = None):
        """Initialize tokenizer."""
        self._vocab = {}
        self._inv_vocab = {}
        self._do_lower_case = True

        if vocab_file:
            self._load_vocab(vocab_file)
        else:
            self._load_from_transformers()

    def _load_vocab(self, vocab_file: str) -> None:
        """Load vocabulary from file."""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                token = line.strip()
                self._vocab[token] = idx
                self._inv_vocab[idx] = token
        logger.info(f"Loaded vocab with {len(self._vocab)} tokens")

    def _load_from_transformers(self) -> None:
        """Load tokenizer from transformers library."""
        try:
            from transformers import BertTokenizer as HFBertTokenizer
            self._hf_tokenizer = HFBertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
            self._vocab = self._hf_tokenizer.vocab
            self._inv_vocab = {v: k for k, v in self._vocab.items()}
            logger.info("Loaded tokenizer from transformers")
        except ImportError:
            logger.warning(
                "transformers not installed, using basic tokenization. "
                "Install with: pip install transformers"
            )
            self._hf_tokenizer = None

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into word pieces."""
        if hasattr(self, '_hf_tokenizer') and self._hf_tokenizer is not None:
            return self._hf_tokenizer.tokenize(text)

        if self._do_lower_case:
            text = text.lower()

        # Simple whitespace tokenization (fallback without HF)
        tokens = text.split()
        return tokens

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to vocabulary IDs."""
        if hasattr(self, '_hf_tokenizer') and self._hf_tokenizer is not None:
            return self._hf_tokenizer.convert_tokens_to_ids(tokens)

        ids = []
        unk_id = self._vocab.get('[UNK]', 100)
        for token in tokens:
            ids.append(self._vocab.get(token, unk_id))
        return ids

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert vocabulary IDs to tokens."""
        if hasattr(self, '_hf_tokenizer') and self._hf_tokenizer is not None:
            return self._hf_tokenizer.convert_ids_to_tokens(ids)

        tokens = []
        for idx in ids:
            tokens.append(self._inv_vocab.get(idx, '[UNK]'))
        return tokens

    def encode_with_mapping(
        self,
        question: str,
        context: str,
        max_length: int = MAX_SEQ_LENGTH,
        padding: bool = True,
    ) -> Dict[str, Any]:
        """Encode question and context for BERT with token-to-original mapping."""
        doc_tokens = context.split()

        char_to_word_offset = []
        for i, word in enumerate(doc_tokens):
            start = context.find(word, len(''.join(doc_tokens[:i])) + i)
            for _ in range(len(word)):
                char_to_word_offset.append(i)
            char_to_word_offset.append(i)  # for space after word

        question_tokens = ['[CLS]'] + self.tokenize(question) + ['[SEP]']

        tok_to_orig_index = []
        all_doc_tokens = []

        for i, word in enumerate(doc_tokens):
            sub_tokens = self.tokenize(word)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        max_context_len = max_length - len(question_tokens) - 1  # -1 for [SEP]
        if len(all_doc_tokens) > max_context_len:
            all_doc_tokens = all_doc_tokens[:max_context_len]
            tok_to_orig_index = tok_to_orig_index[:max_context_len]

        context_tokens = all_doc_tokens + ['[SEP]']
        tokens = question_tokens + context_tokens

        # -1 for question tokens (no mapping to context)
        full_tok_to_orig = [-1] * len(question_tokens) + tok_to_orig_index + [-1]  # -1 for [SEP]

        input_ids = self.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(question_tokens) + [1] * len(context_tokens)

        if padding:
            pad_len = max_length - len(input_ids)
            input_ids = input_ids + [0] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            token_type_ids = token_type_ids + [0] * pad_len
            full_tok_to_orig = full_tok_to_orig + [-1] * pad_len

        return {
            'input_ids': np.array(input_ids, dtype=np.int64),
            'attention_mask': np.array(attention_mask, dtype=np.int64),
            'token_type_ids': np.array(token_type_ids, dtype=np.int64),
            'tok_to_orig_index': full_tok_to_orig,
            'doc_tokens': doc_tokens,
        }

    def encode(
        self,
        question: str,
        context: str,
        max_length: int = MAX_SEQ_LENGTH,
        padding: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Encode question and context for BERT."""
        result = self.encode_with_mapping(question, context, max_length, padding)
        return {
            'input_ids': result['input_ids'],
            'attention_mask': result['attention_mask'],
            'token_type_ids': result['token_type_ids'],
        }

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        if hasattr(self, '_hf_tokenizer') and self._hf_tokenizer is not None:
            return self._hf_tokenizer.decode(token_ids, skip_special_tokens=True)

        tokens = self.convert_ids_to_tokens(token_ids)
        tokens = [t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]']]
        text = ' '.join(tokens).replace(' ##', '')
        return text


class SQuADDataset(BaseDataset):
    """
    SQuAD v1.1 dataset for BERT Question Answering benchmark.

    SQuAD (Stanford Question Answering Dataset) contains questions
    posed by crowdworkers on a set of Wikipedia articles.

    Expected file: dev-v1.1.json
    """

    def __init__(
        self,
        data_path: str,
        vocab_file: Optional[str] = None,
        count: Optional[int] = None,
        max_seq_length: int = MAX_SEQ_LENGTH,
        max_query_length: int = MAX_QUERY_LENGTH,
        doc_stride: int = DOC_STRIDE,
    ):
        """Initialize SQuAD dataset."""
        super().__init__(data_path=data_path, count=count)

        self.data_path = Path(data_path)
        self.vocab_file = vocab_file
        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length
        self.doc_stride = doc_stride

        self._samples: List[Dict[str, Any]] = []
        self._features: List[Dict[str, Any]] = []
        self._tokenizer: Optional[BertTokenizer] = None
        self._cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._is_loaded = False

    def load(self) -> None:
        """Load and preprocess the dataset."""
        if self._is_loaded:
            return

        logger.info(f"Loading SQuAD dataset from {self.data_path}")

        self._tokenizer = BertTokenizer(self.vocab_file)

        if self.data_path.is_file():
            data_file = self.data_path
        else:
            for name in ["dev-v1.1.json", "train-v1.1.json", "squad.json"]:
                candidate = self.data_path / name
                if candidate.exists():
                    data_file = candidate
                    break
            else:
                raise FileNotFoundError(f"No SQuAD data file found in {self.data_path}")

        with open(data_file, 'r', encoding='utf-8') as f:
            squad_data = json.load(f)

        self._parse_squad_data(squad_data)

        if self.count and self.count < len(self._samples):
            self._samples = self._samples[:self.count]

        self._create_features()

        logger.info(f"Loaded {len(self._samples)} examples, {len(self._features)} features")
        self._is_loaded = True

    def _parse_squad_data(self, data: Dict) -> None:
        """Parse SQuAD JSON data into examples."""
        for article in data['data']:
            title = article.get('title', '')

            for paragraph in article['paragraphs']:
                context = paragraph['context']

                for qa in paragraph['qas']:
                    qas_id = qa['id']
                    question = qa['question']

                    answers = []
                    if 'answers' in qa:
                        for answer in qa['answers']:
                            answers.append({
                                'text': answer['text'],
                                'answer_start': answer['answer_start'],
                            })

                    self._samples.append({
                        'id': qas_id,
                        'question': question,
                        'context': context,
                        'answers': answers,
                        'title': title,
                    })

    def _create_features(self) -> None:
        """Create tokenized features from examples with token-to-original mapping."""
        self._features = []

        for idx, sample in enumerate(self._samples):
            encoding = self._tokenizer.encode_with_mapping(
                sample['question'],
                sample['context'],
                max_length=self.max_seq_length,
                padding=True,
            )

            feature = {
                'example_index': idx,
                'qas_id': sample['id'],
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'token_type_ids': encoding['token_type_ids'],
                'tok_to_orig_index': encoding['tok_to_orig_index'],
                'doc_tokens': encoding['doc_tokens'],
            }

            self._features.append(feature)
            self._items.append(sample['id'])

    def __len__(self) -> int:
        return len(self._features)

    @property
    def total_count(self) -> int:
        return len(self._features)

    @property
    def sample_count(self) -> int:
        return len(self._features)

    def get_sample(self, index: int) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Get preprocessed sample."""
        if not self._is_loaded:
            self.load()

        if index in self._cache:
            features = self._cache[index]
        else:
            feature = self._features[index]
            features = {
                'input_ids': feature['input_ids'].reshape(1, -1),
                'attention_mask': feature['attention_mask'].reshape(1, -1),
                'token_type_ids': feature['token_type_ids'].reshape(1, -1),
            }
            self._cache[index] = features

        example_idx = self._features[index]['example_index']
        sample_info = self._samples[example_idx]

        return features, sample_info

    def get_samples(self, indices: List[int]) -> Tuple[Dict[str, np.ndarray], List[Dict]]:
        """Get batch of samples."""
        if not self._is_loaded:
            self.load()

        batch_features = {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
        }
        sample_infos = []

        for idx in indices:
            features, info = self.get_sample(idx)
            for key in batch_features:
                batch_features[key].append(features[key][0])
            sample_infos.append(info)

        return {
            key: np.stack(val) for key, val in batch_features.items()
        }, sample_infos

    def postprocess(
        self,
        results: Union[np.ndarray, Tuple[np.ndarray, np.ndarray], List[Tuple[np.ndarray, np.ndarray]]],
        indices: List[int]
    ) -> List[str]:
        """Postprocess BERT outputs to extract answer spans."""
        if isinstance(results, list):
            # List of tuples from C++ SUT: [(start, end), (start, end), ...]
            # Keep as list - logits may have different lengths (bucketed inference)
            start_logits = [np.asarray(r[0]) for r in results]
            end_logits = [np.asarray(r[1]) for r in results]
        elif isinstance(results, tuple):
            start_logits, end_logits = results
        else:
            # Numpy array - assume shape [batch, 2, seq_len] or [batch, seq_len, 2]
            if results.shape[-1] == 2:
                start_logits = results[..., 0]
                end_logits = results[..., 1]
            else:
                start_logits = results[:, 0, :]
                end_logits = results[:, 1, :]

        predictions = []

        for i, idx in enumerate(indices):
            feature = self._features[idx]
            token_type_ids = feature['token_type_ids']
            tok_to_orig_index = feature.get('tok_to_orig_index', None)
            doc_tokens = feature.get('doc_tokens', None)

            s_logits = start_logits[i]
            e_logits = end_logits[i]

            # Find context boundaries
            # token_type_id: 0 = [CLS] question [SEP], 1 = context [SEP]
            context_start = 0
            context_end = len(token_type_ids) - 1
            for j, tid in enumerate(token_type_ids):
                if tid == 1:
                    context_start = j
                    break
            for j in range(len(token_type_ids) - 1, -1, -1):
                if token_type_ids[j] == 1:
                    context_end = j
                    break

            start_indexes = _get_best_indexes(s_logits, N_BEST_SIZE)
            end_indexes = _get_best_indexes(e_logits, N_BEST_SIZE)

            best_score = float('-inf')
            best_start = context_start
            best_end = context_start

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index > end_index:
                        continue
                    if end_index - start_index + 1 > MAX_ANSWER_LENGTH:
                        continue
                    # Must be within context (not question or special tokens)
                    if start_index < context_start or end_index > context_end:
                        continue
                    # Must have valid mapping to original tokens
                    if tok_to_orig_index is not None:
                        if tok_to_orig_index[start_index] == -1 or tok_to_orig_index[end_index] == -1:
                            continue

                    score = s_logits[start_index] + e_logits[end_index]
                    if score > best_score:
                        best_score = score
                        best_start = start_index
                        best_end = end_index

            if tok_to_orig_index is not None and doc_tokens is not None:
                orig_start = tok_to_orig_index[best_start]
                orig_end = tok_to_orig_index[best_end]

                if orig_start != -1 and orig_end != -1:
                    answer_text = ' '.join(doc_tokens[orig_start:orig_end + 1])
                else:
                    answer_text = ""
            else:
                # Fallback: decode from token IDs
                input_ids = feature['input_ids']
                answer_ids = input_ids[best_start:best_end + 1].tolist()
                answer_text = self._tokenizer.decode(answer_ids)
                answer_text = answer_text.replace(" ##", "").replace("##", "")

            answer_text = answer_text.strip()

            predictions.append(answer_text)

        return predictions

    def compute_accuracy(
        self,
        predictions: List[str],
        indices: List[int]
    ) -> Dict[str, float]:
        """Compute F1 and Exact Match scores."""
        f1_scores = []
        em_scores = []

        for pred, idx in zip(predictions, indices):
            feature = self._features[idx]
            example_idx = feature['example_index']
            sample = self._samples[example_idx]

            ground_truths = [ans['text'] for ans in sample['answers']]

            if not ground_truths:
                continue

            max_f1 = max(compute_f1(pred, gt) for gt in ground_truths)
            max_em = max(compute_exact_match(pred, gt) for gt in ground_truths)

            f1_scores.append(max_f1)
            em_scores.append(max_em)

        return {
            'f1': np.mean(f1_scores) * 100 if f1_scores else 0.0,
            'exact_match': np.mean(em_scores) * 100 if em_scores else 0.0,
            'num_samples': len(f1_scores),
        }

    def get_ground_truth(self, index: int) -> List[str]:
        """Get ground truth answers for a sample."""
        feature = self._features[index]
        example_idx = feature['example_index']
        sample = self._samples[example_idx]
        return [ans['text'] for ans in sample['answers']]


class SQuADQSL(QuerySampleLibrary):
    """
    Query Sample Library for SQuAD dataset.

    Implements the MLPerf LoadGen QSL interface for BERT benchmark.
    """

    # Sequence length buckets for optimized inference
    SEQ_BUCKETS = [128, 165, 256, 384]

    def __init__(
        self,
        data_path: str,
        vocab_file: Optional[str] = None,
        count: Optional[int] = None,
        performance_sample_count: int = 10833,  # MLPerf default
        max_seq_length: int = MAX_SEQ_LENGTH,
    ):
        """Initialize SQuAD QSL."""
        super().__init__()

        self.dataset = SQuADDataset(
            data_path=data_path,
            vocab_file=vocab_file,
            count=count,
            max_seq_length=max_seq_length,
        )

        self._performance_sample_count = performance_sample_count
        self._loaded_samples: Dict[int, Dict[str, np.ndarray]] = {}
        self._sample_seq_lengths: Dict[int, int] = {}  # sample_idx -> actual_seq_len
        self._sample_buckets: Dict[int, int] = {}  # sample_idx -> bucket_idx

    @staticmethod
    def get_bucket_index(seq_len: int) -> int:
        """Get bucket index for a sequence length."""
        for i, bucket in enumerate(SQuADQSL.SEQ_BUCKETS):
            if seq_len <= bucket:
                return i
        return len(SQuADQSL.SEQ_BUCKETS) - 1

    @staticmethod
    def get_bucket_seq_len(bucket_idx: int) -> int:
        """Get sequence length for a bucket index."""
        if 0 <= bucket_idx < len(SQuADQSL.SEQ_BUCKETS):
            return SQuADQSL.SEQ_BUCKETS[bucket_idx]
        return SQuADQSL.SEQ_BUCKETS[-1]

    def get_actual_seq_len(self, sample_idx: int) -> int:
        """Get actual (non-padded) sequence length for a sample."""
        if sample_idx in self._sample_seq_lengths:
            return self._sample_seq_lengths[sample_idx]

        features = self.get_features(sample_idx)
        attention_mask = features['attention_mask']
        actual_len = int(np.sum(attention_mask))
        self._sample_seq_lengths[sample_idx] = actual_len
        self._sample_buckets[sample_idx] = self.get_bucket_index(actual_len)
        return actual_len

    def get_sample_bucket(self, sample_idx: int) -> int:
        """Get bucket index for a sample."""
        if sample_idx not in self._sample_buckets:
            self.get_actual_seq_len(sample_idx)
        return self._sample_buckets.get(sample_idx, len(self.SEQ_BUCKETS) - 1)

    def load(self) -> None:
        """Load the dataset."""
        self.dataset.load()

    @property
    def total_sample_count(self) -> int:
        if not self.dataset._is_loaded:
            self.dataset.load()
        return self.dataset.total_count

    @property
    def performance_sample_count(self) -> int:
        return min(self._performance_sample_count, self.total_sample_count)

    def load_query_samples(self, sample_indices: List[int]) -> None:
        """Load samples into memory with optimized int64 format for C++ SUT."""
        if not self.dataset._is_loaded:
            self.dataset.load()

        for idx in sample_indices:
            if idx not in self._loaded_samples:
                features, _ = self.dataset.get_sample(idx)
                # Pre-convert to int64 and make contiguous for C++ SUT
                # This avoids conversion overhead during inference
                attention_mask = np.ascontiguousarray(
                    features['attention_mask'].flatten(), dtype=np.int64
                )
                optimized = {
                    'input_ids': np.ascontiguousarray(
                        features['input_ids'].flatten(), dtype=np.int64
                    ),
                    'attention_mask': attention_mask,
                    'token_type_ids': np.ascontiguousarray(
                        features['token_type_ids'].flatten(), dtype=np.int64
                    ),
                }
                self._loaded_samples[idx] = optimized

                actual_len = int(np.sum(attention_mask))
                self._sample_seq_lengths[idx] = actual_len
                self._sample_buckets[idx] = self.get_bucket_index(actual_len)

    def unload_query_samples(self, sample_indices: List[int]) -> None:
        """Unload samples from memory."""
        for idx in sample_indices:
            self._loaded_samples.pop(idx, None)

    def get_features(self, sample_index: int) -> Dict[str, np.ndarray]:
        """Get input features for a sample."""
        if sample_index in self._loaded_samples:
            return self._loaded_samples[sample_index]

        features, _ = self.dataset.get_sample(sample_index)
        return features

    def get_label(self, sample_index: int) -> List[str]:
        """Get ground truth answers for a sample."""
        return self.dataset.get_ground_truth(sample_index)
