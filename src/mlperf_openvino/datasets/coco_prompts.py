"""COCO 2014 Captions dataset for Stable Diffusion XL benchmark."""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseDataset, QuerySampleLibrary

logger = logging.getLogger(__name__)

MAX_SEQUENCE_LENGTH = 77
IMAGE_SIZE = 1024
LATENT_SIZE = 128
GUIDANCE_SCALE = 8.0
NUM_INFERENCE_STEPS = 20


class COCOPromptsDataset(BaseDataset):

    def __init__(
        self,
        data_path: str,
        count: Optional[int] = None,
        use_latents: bool = False,
        guidance_scale: float = GUIDANCE_SCALE,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
    ):
        super().__init__(data_path=data_path, count=count)

        self.data_path = Path(data_path)
        self.use_latents = use_latents
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps

        self._samples: List[Dict[str, Any]] = []
        self._latents_cache: Dict[int, np.ndarray] = {}
        self._is_loaded = False

        self._tokenizer = None

    def _load_tokenizer(self):
        if self._tokenizer is not None:
            return

        try:
            from transformers import CLIPTokenizer
            self._tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            logger.debug("Loaded CLIP tokenizer")
        except ImportError:
            logger.warning(
                "transformers not installed. Install with: pip install transformers"
            )
            self._tokenizer = None

    def load(self) -> None:
        if self._is_loaded:
            return

        logger.debug("Loading COCO prompts dataset from %s", self.data_path)

        loaded = False

        # Format 1: MLCommons TSV format
        tsv_file = self.data_path / "coco-1024.tsv"
        if not tsv_file.exists():
            tsv_file = self.data_path / "captions_source.tsv"
        if not tsv_file.exists():
            tsv_file = self.data_path / "captions.tsv"
        if tsv_file.exists():
            self._load_from_tsv(tsv_file)
            loaded = True

        # Format 2: COCO JSON annotations
        if not loaded:
            json_candidates = [
                self.data_path / "captions" / "captions_val2014.json",
                self.data_path / "annotations" / "captions_val2014.json",
                self.data_path / "captions.json",
                self.data_path / "captions_val2014.json",
            ]
            for json_file in json_candidates:
                if json_file.exists():
                    self._load_from_coco_json(json_file)
                    loaded = True
                    break

        # Format 3: Simple text file (one caption per line)
        if not loaded:
            txt_candidates = [
                self.data_path / "prompts.txt",
                self.data_path / "captions.txt",
            ]
            for txt_file in txt_candidates:
                if txt_file.exists():
                    self._load_from_txt(txt_file)
                    loaded = True
                    break

        if not self._samples:
            logger.warning(f"No samples found in {self.data_path}")
            logger.info("For COCO dataset, expected structure:")
            logger.info("  data_path/")
            logger.info("  ├── coco-1024.tsv (MLCommons format)")
            logger.info("  └── coco-1024/ (reference images)")
            logger.info("")
            logger.info("Or use mlperf-ov download-dataset --dataset coco2014")

        # MLPerf uses 5000 samples
        if self.count and self.count < len(self._samples):
            self._samples = self._samples[:self.count]

        if self.use_latents:
            self._load_latents()

        latent_info = "shared latent" if self._latents_cache else "no latents"
        print(f"[Dataset] {len(self._samples)} prompts, {latent_info}", file=sys.stderr)
        self._is_loaded = True

    def _load_from_tsv(self, tsv_file: Path) -> None:
        logger.debug("Loading from TSV: %s", tsv_file)

        is_mlcommons_format = False
        with open(tsv_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split('\t')

                if line_num == 0 and parts[0] == 'id':
                    is_mlcommons_format = True
                    continue

                if is_mlcommons_format and len(parts) >= 3:
                    # captions_source.tsv: id, image_id, caption, height, width, ...
                    image_id = parts[1]
                    caption = parts[2]
                elif len(parts) >= 2:
                    image_id = parts[0]
                    caption = parts[1]
                else:
                    image_id = str(line_num)
                    caption = parts[0]

                image_path = self._find_reference_image(image_id)

                self._samples.append({
                    "id": image_id,
                    "caption": caption,
                    "image_path": image_path,
                })

    def _load_from_coco_json(self, json_file: Path) -> None:
        logger.debug("Loading from COCO JSON: %s", json_file)

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        id_to_filename = {}
        if 'images' in data:
            for img in data['images']:
                id_to_filename[img['id']] = img.get('file_name', '')

        # Take the first caption for each image (MLPerf uses one caption per image)
        seen_images = set()
        annotations = data.get('annotations', [])

        for ann in annotations:
            image_id = ann.get('image_id')
            if image_id in seen_images:
                continue

            seen_images.add(image_id)
            caption = ann.get('caption', '')

            filename = id_to_filename.get(image_id, f"{image_id:012d}.jpg")
            image_path = self._find_reference_image(str(image_id), filename)

            self._samples.append({
                "id": str(image_id),
                "caption": caption,
                "image_path": image_path,
            })

    def _load_from_txt(self, txt_file: Path) -> None:
        logger.debug("Loading from TXT: %s", txt_file)

        with open(txt_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                caption = line.strip()
                if not caption:
                    continue

                self._samples.append({
                    "id": str(idx),
                    "caption": caption,
                    "image_path": None,
                })

    def _find_reference_image(
        self,
        image_id: str,
        filename: Optional[str] = None
    ) -> Optional[str]:
        search_dirs = [
            self.data_path / "coco-1024",
            self.data_path / "images" / "val2014",
            self.data_path / "val2014",
            self.data_path / "images",
        ]

        filenames = []
        if filename:
            filenames.append(filename)
        filenames.extend([
            f"{image_id}.png",
            f"{image_id}.jpg",
            f"COCO_val2014_{int(image_id):012d}.jpg" if image_id.isdigit() else None,
            f"{int(image_id):012d}.png" if image_id.isdigit() else None,
        ])
        filenames = [f for f in filenames if f]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for fname in filenames:
                candidate = search_dir / fname
                if candidate.exists():
                    return str(candidate)

        return None

    def _load_latents(self) -> None:
        # MLCommons: ALL samples share the SAME latent (1, 4, 128, 128), generated with seed=0
        search_paths = [
            self.data_path / "latents" / "latents.pt",
            self.data_path / "latents" / "latents.npy",
            self.data_path / "latents.pt",
            self.data_path / "latents.npy",
            Path.cwd() / "data" / "coco2014" / "latents" / "latents.pt",
            Path.cwd() / "data" / "coco2014" / "latents" / "latents.npy",
        ]

        latents_file = None
        for path in search_paths:
            if path.exists():
                latents_file = path
                break

        if latents_file is None:
            logger.warning(
                "Pre-computed latents file not found. "
                "Generating locally (seed=0). For official Closed Division, "
                "use the exact MLCommons latents.pt from the inference repo."
            )
            self._generate_latents()
            return

        logger.debug("Loading pre-computed latents from %s", latents_file)

        try:
            if latents_file.suffix == '.npy':
                latent = np.load(str(latents_file)).astype(np.float32)
            else:
                import torch
                latent = torch.load(str(latents_file), map_location='cpu', weights_only=True)
                if hasattr(latent, 'numpy'):
                    latent = latent.numpy().astype(np.float32)
                else:
                    latent = np.array(latent, dtype=np.float32)

            if latent.ndim == 3:
                latent = latent[np.newaxis, :]

            logger.debug("Latent tensor shape: %s", latent.shape)

            # Same latent for ALL samples
            self._shared_latent = latent
            for idx in range(len(self._samples)):
                self._latents_cache[idx] = latent.squeeze(0)

            logger.debug(
                "Loaded shared latent for %d samples", len(self._samples)
            )

        except Exception as e:
            logger.warning(f"Failed to load latents: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            self._generate_latents()

    def _generate_latents(self) -> None:
        # MLCommons latent.py: randn_tensor((1, 4, 128, 128), seed=0)
        # Same latent is used for ALL samples
        try:
            import torch
            generator = torch.Generator("cpu")
            generator.manual_seed(0)

            latent = torch.randn(1, 4, LATENT_SIZE, LATENT_SIZE, generator=generator)
            latent_np = latent.squeeze(0).numpy().astype(np.float32)

            for idx in range(len(self._samples)):
                self._latents_cache[idx] = latent_np

            self._shared_latent = latent.numpy().astype(np.float32)

            logger.debug(
                "Generated shared latent (seed=0), shape: %s", latent_np.shape
            )

            save_dir = self.data_path / "latents"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "latents.pt"
            torch.save(latent, str(save_path))
            logger.debug("Saved generated latent to %s", save_path)

        except Exception as e:
            logger.error(f"Failed to generate latents: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def total_count(self) -> int:
        return len(self._samples)

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def get_sample(self, index: int) -> Tuple[Dict[str, Any], str]:
        sample = self._samples[index]
        caption = sample['caption']

        input_dict = {
            'prompt': caption,
            'guidance_scale': self.guidance_scale,
            'num_inference_steps': self.num_inference_steps,
        }

        if index in self._latents_cache:
            input_dict['latents'] = self._latents_cache[index]

        if sample.get('image_path'):
            input_dict['reference_image_path'] = sample['image_path']

        return input_dict, caption

    def get_prompt(self, index: int) -> str:
        return self._samples[index]['caption']

    def get_samples(self, indices: List[int]) -> Tuple[List[Dict[str, Any]], List[str]]:
        inputs = []
        labels = []
        for idx in indices:
            input_dict, caption = self.get_sample(idx)
            inputs.append(input_dict)
            labels.append(caption)
        return inputs, labels

    def get_reference_image_path(self, index: int) -> Optional[str]:
        return self._samples[index].get('image_path')

    def tokenize(self, text: str) -> Dict[str, np.ndarray]:
        self._load_tokenizer()

        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not available")

        tokens = self._tokenizer(
            text,
            padding="max_length",
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            return_tensors="np",
        )

        return {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask'],
        }

    def postprocess(
        self,
        results: np.ndarray,
        indices: List[int]
    ) -> List[np.ndarray]:
        if isinstance(results, list):
            return results
        return [results[i] for i in range(len(indices))]

    def compute_accuracy(
        self,
        generated_images: List[np.ndarray],
        indices: List[int]
    ) -> Dict[str, float]:
        metrics = {
            'clip_score': 0.0,
            'fid_score': 0.0,
            'num_samples': len(generated_images),
        }

        if not generated_images:
            return metrics

        logger.debug("Accuracy evaluation: %d images", len(generated_images))
        if generated_images:
            img0 = generated_images[0]
            if isinstance(img0, np.ndarray):
                logger.debug(
                    "  Sample image: shape=%s, dtype=%s, range=[%s, %s]",
                    img0.shape, img0.dtype, img0.min(), img0.max(),
                )

        clip_score = self._compute_clip_score(generated_images, indices)
        metrics['clip_score'] = clip_score

        reference_images = []
        for idx in indices:
            ref_path = self.get_reference_image_path(idx)
            if ref_path:
                reference_images.append(ref_path)

        statistics_path = self._find_statistics_file()
        if reference_images or statistics_path:
            fid_score = self._compute_fid_score(
                generated_images, reference_images, statistics_path=statistics_path
            )
            metrics['fid_score'] = fid_score

        return metrics

    def _compute_clip_score(
        self,
        images: List[np.ndarray],
        indices: List[int]
    ) -> float:
        try:
            import torch
            import open_clip
            from PIL import Image
        except ImportError:
            logger.warning(
                "open_clip not installed. Install with: pip install open_clip_torch\n"
                "This is required for MLCommons-compliant CLIP score computation."
            )
            return 0.0

        try:
            # MLCommons reference: ViT-B/32 with openai weights.
            # Must use the QuickGELU variant: OpenAI CLIP was trained with
            # QuickGELU (x*sigmoid(1.702*x)), but newer open_clip defaults to
            # standard nn.GELU.  Using the wrong activation produces wrong
            # embeddings and systematically lower CLIP scores.
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32-quickgelu',
                pretrained='openai'
            )
            tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')
            model.eval()

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)

            scores = []
            for img, idx in zip(images, indices):
                prompt = self.get_prompt(idx)

                if isinstance(img, np.ndarray):
                    if img.dtype == np.float32 or img.dtype == np.float64:
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    elif img.dtype != np.uint8:
                        img = img.astype(np.uint8)

                    if img.ndim == 2:
                        # Grayscale to RGB
                        img = np.stack([img, img, img], axis=-1)
                    elif img.ndim == 3 and img.shape[0] == 3:
                        # CHW to HWC
                        img = np.transpose(img, (1, 2, 0))

                    pil_img = Image.fromarray(img)
                else:
                    pil_img = img

                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')

                image_input = preprocess(pil_img).unsqueeze(0).to(device)
                text_input = tokenizer([prompt]).to(device)

                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                    text_features = model.encode_text(text_input)

                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    # Cosine similarity scaled by 100
                    score = (image_features @ text_features.T).item() * 100
                    scores.append(score)
                    logger.debug("CLIP[%d]: %.2f", idx, score)

            mean_clip = float(np.mean(scores)) if scores else 0.0
            logger.debug("CLIP mean=%.4f, std=%.4f, n=%d", mean_clip, np.std(scores), len(scores))
            return mean_clip

        except Exception as e:
            import traceback
            logger.error(f"Error computing CLIP score: {e}")
            logger.error(traceback.format_exc())
            return 0.0

    def _compute_fid_score(
        self,
        generated_images: List[np.ndarray],
        reference_paths: List[str],
        statistics_path: Optional[str] = None
    ) -> float:
        # CRITICAL: Must use pytorch-fid InceptionV3 weights (pt_inception-2015-12-05-6726825d.pth),
        # NOT torchvision weights. val2014.npz statistics were computed with these exact weights.
        try:
            from scipy import linalg
            import torch
            from torchvision import transforms
            from PIL import Image
        except ImportError:
            logger.warning("Required packages not installed for FID computation")
            return 0.0

        num_gen = len(generated_images)

        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            try:
                from pytorch_fid.inception import InceptionV3
                dims = 2048
                block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
                model = InceptionV3([block_idx]).to(device)
                logger.debug("FID: Using pytorch-fid InceptionV3 (correct weights)")
            except ImportError:
                logger.error(
                    "pytorch-fid is NOT installed. FID scores WILL NOT match MLCommons. "
                    "Install with: pip install pytorch-fid"
                )
                return 0.0

            model.eval()

            # ToTensor scales to [0,1]; pytorch-fid InceptionV3 resizes to 299x299
            # and normalizes to [-1,1] internally (resize_input=True, normalize_input=True).
            # Do NOT resize externally — pytorch-fid uses F.interpolate internally,
            # and PIL Resize produces different pixel values for the same downscale.
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])

            def get_features(images_or_paths, is_path=False):
                features = []
                for item in images_or_paths:
                    if is_path:
                        img = Image.open(item).convert('RGB')
                    else:
                        if isinstance(item, np.ndarray):
                            img = Image.fromarray(item.astype(np.uint8))
                        else:
                            img = item
                        if hasattr(img, 'mode') and img.mode != 'RGB':
                            img = img.convert('RGB')

                    tensor = transform(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        feat = model(tensor)
                        # pytorch_fid returns list of feature tensors per block
                        if isinstance(feat, list):
                            feat = feat[0]
                        if feat.dim() > 2:
                            feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                        feat = feat.squeeze().cpu().numpy()
                    features.append(feat)
                return np.array(features)

            gen_features = get_features(generated_images, is_path=False)
            logger.debug(
                "FID: gen_features shape=%s, mean=%.4f, std=%.4f",
                gen_features.shape, gen_features.mean(), gen_features.std(),
            )

            mu_gen = np.mean(gen_features, axis=0)
            sigma_gen = np.cov(gen_features, rowvar=False)

            if statistics_path is None:
                statistics_path = self._find_statistics_file()

            if statistics_path and Path(statistics_path).exists():
                logger.debug("FID: Using pre-computed statistics from %s", statistics_path)
                stats = np.load(statistics_path)
                mu_ref = stats['mu']
                sigma_ref = stats['sigma']
            else:
                logger.debug("FID: Computing statistics from reference images")
                ref_features = get_features(reference_paths, is_path=True)
                mu_ref = np.mean(ref_features, axis=0)
                sigma_ref = np.cov(ref_features, rowvar=False)

            diff = mu_gen - mu_ref
            mean_term = float(diff @ diff)
            tr_gen = float(np.trace(sigma_gen))
            tr_ref = float(np.trace(sigma_ref))

            if num_gen < 2048:
                logger.debug(
                    "FID: %d samples — covariance rank limited, FID may differ from 5000-sample target",
                    num_gen,
                )

            covmean, _ = linalg.sqrtm(sigma_gen @ sigma_ref, disp=False)
            if np.iscomplexobj(covmean):
                if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                    logger.warning("FID: Significant complex values in sqrtm")
                covmean = covmean.real

            tr_covmean = float(np.trace(covmean))
            trace_term = tr_gen + tr_ref - 2 * tr_covmean
            fid = mean_term + trace_term

            logger.debug(
                "FID=%.4f (mean_diff=%.4f + trace=%.4f)", fid, mean_term, trace_term,
            )

            return float(fid)

        except Exception as e:
            logger.warning(f"Error computing FID score: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return 0.0

    def _find_statistics_file(self) -> Optional[str]:
        search_paths = [
            Path(self.data_path) / "val2014.npz",
            Path(self.data_path) / "fid_statistics.npz",
            Path(self.data_path).parent / "val2014.npz",
            Path.cwd() / "val2014.npz",
            Path.cwd() / "data" / "val2014.npz",
            Path.cwd() / "data" / "coco2014" / "val2014.npz",
        ]

        for path in search_paths:
            if path.exists():
                return str(path)

        return None

    def get_latents(self, index: int) -> Optional[np.ndarray]:
        if not self._is_loaded:
            self.load()

        sample = self._samples[index]
        return sample.get('latents', None)

    def load_latents(self, latents_path: str) -> int:
        latents_file = Path(latents_path)
        if not latents_file.exists():
            logger.warning(f"Latents file not found: {latents_path}")
            return 0

        try:
            if latents_file.suffix == '.pt':
                import torch
                latents_data = torch.load(latents_path)
                if isinstance(latents_data, dict):
                    latents_list = latents_data.get('latents', [])
                else:
                    latents_list = latents_data
            elif latents_file.suffix == '.npy':
                latents_list = np.load(latents_path, allow_pickle=True)
            elif latents_file.suffix == '.npz':
                data = np.load(latents_path)
                latents_list = data['latents'] if 'latents' in data else list(data.values())[0]
            else:
                logger.warning(f"Unsupported latents file format: {latents_file.suffix}")
                return 0

            loaded = 0
            for i, latent in enumerate(latents_list):
                if i < len(self._samples):
                    if isinstance(latent, np.ndarray):
                        self._samples[i]['latents'] = latent
                    else:
                        self._samples[i]['latents'] = latent.numpy() if hasattr(latent, 'numpy') else np.array(latent)
                    loaded += 1

            logger.debug("Loaded %d pre-generated latents from %s", loaded, latents_path)
            return loaded

        except Exception as e:
            logger.warning(f"Error loading latents: {e}")
            return 0

    def get_compliance_indices(self) -> List[int]:
        np.random.seed(42)  # MLCommons compliance seed
        indices = np.random.choice(len(self._samples), size=10, replace=False)
        return sorted(indices.tolist())

    def save_compliance_images(
        self,
        images: Dict[int, np.ndarray],
        output_dir: str
    ) -> List[str]:
        from PIL import Image

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        compliance_indices = self.get_compliance_indices()
        saved_paths = []

        for idx in compliance_indices:
            if idx in images:
                img_array = images[idx]
                if isinstance(img_array, np.ndarray):
                    pil_img = Image.fromarray(img_array.astype(np.uint8))
                else:
                    pil_img = img_array

                filename = f"compliance_sample_{idx:05d}.png"
                filepath = output_path / filename
                pil_img.save(filepath, 'PNG')
                saved_paths.append(str(filepath))

                logger.debug("Saved compliance image: %s", filepath)

        logger.debug("Saved %d compliance images to %s", len(saved_paths), output_dir)
        return saved_paths


class COCOPromptsQSL(QuerySampleLibrary):

    def __init__(
        self,
        data_path: str,
        count: Optional[int] = None,
        performance_sample_count: int = 5000,  # MLPerf default
        guidance_scale: float = GUIDANCE_SCALE,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        use_latents: bool = True,
    ):
        super().__init__()

        self.dataset = COCOPromptsDataset(
            data_path=data_path,
            count=count,
            use_latents=use_latents,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )

        self._performance_sample_count = performance_sample_count
        self._loaded_samples: Dict[int, Dict[str, Any]] = {}

    def load(self) -> None:
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
        for idx in sample_indices:
            if idx not in self._loaded_samples:
                input_dict, _ = self.dataset.get_sample(idx)
                self._loaded_samples[idx] = input_dict

    def unload_query_samples(self, sample_indices: List[int]) -> None:
        for idx in sample_indices:
            self._loaded_samples.pop(idx, None)

    def get_features(self, sample_index: int) -> Dict[str, Any]:
        if sample_index in self._loaded_samples:
            return self._loaded_samples[sample_index]
        else:
            input_dict, _ = self.dataset.get_sample(sample_index)
            return input_dict

    def get_prompt(self, sample_index: int) -> str:
        return self.dataset.get_prompt(sample_index)

    def get_label(self, sample_index: int) -> str:
        return self.dataset.get_prompt(sample_index)
