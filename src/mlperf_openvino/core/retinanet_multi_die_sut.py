import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    from ..cpp import RetinaNetMultiDieCppSUT, CPP_AVAILABLE
    CPP_SUT_AVAILABLE = CPP_AVAILABLE and RetinaNetMultiDieCppSUT is not None
except ImportError:
    CPP_SUT_AVAILABLE = False
    RetinaNetMultiDieCppSUT = None

try:
    import mlperf_loadgen as lg
    LOADGEN_AVAILABLE = True
except ImportError:
    LOADGEN_AVAILABLE = False
    lg = None

from .image_multi_die_sut_base import ImageMultiDieSUTBase

logger = logging.getLogger(__name__)


class RetinaNetMultiDieCppSUTWrapper(ImageMultiDieSUTBase):

    MODEL_NAME = "RetinaNet"
    DEFAULT_OFFLINE_BATCH_SIZE = 2  # Larger batches than ResNet
    DEFAULT_OFFLINE_NIREQ_MULTIPLIER = 2  # Lower than ResNet due to 800x800 input
    DEFAULT_SERVER_NIREQ_MULTIPLIER = 2
    DEFAULT_EXPLICIT_BATCH_SIZE = 2  # Smaller for large inputs
    DEFAULT_BATCH_TIMEOUT_US = 1000  # Longer timeout for large input
    BATCH_SERVER_ACCURACY = True  # Use Offline-style batch dispatch for Server accuracy

    def _check_cpp_availability(self) -> None:
        if not CPP_SUT_AVAILABLE:
            raise ImportError(
                "C++ SUT not available. Build with: "
                "cd src/mlperf_openvino/cpp && mkdir build && cd build && cmake .. && make"
            )

    def _create_cpp_sut(
        self,
        model_path: str,
        device_prefix: str,
        batch_size: int,
        compile_props: Dict,
        use_nhwc: bool,
        nireq_multiplier: int
    ) -> Any:
        return RetinaNetMultiDieCppSUT(
            model_path,
            device_prefix,
            batch_size,
            compile_props,
            use_nhwc,
            nireq_multiplier
        )

    def get_predictions(self) -> Dict[int, Dict[str, np.ndarray]]:
        cpp_preds = self._cpp_sut.get_predictions()
        result = {}
        for idx, pred in cpp_preds.items():
            boxes = np.array(pred['boxes'], dtype=np.float32)
            scores = np.array(pred['scores'], dtype=np.float32)
            labels = np.array(pred['labels'], dtype=np.float32) if pred['labels'] is not None else np.array([])

            if boxes.size > 0 and boxes.ndim == 1:
                boxes = boxes.reshape(-1, 4)

            result[idx] = {
                'boxes': boxes,
                'scores': scores,
                'labels': labels,
            }
        return result

    def compute_accuracy(self) -> Dict[str, float]:
        predictions = self.get_predictions()

        if not predictions:
            logger.warning(f"No predictions (sample_count={self._sample_count}, query_count={self._query_count})")
            return {'mAP': 0.0, 'num_samples': 0}

        try:
            from ..datasets.coco_eval import evaluate_openimages_accuracy, PYCOCOTOOLS_AVAILABLE

            if PYCOCOTOOLS_AVAILABLE:
                coco_file = None
                data_path = Path(self.qsl.dataset.data_path)

                for name in ["annotations/openimages-mlperf.json", "openimages-mlperf.json"]:
                    path = data_path / name
                    if path.exists():
                        coco_file = str(path)
                        break

                if coco_file:
                    logger.info(f"Using pycocotools with {coco_file}")

                    sample_to_filename = None
                    if hasattr(self.qsl, 'get_sample_to_filename_mapping'):
                        sample_to_filename = self.qsl.get_sample_to_filename_mapping()
                        logger.info(f"Got filename mapping for {len(sample_to_filename)} samples")

                    return evaluate_openimages_accuracy(
                        predictions=predictions,
                        coco_annotations_file=coco_file,
                        input_size=800,
                        model_labels_zero_indexed=True,
                        boxes_in_pixels=True,
                        sample_to_filename=sample_to_filename,
                    )
                else:
                    logger.warning("COCO annotations file not found, using fallback mAP calculation")
        except Exception as e:
            logger.warning(f"pycocotools evaluation failed: {e}, using fallback")
            import traceback
            traceback.print_exc()

        pred_list = []
        indices = []

        for sample_idx in sorted(predictions.keys()):
            pred_list.append(predictions[sample_idx])
            indices.append(sample_idx)

        return self.qsl.dataset.compute_accuracy(pred_list, indices)


def is_retinanet_multi_die_cpp_available() -> bool:
    return CPP_SUT_AVAILABLE and LOADGEN_AVAILABLE
