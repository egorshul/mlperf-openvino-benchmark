"""Factory for creating model-specific multi-die SUTs."""

import logging
from typing import Any, Optional

from .config import BenchmarkConfig, ModelType, TestMode
from ..datasets.base import QuerySampleLibrary

logger = logging.getLogger(__name__)


class SUTFactory:
    """Factory for creating multi-die SUTs based on model type."""

    @staticmethod
    def create_multi_die_sut(
        model_type: ModelType,
        config: BenchmarkConfig,
        qsl: QuerySampleLibrary,
        backend: Optional[Any] = None,
        encoder_path: Optional[str] = None,
        decoder_path: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> Any:
        """Create the appropriate multi-die SUT for the given model type.

        Args:
            model_type: Type of model (RESNET50, BERT, RETINANET, WHISPER, SDXL)
            config: Benchmark configuration
            qsl: Query Sample Library
            backend: Optional backend for fallback (Python SUTs)
            encoder_path: Path to encoder model (for Whisper)
            decoder_path: Path to decoder model (for Whisper)
            model_path: Path to model directory (for SDXL)

        Returns:
            Configured SUT instance

        Raises:
            RuntimeError: If C++ SUT not available and no backend provided
            ValueError: If model type not supported
        """
        is_accuracy_mode = config.test_mode == TestMode.ACCURACY_ONLY

        if model_type == ModelType.RESNET50:
            return SUTFactory._create_resnet_sut(config, qsl, is_accuracy_mode, backend)
        elif model_type == ModelType.BERT:
            return SUTFactory._create_bert_sut(config, qsl, is_accuracy_mode, backend)
        elif model_type == ModelType.RETINANET:
            return SUTFactory._create_retinanet_sut(config, qsl, is_accuracy_mode, backend)
        elif model_type == ModelType.WHISPER:
            return SUTFactory._create_whisper_sut(config, qsl, encoder_path, decoder_path)
        elif model_type == ModelType.SDXL:
            return SUTFactory._create_sdxl_sut(config, qsl, model_path)
        else:
            raise ValueError(f"Multi-die SUT not supported for model type: {model_type}")

    @staticmethod
    def _create_resnet_sut(
        config: BenchmarkConfig,
        qsl: QuerySampleLibrary,
        is_accuracy_mode: bool,
        backend: Optional[Any],
    ) -> Any:
        """Create ResNet multi-die SUT."""
        try:
            from .resnet_multi_die_sut import (
                ResNetMultiDieCppSUTWrapper,
                is_resnet_multi_die_cpp_available
            )

            if is_resnet_multi_die_cpp_available():
                logger.info(f"Using ResNet C++ multi-die SUT on {config.openvino.device}")
                sut = ResNetMultiDieCppSUTWrapper(
                    config=config,
                    qsl=qsl,
                    scenario=config.scenario,
                )
                sut.load(is_accuracy_mode=is_accuracy_mode)
                return sut
        except ImportError as e:
            logger.warning(f"C++ ResNet multi-die SUT not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to create C++ ResNet multi-die SUT: {e}, falling back to Python")

        if backend is None:
            raise RuntimeError("C++ SUT not available and Python backend is None. Build C++ SUT first.")

        from .resnet_multi_device_sut import ResNetMultiDeviceSUT
        logger.info(f"Using ResNet Python multi-die SUT ({backend.num_dies} dies)")
        return ResNetMultiDeviceSUT(
            config=config,
            backend=backend,
            qsl=qsl,
            scenario=config.scenario,
        )

    @staticmethod
    def _create_bert_sut(
        config: BenchmarkConfig,
        qsl: QuerySampleLibrary,
        is_accuracy_mode: bool,
        backend: Optional[Any],
    ) -> Any:
        """Create BERT multi-die SUT."""
        try:
            from .bert_multi_die_sut import (
                BertMultiDieSUTWrapper,
                is_bert_multi_die_cpp_available
            )

            if is_bert_multi_die_cpp_available():
                logger.info(f"Using BERT C++ multi-die SUT on {config.openvino.device}")
                sut = BertMultiDieSUTWrapper(
                    config=config,
                    qsl=qsl,
                    scenario=config.scenario,
                )
                sut.load(is_accuracy_mode=is_accuracy_mode)
                return sut
        except ImportError as e:
            logger.warning(f"C++ BERT multi-die SUT not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to create C++ BERT multi-die SUT: {e}")

        if backend is None:
            raise RuntimeError("C++ SUT not available and Python backend is None. Build C++ SUT first.")

        from .bert_sut import BertSUT
        logger.info(f"Using BERT Python SUT on {config.openvino.device}")
        return BertSUT(
            config=config,
            backend=backend,
            qsl=qsl,
            scenario=config.scenario,
        )

    @staticmethod
    def _create_retinanet_sut(
        config: BenchmarkConfig,
        qsl: QuerySampleLibrary,
        is_accuracy_mode: bool,
        backend: Optional[Any],
    ) -> Any:
        """Create RetinaNet multi-die SUT."""
        try:
            from .retinanet_multi_die_sut import (
                RetinaNetMultiDieCppSUTWrapper,
                is_retinanet_multi_die_cpp_available
            )

            if is_retinanet_multi_die_cpp_available():
                logger.info(f"Using RetinaNet C++ multi-die SUT on {config.openvino.device}")
                sut = RetinaNetMultiDieCppSUTWrapper(
                    config=config,
                    qsl=qsl,
                    scenario=config.scenario,
                )
                sut.load(is_accuracy_mode=is_accuracy_mode)
                return sut
        except ImportError as e:
            logger.warning(f"C++ RetinaNet multi-die SUT not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to create C++ RetinaNet multi-die SUT: {e}")

        if backend is None:
            raise RuntimeError("C++ SUT not available and Python backend is None. Build C++ SUT first.")

        from .retinanet_sut import RetinaNetSUT
        logger.info(f"Using RetinaNet Python SUT on {config.openvino.device}")
        return RetinaNetSUT(
            config=config,
            backend=backend,
            qsl=qsl,
            scenario=config.scenario,
        )

    @staticmethod
    def _create_whisper_sut(
        config: BenchmarkConfig,
        qsl: QuerySampleLibrary,
        encoder_path: Optional[str],
        decoder_path: Optional[str],
    ) -> Any:
        """Create Whisper multi-die SUT (Python only, no C++ implementation)."""
        if not encoder_path or not decoder_path:
            raise ValueError(
                "Whisper multi-die SUT requires encoder_path and decoder_path"
            )

        from .whisper_sut import WhisperMultiDieSUT
        logger.info(f"Using Whisper Python multi-die SUT on {config.openvino.device}")
        return WhisperMultiDieSUT(
            config=config,
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            qsl=qsl,
            scenario=config.scenario,
        )

    @staticmethod
    def _create_sdxl_sut(
        config: BenchmarkConfig,
        qsl: QuerySampleLibrary,
        model_path: Optional[str],
    ) -> Any:
        """Create SDXL multi-die SUT (Python only, no C++ implementation)."""
        if not model_path:
            raise ValueError(
                "SDXL multi-die SUT requires model_path"
            )

        from .sdxl_sut import SDXLMultiDieSUT
        logger.info(f"Using SDXL Python multi-die SUT on {config.openvino.device}")
        return SDXLMultiDieSUT(
            config=config,
            model_path=model_path,
            qsl=qsl,
            scenario=config.scenario,
        )
