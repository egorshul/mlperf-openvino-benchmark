"""
Whisper-specific System Under Test implementation.

This module provides SUT implementation optimized for Whisper ASR model,
using optimum-intel OVModelForSpeechSeq2Seq for proper encoder-decoder inference.
"""

import array
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

try:
    import mlperf_loadgen as lg
    LOADGEN_AVAILABLE = True
except ImportError:
    LOADGEN_AVAILABLE = False
    lg = None

# Optimum-Intel for proper Whisper inference
try:
    from optimum.intel.openvino import OVModelForSpeechSeq2Seq
    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False
    OVModelForSpeechSeq2Seq = None

from .config import BenchmarkConfig, Scenario
from ..backends.base import BaseBackend
from ..datasets.librispeech import LibriSpeechQSL

logger = logging.getLogger(__name__)


class CompiledModelWrapper:
    """
    Wrapper that provides both CompiledModel and InferRequest interfaces.

    optimum-intel expects the request object to support:
    - Callable interface: request(inputs, share_inputs=True, share_outputs=True)
    - Async interface: request.start_async(), request.wait(), etc.

    This wrapper delegates to the appropriate underlying object.
    """

    def __init__(self, compiled_model):
        """
        Args:
            compiled_model: OpenVINO CompiledModel
        """
        self._compiled = compiled_model
        self._infer_request = compiled_model.create_infer_request()

    def __call__(self, inputs, share_inputs=True, share_outputs=True):
        """Callable interface for synchronous inference (like CompiledModel)."""
        return self._compiled(inputs, share_inputs=share_inputs, share_outputs=share_outputs)

    def start_async(self, inputs=None, **kwargs):
        """Start async inference (like InferRequest).

        Note: InferRequest.start_async() doesn't accept share_inputs/share_outputs,
        so we ignore those kwargs if passed.
        """
        # InferRequest.start_async() signature varies by OV version
        # Safest approach: set inputs via tensors if needed, then call start_async()
        if inputs is not None:
            # Set input tensors from dict
            for name, tensor in inputs.items():
                try:
                    self._infer_request.set_tensor(name, tensor)
                except Exception:
                    # Try by index if name doesn't work
                    pass
        self._infer_request.start_async()

    def wait(self):
        """Wait for async inference to complete."""
        self._infer_request.wait()

    def get_output_tensor(self, index=0):
        """Get output tensor by index."""
        return self._infer_request.get_output_tensor(index)

    def get_tensor(self, name):
        """Get tensor by name."""
        return self._infer_request.get_tensor(name)

    def set_tensor(self, name, tensor):
        """Set tensor by name."""
        self._infer_request.set_tensor(name, tensor)

    def set_input_tensor(self, index, tensor):
        """Set input tensor by index."""
        self._infer_request.set_input_tensor(index, tensor)

    def set_output_tensor(self, index, tensor):
        """Set output tensor by index."""
        self._infer_request.set_output_tensor(index, tensor)

    def infer(self, inputs=None, **kwargs):
        """Synchronous inference using InferRequest.

        Note: ignores share_inputs/share_outputs as InferRequest doesn't use them.
        """
        if inputs is not None:
            return self._infer_request.infer(inputs)
        return self._infer_request.infer()

    @property
    def results(self):
        """Get inference results."""
        return self._infer_request.results

    @property
    def input_tensors(self):
        """Get input tensors."""
        return [self._infer_request.get_input_tensor(i)
                for i in range(len(self._compiled.inputs))]

    @property
    def output_tensors(self):
        """Get output tensors."""
        return [self._infer_request.get_output_tensor(i)
                for i in range(len(self._compiled.outputs))]

    def __getattr__(self, name):
        """Delegate unknown attributes to InferRequest."""
        return getattr(self._infer_request, name)


class WhisperOptimumSUT:
    """
    System Under Test for Whisper ASR using Optimum-Intel.

    Uses OVModelForSpeechSeq2Seq for proper encoder-decoder inference
    with correct KV-cache handling and token generation.

    Supports both CPU and NPU devices via optimum-intel.
    Supports hybrid mode: encoder on NPU, decoder on CPU (for devices
    where decoder compilation fails).
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        model_path: Union[str, Path],
        qsl: LibriSpeechQSL,
        scenario: Scenario = Scenario.OFFLINE,
        max_new_tokens: int = 440,  # Leave room for special tokens (448 - 8)
        device: Optional[str] = None,
        encoder_device: Optional[str] = None,
        decoder_device: Optional[str] = None,
    ):
        """
        Initialize Whisper SUT using Optimum-Intel.

        Args:
            config: Benchmark configuration
            model_path: Path to OpenVINO Whisper model directory
            qsl: Query Sample Library
            scenario: MLPerf scenario
            max_new_tokens: Maximum tokens to generate
            device: Device to run inference on (CPU, NPU, NPU.0, etc.)
            encoder_device: Specific device for encoder (overrides device)
            decoder_device: Specific device for decoder (overrides device)
        """
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")

        if not OPTIMUM_AVAILABLE:
            raise ImportError(
                "Optimum-Intel is required for Whisper inference. "
                "Install with: pip install optimum[openvino]"
            )

        self.config = config
        self.model_path = Path(model_path)
        self.qsl = qsl
        self.scenario = scenario
        self.max_new_tokens = max_new_tokens

        # Get device from config if not specified
        if device is None:
            device = config.openvino.device if hasattr(config, 'openvino') else "CPU"
        self.device = device.upper() if device else "CPU"

        # Support hybrid mode: separate devices for encoder and decoder
        self.encoder_device = encoder_device.upper() if encoder_device else self.device
        self.decoder_device = decoder_device.upper() if decoder_device else self.device
        self._hybrid_mode = (self.encoder_device != self.decoder_device)

        # Results storage
        self._predictions: Dict[int, str] = {}
        self._query_count = 0
        self._sample_count = 0

        # Progress tracking
        self._progress_bar: Optional[Any] = None
        self._start_time = 0.0
        self._last_progress_update = 0.0
        self._progress_update_interval = 0.5  # seconds

        # Create LoadGen handles
        self._sut_handle = None
        self._qsl_handle = None

        # Load model and processor
        self._load_model()

    def _load_model(self) -> None:
        """Load Whisper model using Optimum-Intel with hybrid device support."""
        from transformers import AutoProcessor

        if self._hybrid_mode:
            logger.info(
                f"Loading Whisper model from {self.model_path} "
                f"(encoder: {self.encoder_device}, decoder: {self.decoder_device})"
            )
        else:
            logger.info(f"Loading Whisper model from {self.model_path} on device {self.device}")

        # Load processor (tokenizer + feature extractor)
        logger.debug(f"Loading processor from {self.model_path}")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            logger.debug("Processor loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load processor from model path: {e}")
            logger.info("Falling back to openai/whisper-large-v3 processor")
            self.processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

        # Build OpenVINO config
        ov_config = {"CACHE_DIR": ""}

        # Add device-specific properties from config
        if hasattr(self.config, 'openvino') and hasattr(self.config.openvino, 'device_properties'):
            device_props = self.config.openvino.device_properties
            if device_props:
                for key, value in device_props.items():
                    ov_config[key] = value

        logger.info(f"OV config: {ov_config}")

        # List model files
        if self.model_path.is_dir():
            xml_files = list(self.model_path.glob("*.xml"))
            logger.info(f"Model files found: {[f.name for f in xml_files]}")

        # Determine initial device for loading
        initial_device = self.encoder_device if self._hybrid_mode else self.device

        # Load OpenVINO model
        logger.info(f"Loading OVModelForSpeechSeq2Seq (initial device: {initial_device})...")
        try:
            self.model = OVModelForSpeechSeq2Seq.from_pretrained(
                self.model_path,
                ov_config=ov_config,
                device=initial_device,
                compile=False,  # Don't compile yet
            )
            logger.info("Model loaded successfully (not compiled yet)")

            # Log submodels info
            submodels = ['encoder', 'decoder', 'decoder_with_past']
            for name in submodels:
                if hasattr(self.model, name) and getattr(self.model, name) is not None:
                    submodel = getattr(self.model, name)
                    logger.info(f"  Submodel '{name}': present")
                    if hasattr(submodel, 'model') and submodel.model is not None:
                        try:
                            inputs = [inp.get_any_name() for inp in submodel.model.inputs]
                            outputs = [out.get_any_name() for out in submodel.model.outputs]
                            logger.info(f"    Inputs ({len(inputs)}): {inputs[:3]}...")
                            logger.info(f"    Outputs ({len(outputs)}): {outputs[:3]}...")
                        except Exception:
                            pass
                else:
                    logger.info(f"  Submodel '{name}': NOT present")

            # Compile submodels with device-specific handling
            self._compile_submodels(ov_config)

        except Exception as load_error:
            logger.error(f"Failed to load/compile model: {load_error}")
            logger.error(f"Error type: {type(load_error).__name__}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise

        if self._hybrid_mode:
            logger.info(
                f"Whisper model ready (encoder: {self.encoder_device}, decoder: {self.decoder_device})"
            )
        else:
            logger.info(f"Whisper model ready on {self.device}")

    def _compile_submodels(self, ov_config: dict) -> None:
        """Compile encoder and decoder on their respective devices."""
        import openvino as ov

        core = ov.Core()

        # Compile encoder on encoder_device
        if hasattr(self.model, 'encoder') and self.model.encoder is not None:
            logger.info(f"Compiling ENCODER on {self.encoder_device}...")
            try:
                self._compile_submodel_to_device(
                    self.model.encoder, self.encoder_device, core, ov_config, "encoder"
                )
                logger.info("ENCODER compiled successfully!")
            except Exception as e:
                logger.error(f"ENCODER compilation FAILED on {self.encoder_device}: {e}")
                raise RuntimeError(f"Encoder compilation failed on {self.encoder_device}: {e}") from e

        # Compile decoder on decoder_device
        if hasattr(self.model, 'decoder') and self.model.decoder is not None:
            self._compile_decoder_with_reshape(core, ov_config)

        # Compile decoder_with_past on decoder_device (if exists)
        if hasattr(self.model, 'decoder_with_past') and self.model.decoder_with_past is not None:
            self._compile_decoder_with_past_with_reshape(core, ov_config)

        logger.info("All submodels compiled successfully")

    def _compile_submodel_to_device(
        self,
        submodel,
        device: str,
        core: "ov.Core",
        ov_config: dict,
        name: str,
    ) -> None:
        """Compile a submodel to a specific device using OpenVINO Core."""
        import openvino as ov

        # Get the underlying OV model
        if hasattr(submodel, 'model') and submodel.model is not None:
            ov_model = submodel.model

            # Analyze model
            self._analyze_model_for_npu(ov_model, name)

            # Compile using Core directly
            logger.info(f"Compiling {name} to {device} using OpenVINO Core...")

            # Build config for compilation
            compile_config = {}
            for key, value in ov_config.items():
                if key != "CACHE_DIR":  # Skip cache dir for now
                    compile_config[key] = value

            # Compile
            compiled = core.compile_model(ov_model, device, compile_config)

            # Use wrapper that provides both callable and async interfaces
            submodel.request = CompiledModelWrapper(compiled)

            logger.info(f"{name} compiled to {device}")
        else:
            # Fallback to default compilation
            logger.info(f"Using default compilation for {name}")
            submodel._compile()

    def _analyze_model_for_npu(self, model: "ov.Model", name: str) -> Dict[str, int]:
        """
        Analyze model for NPU compatibility issues.

        Returns:
            Dictionary with analysis results including stateful op counts
        """
        results = {"has_dynamic": False, "read_values": 0, "assigns": 0}

        # Check for dynamic shapes
        num_inputs = len(model.inputs)
        num_outputs = len(model.outputs)
        dynamic_inputs = 0

        for inp in model.inputs:
            if inp.get_partial_shape().is_dynamic:
                results["has_dynamic"] = True
                dynamic_inputs += 1

        for out in model.outputs:
            if out.get_partial_shape().is_dynamic:
                results["has_dynamic"] = True

        # Count stateful operations
        for op in model.get_ordered_ops():
            op_type = op.get_type_name()
            if op_type == "ReadValue":
                results["read_values"] += 1
            elif op_type == "Assign":
                results["assigns"] += 1

        # Log summary (not every input/output)
        total_stateful = results["read_values"] + results["assigns"]
        if total_stateful > 0:
            logger.warning(
                f"  {name}: {num_inputs} inputs, {num_outputs} outputs, "
                f"{dynamic_inputs} dynamic, {total_stateful} stateful ops (ReadValue/Assign)"
            )
        else:
            logger.info(
                f"  {name}: {num_inputs} inputs, {num_outputs} outputs, "
                f"{dynamic_inputs} dynamic, no stateful ops"
            )

        return results

    def _reshape_decoder_to_static(
        self,
        model: "ov.Model",
        batch_size: int = 1,
        seq_len: int = 448,
        encoder_seq_len: int = 1500,
    ) -> "ov.Model":
        """Reshape decoder model to static shapes for NPU compatibility."""
        logger.info(
            f"Reshaping to static: batch={batch_size}, seq={seq_len}, encoder_seq={encoder_seq_len}"
        )

        # Build static shapes for each input
        new_shapes = {}
        for inp in model.inputs:
            name = inp.get_any_name()
            current_shape = inp.get_partial_shape()

            # Determine static shape based on input name
            if "input_ids" in name.lower() or "decoder_input" in name.lower():
                new_shapes[name] = [batch_size, seq_len]
            elif "encoder_hidden" in name.lower() or "encoder_output" in name.lower():
                # Encoder output: [batch, encoder_seq, hidden_dim]
                hidden_dim = current_shape[-1]
                if hidden_dim.is_dynamic:
                    hidden_dim = 1280  # Whisper large hidden dim
                else:
                    hidden_dim = hidden_dim.get_length()
                new_shapes[name] = [batch_size, encoder_seq_len, hidden_dim]
            elif "beam_idx" in name.lower():
                # Beam index for beam search - [batch_size] for greedy, [batch*num_beams] for beam
                new_shapes[name] = [batch_size]
            elif "attention_mask" in name.lower():
                if "encoder" in name.lower():
                    new_shapes[name] = [batch_size, encoder_seq_len]
                else:
                    new_shapes[name] = [batch_size, seq_len]
            elif "past_key_value" in name.lower() or "cache" in name.lower():
                # KV cache - keep dynamic or set reasonable static size
                # Shape is usually [batch, num_heads, seq, head_dim]
                if len(current_shape) == 4:
                    num_heads = current_shape[1]
                    head_dim = current_shape[3]
                    if num_heads.is_dynamic:
                        num_heads = 20  # Whisper large
                    else:
                        num_heads = num_heads.get_length()
                    if head_dim.is_dynamic:
                        head_dim = 64
                    else:
                        head_dim = head_dim.get_length()
                    new_shapes[name] = [batch_size, num_heads, seq_len, head_dim]

        if new_shapes:
            try:
                model.reshape(new_shapes)
                logger.info(f"Reshaped {len(new_shapes)} inputs to static shapes")
            except Exception as e:
                logger.warning(f"Failed to reshape decoder: {e}")

        return model

    def _compile_decoder_with_reshape(self, core: "ov.Core", ov_config: dict) -> None:
        """Compile decoder with optional static shape reshape for NPU."""
        import openvino as ov

        decoder = self.model.decoder
        device = self.decoder_device

        logger.info(f"Compiling DECODER on {device}...")

        # Get the underlying OV model
        if hasattr(decoder, 'model') and decoder.model is not None:
            ov_model = decoder.model
            analysis = self._analyze_model_for_npu(ov_model, "decoder")

            # Check for stateful ops on non-CPU devices
            total_stateful = analysis.get("read_values", 0) + analysis.get("assigns", 0)
            if total_stateful > 0 and device != "CPU":
                logger.warning(
                    f"Decoder has {total_stateful} stateful ops - {device} may not support them. "
                    f"Use: mlperf-ov export-whisper-npu --stateless"
                )

            # Check if shapes are dynamic
            has_dynamic = analysis.get("has_dynamic", False)

            if has_dynamic and device != "CPU":
                logger.info("Decoder has dynamic shapes, attempting static reshape for NPU...")
                try:
                    self._reshape_decoder_to_static(ov_model)
                except Exception as e:
                    logger.warning(f"Static reshape failed: {e}")

            # Build config for compilation
            compile_config = {}
            for key, value in ov_config.items():
                if key != "CACHE_DIR":
                    compile_config[key] = value

            # Compile using Core directly to specified device
            try:
                compiled = core.compile_model(ov_model, device, compile_config)
                # Use wrapper that provides both callable and async interfaces
                decoder.request = CompiledModelWrapper(compiled)
                logger.info("DECODER compiled successfully!")
                return
            except Exception as e:
                self._log_compilation_error(e, "decoder", device, has_stateful=(total_stateful > 0))
                raise RuntimeError(f"Decoder compilation failed on {device}: {e}") from e
        else:
            # No model attribute, try default compilation
            try:
                decoder._compile()
                logger.info("DECODER compiled successfully!")
            except Exception as e:
                self._log_compilation_error(e, "decoder", device)
                raise RuntimeError(f"Decoder compilation failed on {device}: {e}") from e

    def _compile_decoder_with_past_with_reshape(self, core: "ov.Core", ov_config: dict) -> None:
        """Compile decoder_with_past with optional static shape reshape for NPU."""
        import openvino as ov

        decoder = self.model.decoder_with_past
        device = self.decoder_device

        logger.info(f"Compiling DECODER_WITH_PAST on {device}...")

        # Get the underlying OV model
        if hasattr(decoder, 'model') and decoder.model is not None:
            ov_model = decoder.model
            analysis = self._analyze_model_for_npu(ov_model, "decoder_with_past")

            # Check for stateful ops on non-CPU devices
            total_stateful = analysis.get("read_values", 0) + analysis.get("assigns", 0)
            if total_stateful > 0 and device != "CPU":
                logger.warning(
                    f"Decoder_with_past has {total_stateful} stateful ops - {device} may not support them"
                )

            # Check if shapes are dynamic
            has_dynamic = analysis.get("has_dynamic", False)

            if has_dynamic and device != "CPU":
                logger.info("Decoder_with_past has dynamic shapes, attempting static reshape...")
                try:
                    # For decoder_with_past, seq_len is usually 1 (single token at a time)
                    self._reshape_decoder_to_static(ov_model, seq_len=1)
                except Exception as e:
                    logger.warning(f"Static reshape failed: {e}")

            # Build config for compilation
            compile_config = {}
            for key, value in ov_config.items():
                if key != "CACHE_DIR":
                    compile_config[key] = value

            # Compile using Core directly to specified device
            try:
                compiled = core.compile_model(ov_model, device, compile_config)
                # Use wrapper that provides both callable and async interfaces
                decoder.request = CompiledModelWrapper(compiled)
                logger.info("DECODER_WITH_PAST compiled successfully!")
                return
            except Exception as e:
                self._log_compilation_error(e, "decoder_with_past", device, has_stateful=(total_stateful > 0))
                raise RuntimeError(f"Decoder_with_past compilation failed on {device}: {e}") from e
        else:
            # No model attribute, try default compilation
            try:
                decoder._compile()
                logger.info("DECODER_WITH_PAST compiled successfully!")
            except Exception as e:
                self._log_compilation_error(e, "decoder_with_past", device)
                raise RuntimeError(f"Decoder_with_past compilation failed on {device}: {e}") from e

    def _log_compilation_error(
        self,
        error: Exception,
        model_name: str,
        device: str,
        has_stateful: bool = False,
    ) -> None:
        """Log compilation error with diagnosis."""
        error_str = str(error).lower()
        logger.error(f"{model_name} compilation FAILED on {device}: {error}")

        if has_stateful:
            logger.error("Cause: Stateful ops (ReadValue/Assign). Fix: mlperf-ov export-whisper-npu --stateless")
        elif "not supported" in error_str or "unsupported" in error_str:
            logger.error("Cause: Unsupported ops. Fix: Try --stateless export")
        elif "dynamic" in error_str or "shape" in error_str:
            logger.error("Cause: Dynamic shapes. Fix: Static shape export")
        elif "memory" in error_str:
            logger.error("Cause: Out of memory. Fix: Reduce batch size")

    def _start_progress(self, total: int, desc: str = "Processing") -> None:
        """Start progress tracking."""
        self._start_time = time.time()
        if TQDM_AVAILABLE:
            self._progress_bar = tqdm(
                total=total,
                desc=desc,
                unit="samples",
                file=sys.stderr,
                dynamic_ncols=True,
            )
        else:
            logger.info(f"Starting: {desc} ({total} samples)")
            self._last_progress_update = time.time()

    def _update_progress(self, n: int = 1) -> None:
        """Update progress by n samples."""
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.update(n)
        else:
            current_time = time.time()
            if current_time - self._last_progress_update >= self._progress_update_interval:
                elapsed = current_time - self._start_time
                throughput = self._sample_count / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {self._sample_count} samples, {throughput:.1f} samples/sec")
                self._last_progress_update = current_time

    def _close_progress(self) -> None:
        """Close progress tracking."""
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
        else:
            elapsed = time.time() - self._start_time
            throughput = self._sample_count / elapsed if elapsed > 0 else 0
            logger.info(f"Completed: {self._sample_count} samples in {elapsed:.1f}s ({throughput:.1f} samples/sec)")

    def flush_queries(self) -> None:
        """Flush any pending queries."""
        if self._progress_bar is not None:
            self._close_progress()

    def _process_sample(self, sample_idx: int) -> str:
        """
        Process a single audio sample.

        Args:
            sample_idx: Sample index

        Returns:
            Transcribed text
        """
        import torch

        # Get preprocessed mel features from QSL
        features = self.qsl.get_features(sample_idx)
        input_features = features["input_features"]

        # Convert to tensor
        if isinstance(input_features, np.ndarray):
            input_features = torch.from_numpy(input_features)

        # Ensure correct shape (batch, n_mels, time)
        if input_features.dim() == 2:
            input_features = input_features.unsqueeze(0)

        # Generate transcription using model with KV-cache support
        generated_ids = self.model.generate(
            input_features,
            max_new_tokens=self.max_new_tokens,
            language="en",
            task="transcribe",
        )

        # Decode tokens to text
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return text

    def issue_queries(self, query_samples: List[Any]) -> None:
        """Process queries from LoadGen."""
        self._query_count += len(query_samples)

        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        elif self.scenario == Scenario.SERVER:
            self._issue_query_server(query_samples)
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")

    def _issue_query_offline(self, query_samples: List[Any]) -> None:
        """Process queries for Offline scenario."""
        responses = []
        response_arrays = []

        total_samples = len(query_samples)
        self._start_progress(total_samples, desc="Whisper Offline inference")

        for sample in query_samples:
            sample_idx = sample.index
            self._sample_count += 1

            text = self._process_sample(sample_idx)
            self._predictions[sample_idx] = text

            response_data = np.array([len(text)], dtype=np.int64)
            response_array = array.array('B', response_data.tobytes())
            response_arrays.append(response_array)
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(sample.id, bi[0], bi[1])
            responses.append(response)

            self._update_progress(1)

        self._close_progress()
        lg.QuerySamplesComplete(responses)

    def _issue_query_server(self, query_samples: List[Any]) -> None:
        """Process queries for Server scenario."""
        responses = []
        response_arrays = []

        if self._sample_count == 0:
            self._start_progress(0, desc="Whisper Server inference")

        for sample in query_samples:
            sample_idx = sample.index
            self._sample_count += 1

            text = self._process_sample(sample_idx)
            self._predictions[sample_idx] = text

            response_data = np.array([len(text)], dtype=np.int64)
            response_array = array.array('B', response_data.tobytes())
            response_arrays.append(response_array)
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(sample.id, bi[0], bi[1])
            responses.append(response)

            self._update_progress(1)

        lg.QuerySamplesComplete(responses)

    def get_sut(self) -> Any:
        """Get LoadGen SUT handle.

        Returns:
            LoadGen SUT handle for benchmark execution.
        """
        if self._sut_handle is None:
            self._sut_handle = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self._sut_handle

    def get_qsl(self) -> Any:
        """Get LoadGen QSL handle.

        Returns:
            LoadGen QSL handle for sample management.
        """
        if self._qsl_handle is None:
            self._qsl_handle = lg.ConstructQSL(
                self.qsl.total_sample_count,
                self.qsl.performance_sample_count,
                self.qsl.load_query_samples,
                self.qsl.unload_query_samples
            )
        return self._qsl_handle

    def get_predictions(self) -> Dict[int, str]:
        """Get all predictions."""
        return self._predictions.copy()

    def reset(self) -> None:
        """Reset state for new run."""
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0


class WhisperSUT:
    """
    System Under Test for Whisper ASR model (fallback implementation).

    Uses manual encoder-decoder inference when optimum-intel is not available.
    Prefer WhisperOptimumSUT when possible.
    """

    # Whisper special tokens
    SOT_TOKEN = 50258  # Start of transcript
    EOT_TOKEN = 50257  # End of transcript
    TRANSCRIBE_TOKEN = 50359  # Transcribe task
    NO_TIMESTAMPS_TOKEN = 50363  # No timestamps
    EN_TOKEN = 50259  # English language

    def __init__(
        self,
        config: BenchmarkConfig,
        encoder_backend: BaseBackend,
        decoder_backend: BaseBackend,
        qsl: LibriSpeechQSL,
        scenario: Scenario = Scenario.OFFLINE,
        max_new_tokens: int = 440,  # Leave room for special tokens (448 - 8)
    ):
        """
        Initialize Whisper SUT.

        Args:
            config: Benchmark configuration
            encoder_backend: OpenVINO backend for encoder
            decoder_backend: OpenVINO backend for decoder
            qsl: Query Sample Library
            scenario: MLPerf scenario
            max_new_tokens: Maximum tokens to generate
        """
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")

        self.config = config
        self.encoder = encoder_backend
        self.decoder = decoder_backend
        self.qsl = qsl
        self.scenario = scenario
        self.max_new_tokens = max_new_tokens

        # Discover decoder input names
        self._decoder_input_names = self._discover_decoder_inputs()

        # Results storage
        self._predictions: Dict[int, str] = {}
        self._query_count = 0
        self._sample_count = 0

        # Progress tracking
        self._progress_bar: Optional[Any] = None
        self._start_time = 0.0
        self._last_progress_update = 0.0
        self._progress_update_interval = 0.5  # seconds

        # Create LoadGen handles
        self._sut_handle = None
        self._qsl_handle = None

        # Tokenizer for decoding (lazy loaded)
        self._tokenizer = None
    
    def _start_progress(self, total: int, desc: str = "Processing") -> None:
        """Start progress tracking."""
        self._start_time = time.time()
        if TQDM_AVAILABLE:
            self._progress_bar = tqdm(
                total=total,
                desc=desc,
                unit="samples",
                file=sys.stderr,
                dynamic_ncols=True,
            )
        else:
            logger.info(f"Starting: {desc} ({total} samples)")
            self._last_progress_update = time.time()

    def _update_progress(self, n: int = 1) -> None:
        """Update progress by n samples."""
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.update(n)
        else:
            # Simple text-based progress update
            current_time = time.time()
            if current_time - self._last_progress_update >= self._progress_update_interval:
                elapsed = current_time - self._start_time
                throughput = self._sample_count / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {self._sample_count} samples, {throughput:.1f} samples/sec")
                self._last_progress_update = current_time

    def _close_progress(self) -> None:
        """Close progress tracking."""
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
        else:
            elapsed = time.time() - self._start_time
            throughput = self._sample_count / elapsed if elapsed > 0 else 0
            logger.info(f"Completed: {self._sample_count} samples in {elapsed:.1f}s ({throughput:.1f} samples/sec)")

    def flush_queries(self) -> None:
        """Flush any pending queries."""
        # Close progress bar if still open (for Server mode)
        if self._progress_bar is not None:
            self._close_progress()

    def _discover_decoder_inputs(self) -> Dict[str, str]:
        """
        Discover decoder input names from the model.

        Returns:
            Dictionary mapping semantic names to actual input names:
            - 'input_ids': name for decoder input IDs
            - 'encoder_hidden_states': name for encoder output
            - 'attention_mask': name for attention mask (optional)
        """
        input_names = self.decoder.input_names
        result = {}

        # Find input_ids (decoder_input_ids or input_ids)
        for name in input_names:
            name_lower = name.lower()
            if 'input_id' in name_lower or 'decoder_input' in name_lower:
                result['input_ids'] = name
            elif 'encoder_hidden' in name_lower or 'encoder_output' in name_lower:
                result['encoder_hidden_states'] = name
            elif 'attention_mask' in name_lower and 'encoder' not in name_lower:
                result['attention_mask'] = name
            elif 'encoder_attention_mask' in name_lower:
                result['encoder_attention_mask'] = name

        # Fallback if not found
        if 'input_ids' not in result:
            # Try first input that looks like IDs
            for name in input_names:
                if 'id' in name.lower():
                    result['input_ids'] = name
                    break

        if 'encoder_hidden_states' not in result:
            # Try to find encoder output
            for name in input_names:
                if 'encoder' in name.lower() and 'mask' not in name.lower():
                    result['encoder_hidden_states'] = name
                    break

        return result

    def _load_tokenizer(self):
        """Load Whisper tokenizer."""
        if self._tokenizer is not None:
            return

        try:
            from transformers import WhisperTokenizer
            self._tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
            logger.info("Loaded Whisper tokenizer")
        except ImportError:
            logger.warning(
                "transformers not installed, using basic token decoding. "
                "Install with: pip install transformers"
            )
            self._tokenizer = None
    
    def _decode_tokens(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        self._load_tokenizer()
        
        if self._tokenizer is not None:
            # Filter special tokens
            filtered = [t for t in token_ids if t < 50257]
            return self._tokenizer.decode(filtered, skip_special_tokens=True)
        else:
            # Basic fallback - just return token IDs as string
            return f"[tokens: {len(token_ids)}]"
    
    def _encode(self, mel_features: np.ndarray) -> np.ndarray:
        """
        Run encoder on mel spectrogram.
        
        Args:
            mel_features: Mel spectrogram of shape (batch, n_mels, time)
            
        Returns:
            Encoder hidden states
        """
        inputs = {self.encoder.input_names[0]: mel_features}
        outputs = self.encoder.predict(inputs)
        return list(outputs.values())[0]
    
    def _decode_step(
        self,
        encoder_hidden_states: np.ndarray,
        decoder_input_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Run one decoder step.

        Args:
            encoder_hidden_states: Encoder output
            decoder_input_ids: Current token sequence

        Returns:
            Logits for next token
        """
        inputs = {}

        # Use discovered input names
        if 'input_ids' in self._decoder_input_names:
            inputs[self._decoder_input_names['input_ids']] = decoder_input_ids
        else:
            inputs['decoder_input_ids'] = decoder_input_ids

        if 'encoder_hidden_states' in self._decoder_input_names:
            inputs[self._decoder_input_names['encoder_hidden_states']] = encoder_hidden_states
        else:
            inputs['encoder_hidden_states'] = encoder_hidden_states

        # Add attention masks if required by model
        if 'attention_mask' in self._decoder_input_names:
            # Create attention mask (all ones for valid tokens)
            attn_mask = np.ones(decoder_input_ids.shape, dtype=np.int64)
            inputs[self._decoder_input_names['attention_mask']] = attn_mask

        if 'encoder_attention_mask' in self._decoder_input_names:
            # Create encoder attention mask
            batch_size = encoder_hidden_states.shape[0]
            seq_len = encoder_hidden_states.shape[1]
            enc_attn_mask = np.ones((batch_size, seq_len), dtype=np.int64)
            inputs[self._decoder_input_names['encoder_attention_mask']] = enc_attn_mask

        outputs = self.decoder.predict(inputs)
        return list(outputs.values())[0]
    
    def _generate(
        self,
        mel_features: np.ndarray,
        temperature: float = 0.0,
    ) -> Tuple[List[int], str]:
        """
        Generate transcript from mel spectrogram.

        Args:
            mel_features: Mel spectrogram
            temperature: Sampling temperature (0 = greedy)

        Returns:
            Tuple of (token_ids, decoded_text)
        """
        # Encode audio
        encoder_hidden_states = self._encode(mel_features)

        # Initialize decoder input with special tokens
        # [SOT, language, task, no_timestamps]
        decoder_input = [
            self.SOT_TOKEN,
            self.EN_TOKEN,
            self.TRANSCRIBE_TOKEN,
            self.NO_TIMESTAMPS_TOKEN,
        ]

        generated_tokens = []

        for step in range(self.max_new_tokens):
            # Prepare decoder input
            decoder_input_ids = np.array([decoder_input], dtype=np.int64)

            # Get logits
            logits = self._decode_step(encoder_hidden_states, decoder_input_ids)

            # Get next token (greedy or sampling)
            # Logits shape should be (batch, seq_len, vocab_size)
            if logits.ndim == 3:
                next_token_logits = logits[0, -1, :]
            elif logits.ndim == 2:
                # Shape might be (batch, vocab_size) for last token only
                next_token_logits = logits[0, :]
            else:
                break

            if temperature == 0.0:
                next_token = int(np.argmax(next_token_logits))
            else:
                # Softmax with temperature
                probs = np.exp(next_token_logits / temperature)
                probs = probs / probs.sum()
                next_token = int(np.random.choice(len(probs), p=probs))

            # Check for end of transcript
            if next_token == self.EOT_TOKEN:
                break

            generated_tokens.append(next_token)
            decoder_input.append(next_token)

        # Decode tokens to text
        text = self._decode_tokens(generated_tokens)

        return generated_tokens, text
    
    def _process_sample(self, sample_idx: int) -> str:
        """
        Process a single audio sample.

        Args:
            sample_idx: Sample index

        Returns:
            Transcribed text
        """
        features = self.qsl.get_features(sample_idx)
        mel_features = features["input_features"]
        tokens, text = self._generate(mel_features)
        return text
    
    def issue_queries(self, query_samples: List[Any]) -> None:
        """
        Process queries from LoadGen.
        
        Args:
            query_samples: List of QuerySample objects
        """
        self._query_count += len(query_samples)
        
        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        elif self.scenario == Scenario.SERVER:
            self._issue_query_server(query_samples)
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")
    
    def _issue_query_offline(self, query_samples: List[Any]) -> None:
        """Process queries for Offline scenario."""
        responses = []
        response_arrays = []  # Keep arrays alive until QuerySamplesComplete!

        # Start progress tracking
        total_samples = len(query_samples)
        self._start_progress(total_samples, desc="Whisper Offline inference")

        for sample in query_samples:
            sample_idx = sample.index
            self._sample_count += 1

            # Process sample
            text = self._process_sample(sample_idx)
            self._predictions[sample_idx] = text

            # Create response (using dummy data for LoadGen)
            response_data = np.array([len(text)], dtype=np.int64)
            response_array = array.array('B', response_data.tobytes())
            response_arrays.append(response_array)  # Keep alive!
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(
                sample.id,
                bi[0],
                bi[1]
            )
            responses.append(response)

            # Update progress
            self._update_progress(1)

        # Close progress
        self._close_progress()

        lg.QuerySamplesComplete(responses)
    
    def _issue_query_server(self, query_samples: List[Any]) -> None:
        """Process queries for Server scenario."""
        responses = []
        response_arrays = []  # Keep arrays alive until QuerySamplesComplete!

        # Start progress tracking if first query
        if self._sample_count == 0:
            self._start_progress(0, desc="Whisper Server inference")

        for sample in query_samples:
            sample_idx = sample.index
            self._sample_count += 1

            # Process sample
            text = self._process_sample(sample_idx)
            self._predictions[sample_idx] = text

            # Create response
            response_data = np.array([len(text)], dtype=np.int64)
            response_array = array.array('B', response_data.tobytes())
            response_arrays.append(response_array)  # Keep alive!
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(
                sample.id,
                bi[0],
                bi[1]
            )
            responses.append(response)

            # Update progress
            self._update_progress(1)

        lg.QuerySamplesComplete(responses)
    
    def get_sut(self) -> Any:
        """Get LoadGen SUT handle.

        Returns:
            LoadGen SUT handle for benchmark execution.
        """
        if self._sut_handle is None:
            self._sut_handle = lg.ConstructSUT(
                self.issue_queries,
                self.flush_queries
            )
        return self._sut_handle

    def get_qsl(self) -> Any:
        """Get LoadGen QSL handle.

        Returns:
            LoadGen QSL handle for sample management.
        """
        if self._qsl_handle is None:
            self._qsl_handle = lg.ConstructQSL(
                self.qsl.total_sample_count,
                self.qsl.performance_sample_count,
                self.qsl.load_query_samples,
                self.qsl.unload_query_samples
            )
        return self._qsl_handle

    def get_predictions(self) -> Dict[int, str]:
        """Get all predictions."""
        return self._predictions.copy()

    def reset(self) -> None:
        """Reset state for new run."""
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0


class WhisperEncoderOnlySUT:
    """
    Simplified SUT that only runs the Whisper encoder.
    
    Useful for benchmarking encoder performance separately,
    or when using external decoder/beam search.
    """
    
    def __init__(
        self,
        config: BenchmarkConfig,
        backend: BaseBackend,
        qsl: LibriSpeechQSL,
        scenario: Scenario = Scenario.OFFLINE,
    ):
        """
        Initialize encoder-only SUT.
        
        Args:
            config: Benchmark configuration
            backend: OpenVINO backend for encoder
            qsl: Query Sample Library
            scenario: MLPerf scenario
        """
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")
        
        self.config = config
        self.backend = backend
        self.qsl = qsl
        self.scenario = scenario
        
        self._predictions: Dict[int, np.ndarray] = {}
        self._query_count = 0
        self._sample_count = 0

        # Progress tracking
        self._progress_bar: Optional[Any] = None
        self._start_time = 0.0
        self._last_progress_update = 0.0
        self._progress_update_interval = 0.5  # seconds

        # Create LoadGen handles
        self._sut_handle = None
        self._qsl_handle = None

    def _start_progress(self, total: int, desc: str = "Processing") -> None:
        """Start progress tracking."""
        self._start_time = time.time()
        if TQDM_AVAILABLE:
            self._progress_bar = tqdm(
                total=total,
                desc=desc,
                unit="samples",
                file=sys.stderr,
                dynamic_ncols=True,
            )
        else:
            logger.info(f"Starting: {desc} ({total} samples)")
            self._last_progress_update = time.time()

    def _update_progress(self, n: int = 1) -> None:
        """Update progress by n samples."""
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.update(n)
        else:
            # Simple text-based progress update
            current_time = time.time()
            if current_time - self._last_progress_update >= self._progress_update_interval:
                elapsed = current_time - self._start_time
                throughput = self._sample_count / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {self._sample_count} samples, {throughput:.1f} samples/sec")
                self._last_progress_update = current_time

    def _close_progress(self) -> None:
        """Close progress tracking."""
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
        else:
            elapsed = time.time() - self._start_time
            throughput = self._sample_count / elapsed if elapsed > 0 else 0
            logger.info(f"Completed: {self._sample_count} samples in {elapsed:.1f}s ({throughput:.1f} samples/sec)")

    def flush_queries(self) -> None:
        """Flush any pending queries."""
        # Close progress bar if still open
        if self._progress_bar is not None:
            self._close_progress()

    def issue_queries(self, query_samples: List[Any]) -> None:
        """Process queries from LoadGen."""
        self._query_count += len(query_samples)

        # Start progress tracking
        total_samples = len(query_samples)
        self._start_progress(total_samples, desc="Whisper encoder inference")

        responses = []
        response_arrays = []  # Keep arrays alive until QuerySamplesComplete!

        for sample in query_samples:
            sample_idx = sample.index
            self._sample_count += 1

            # Get input features
            features = self.qsl.get_features(sample_idx)
            mel_features = features["input_features"]

            # Run encoder
            inputs = {self.backend.input_names[0]: mel_features}
            outputs = self.backend.predict(inputs)
            encoder_output = list(outputs.values())[0]

            self._predictions[sample_idx] = encoder_output

            # Create response - use array.array for safe memory handling
            response_array = array.array('B', encoder_output.tobytes())
            response_arrays.append(response_array)  # Keep alive!
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(
                sample.id,
                bi[0],
                bi[1]
            )
            responses.append(response)

            # Update progress
            self._update_progress(1)

        # Close progress
        self._close_progress()

        lg.QuerySamplesComplete(responses)

    def get_sut(self) -> Any:
        """Get LoadGen SUT handle.

        Returns:
            LoadGen SUT handle for benchmark execution.
        """
        if self._sut_handle is None:
            self._sut_handle = lg.ConstructSUT(
                self.issue_queries,
                self.flush_queries
            )
        return self._sut_handle

    def get_qsl(self) -> Any:
        """Get LoadGen QSL handle.

        Returns:
            LoadGen QSL handle for sample management.
        """
        if self._qsl_handle is None:
            self._qsl_handle = lg.ConstructQSL(
                self.qsl.total_sample_count,
                self.qsl.performance_sample_count,
                self.qsl.load_query_samples,
                self.qsl.unload_query_samples
            )
        return self._qsl_handle

    def get_predictions(self) -> Dict[int, np.ndarray]:
        """Get all encoder outputs."""
        return self._predictions.copy()

    def reset(self) -> None:
        """Reset state for new run."""
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0


class WhisperMultiDieSUT:
    """
    System Under Test for Whisper ASR on multi-die NPU accelerators.

    Distributes inference across multiple NPU dies for maximum throughput.
    Uses round-robin distribution of samples across dies.

    Each die runs a separate WhisperOptimumSUT instance.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        model_path: Union[str, Path],
        qsl: LibriSpeechQSL,
        scenario: Scenario = Scenario.OFFLINE,
        max_new_tokens: int = 440,
        target_devices: Optional[List[str]] = None,
    ):
        """
        Initialize Whisper Multi-Die SUT.

        Args:
            config: Benchmark configuration
            model_path: Path to OpenVINO Whisper model directory
            qsl: Query Sample Library
            scenario: MLPerf scenario
            max_new_tokens: Maximum tokens to generate
            target_devices: List of target devices (e.g., ['NPU.0', 'NPU.1'])
        """
        if not LOADGEN_AVAILABLE:
            raise ImportError("MLPerf LoadGen is not installed")

        if not OPTIMUM_AVAILABLE:
            raise ImportError(
                "Optimum-Intel is required for Whisper inference. "
                "Install with: pip install optimum[openvino]"
            )

        self.config = config
        self.model_path = Path(model_path)
        self.qsl = qsl
        self.scenario = scenario
        self.max_new_tokens = max_new_tokens

        # Discover or use provided devices
        if target_devices:
            self._active_devices = target_devices
        else:
            self._active_devices = self._discover_devices()

        if not self._active_devices:
            raise RuntimeError("No NPU devices available for multi-die inference")

        logger.info(f"WhisperMultiDieSUT: using devices {self._active_devices}")

        # Create SUT instance for each die
        self._die_suts: Dict[str, WhisperOptimumSUT] = {}
        self._load_models()

        # Results storage
        self._predictions: Dict[int, str] = {}
        self._query_count = 0
        self._sample_count = 0

        # Round-robin counter
        self._die_index = 0

        # Progress tracking
        self._progress_bar: Optional[Any] = None
        self._start_time = 0.0
        self._last_progress_update = 0.0
        self._progress_update_interval = 0.5

        # LoadGen handles
        self._sut_handle = None
        self._qsl_handle = None

    def _discover_devices(self) -> List[str]:
        """Discover available NPU devices."""
        try:
            from openvino import Core
            from ..backends.device_discovery import discover_accelerator_devices

            core = Core()
            device_prefix = self.config.openvino.get_device_prefix()
            devices = discover_accelerator_devices(core, device_prefix)
            return devices
        except Exception as e:
            logger.warning(f"Failed to discover devices: {e}")
            return []

    def _load_models(self) -> None:
        """Load Whisper model on each die."""
        from transformers import AutoProcessor

        logger.info(f"Loading Whisper models on {len(self._active_devices)} dies...")

        # Load processor once (shared)
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        except Exception:
            self.processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

        # Create SUT for each die
        for device in self._active_devices:
            logger.info(f"Loading model on {device}...")

            sut = WhisperOptimumSUT(
                config=self.config,
                model_path=self.model_path,
                qsl=self.qsl,
                scenario=self.scenario,
                max_new_tokens=self.max_new_tokens,
                device=device,
            )
            self._die_suts[device] = sut

        logger.info(f"All {len(self._active_devices)} dies loaded successfully")

    @property
    def num_dies(self) -> int:
        """Number of active dies."""
        return len(self._active_devices)

    def _get_next_die(self) -> str:
        """Get next die for round-robin distribution."""
        die = self._active_devices[self._die_index % len(self._active_devices)]
        self._die_index += 1
        return die

    def _start_progress(self, total: int, desc: str = "Processing") -> None:
        """Start progress tracking."""
        self._start_time = time.time()
        if TQDM_AVAILABLE:
            self._progress_bar = tqdm(
                total=total,
                desc=desc,
                unit="samples",
                file=sys.stderr,
                dynamic_ncols=True,
            )
        else:
            logger.info(f"Starting: {desc} ({total} samples)")
            self._last_progress_update = time.time()

    def _update_progress(self, n: int = 1) -> None:
        """Update progress by n samples."""
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.update(n)
        else:
            current_time = time.time()
            if current_time - self._last_progress_update >= self._progress_update_interval:
                elapsed = current_time - self._start_time
                throughput = self._sample_count / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Progress: {self._sample_count} samples, "
                    f"{throughput:.1f} samples/sec ({self.num_dies} dies)"
                )
                self._last_progress_update = current_time

    def _close_progress(self) -> None:
        """Close progress tracking."""
        if TQDM_AVAILABLE and self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
        else:
            elapsed = time.time() - self._start_time
            throughput = self._sample_count / elapsed if elapsed > 0 else 0
            logger.info(
                f"Completed: {self._sample_count} samples in {elapsed:.1f}s "
                f"({throughput:.1f} samples/sec, {self.num_dies} dies)"
            )

    def _process_sample(self, sample_idx: int, die_name: str) -> str:
        """Process a single sample on specified die."""
        sut = self._die_suts[die_name]
        return sut._process_sample(sample_idx)

    def issue_queries(self, query_samples: List[Any]) -> None:
        """Process queries from LoadGen."""
        self._query_count += len(query_samples)

        if self.scenario == Scenario.OFFLINE:
            self._issue_query_offline(query_samples)
        elif self.scenario == Scenario.SERVER:
            self._issue_query_server(query_samples)
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")

    def _issue_query_offline(self, query_samples: List[Any]) -> None:
        """Process queries for Offline scenario with round-robin distribution."""
        responses = []
        response_arrays = []

        total_samples = len(query_samples)
        self._start_progress(total_samples, f"Whisper Offline ({self.num_dies} dies)")

        # Process samples with round-robin distribution across dies
        for sample in query_samples:
            sample_idx = sample.index
            self._sample_count += 1

            # Select die for this sample
            die_name = self._get_next_die()

            # Process on selected die
            text = self._process_sample(sample_idx, die_name)
            self._predictions[sample_idx] = text

            # Create response
            response_data = np.array([len(text)], dtype=np.int64)
            response_array = array.array('B', response_data.tobytes())
            response_arrays.append(response_array)
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(sample.id, bi[0], bi[1])
            responses.append(response)

            self._update_progress(1)

        self._close_progress()
        lg.QuerySamplesComplete(responses)

    def _issue_query_server(self, query_samples: List[Any]) -> None:
        """Process queries for Server scenario."""
        responses = []
        response_arrays = []

        if self._sample_count == 0:
            self._start_progress(0, f"Whisper Server ({self.num_dies} dies)")

        for sample in query_samples:
            sample_idx = sample.index
            self._sample_count += 1

            # Select die for this sample
            die_name = self._get_next_die()

            # Process on selected die
            text = self._process_sample(sample_idx, die_name)
            self._predictions[sample_idx] = text

            # Create response
            response_data = np.array([len(text)], dtype=np.int64)
            response_array = array.array('B', response_data.tobytes())
            response_arrays.append(response_array)
            bi = response_array.buffer_info()

            response = lg.QuerySampleResponse(sample.id, bi[0], bi[1])
            responses.append(response)

            self._update_progress(1)

        lg.QuerySamplesComplete(responses)

    def flush_queries(self) -> None:
        """Flush any pending queries."""
        if self._progress_bar is not None:
            self._close_progress()

    def get_sut(self) -> Any:
        """Get LoadGen SUT handle."""
        if self._sut_handle is None:
            self._sut_handle = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self._sut_handle

    def get_qsl(self) -> Any:
        """Get LoadGen QSL handle."""
        if self._qsl_handle is None:
            self._qsl_handle = lg.ConstructQSL(
                self.qsl.total_sample_count,
                self.qsl.performance_sample_count,
                self.qsl.load_query_samples,
                self.qsl.unload_query_samples
            )
        return self._qsl_handle

    def get_predictions(self) -> Dict[int, str]:
        """Get all predictions."""
        return self._predictions.copy()

    def reset(self) -> None:
        """Reset state for new run."""
        self._predictions.clear()
        self._query_count = 0
        self._sample_count = 0
        self._die_index = 0

        # Reset all die SUTs
        for sut in self._die_suts.values():
            sut.reset()


def is_whisper_multi_die_available() -> bool:
    """Check if multi-die Whisper SUT is available."""
    return LOADGEN_AVAILABLE and OPTIMUM_AVAILABLE
