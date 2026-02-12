import logging
import sys
from pathlib import Path
from typing import Optional

import click
import numpy as np

from .core.config import BenchmarkConfig, Scenario, TestMode, ModelType, SUPPORTED_SCENARIOS
from .core.benchmark_runner import BenchmarkRunner
from .utils.model_downloader import download_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="1.0.0", prog_name="mlperf-ov")
def main():
    """MLPerf OpenVINO Benchmark Tool."""
    pass


def get_default_config(model: str) -> BenchmarkConfig:
    if model == 'resnet50':
        return BenchmarkConfig.default_resnet50()
    elif model == 'bert':
        return BenchmarkConfig.default_bert()
    elif model == 'retinanet':
        return BenchmarkConfig.default_retinanet()
    elif model == 'whisper':
        return BenchmarkConfig.default_whisper()
    elif model == 'sdxl':
        return BenchmarkConfig.default_sdxl()
    elif model == 'ssd-resnet34':
        return BenchmarkConfig.default_ssd_resnet34()
    elif model == 'llama3.1-8b':
        return BenchmarkConfig.default_llama3_1_8b()
    else:
        return BenchmarkConfig.default_resnet50()


@main.command()
@click.option('--model', '-m', type=click.Choice(['resnet50', 'bert', 'retinanet', 'whisper', 'sdxl', 'ssd-resnet34', 'llama3.1-8b']),
              default='resnet50', help='Model to benchmark')
@click.option('--scenario', '-s', type=click.Choice(['Offline', 'Server']),
              default='Offline', help='Test scenario')
@click.option('--mode', type=click.Choice(['accuracy', 'performance', 'both']),
              default='performance', help='Test mode')
@click.option('--model-path', type=click.Path(exists=True),
              help='Path to model file (ONNX or OpenVINO IR)')
@click.option('--data-path', type=click.Path(exists=True),
              help='Path to dataset directory')
@click.option('--output-dir', '-o', type=click.Path(),
              default='./results', help='Output directory for results')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to configuration file')
@click.option('--device', '-d', type=str, default='CPU',
              help='Device: CPU (default), NPU (all dies), NPU.0 (one die), NPU.0,NPU.2 (selected dies)')
@click.option('--properties', '-p', type=str, default='',
              help='Device-specific properties (KEY=VALUE,KEY2=VALUE2,...)')
@click.option('--num-threads', type=int, default=0,
              help='Number of threads (0 = auto)')
@click.option('--num-streams', type=str, default='AUTO',
              help='Number of inference streams')
@click.option('--batch-size', '-b', type=int, default=None,
              help='Inference batch size (default: 1, or 32 for CPU Offline with THROUGHPUT hint)')
@click.option('--nchw', is_flag=True,
              help='Use NCHW input layout (default is NHWC for ResNet50)')
@click.option('--performance-hint', type=click.Choice(['THROUGHPUT', 'LATENCY', 'AUTO']),
              default='AUTO', help='Performance hint (AUTO selects based on scenario)')
@click.option('--duration', type=int, default=None,
              help='Minimum test duration in ms (default: from config)')
@click.option('--min-query-count', type=int, default=None,
              help='Minimum query count (default: from config)')
@click.option('--target-qps', type=float, default=0,
              help='Target QPS (queries per second)')
@click.option('--target-latency-ns', type=int, default=0,
              help='Target latency in nanoseconds (Server mode, 0=use default 15ms)')
@click.option('--count', type=int, default=0,
              help='Number of samples to use (0 = all)')
@click.option('--warmup', type=int, default=10,
              help='Number of warmup iterations')
@click.option('--nireq-multiplier', type=int, default=None,
              help='In-flight request multiplier (default: from config, lower = less latency)')
@click.option('--auto-batch-timeout-ms', type=int, default=0,
              help='AUTO_BATCH timeout in ms (0=disabled, 1=1ms). Enables OpenVINO auto-batching.')
@click.option('--optimal-batch-size', type=int, default=0,
              help='Optimal batch size for AUTO_BATCH (0=auto, 4=recommended for NPU)')
@click.option('--explicit-batching', is_flag=True, default=False,
              help='Enable Intel-style explicit batching for Server mode (recommended for NPU)')
@click.option('--batch-timeout-us', type=int, default=None,
              help='Explicit batching timeout in microseconds (default: 2000 for explicit batching)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def run(model: str, scenario: str, mode: str, model_path: Optional[str],
        data_path: Optional[str], output_dir: str, config: Optional[str],
        device: str, properties: str, num_threads: int, num_streams: str,
        batch_size: int, nchw: bool, performance_hint: str, duration: Optional[int],
        min_query_count: Optional[int], target_qps: float,
        target_latency_ns: int, count: int, warmup: int, nireq_multiplier: int,
        auto_batch_timeout_ms: int, optimal_batch_size: int,
        explicit_batching: bool, batch_timeout_us: Optional[int], verbose: bool):
    """Run MLPerf benchmark."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    click.echo(f"\n{'='*60}")
    click.echo("MLPerf OpenVINO Benchmark")
    click.echo(f"{'='*60}\n")

    if config:
        benchmark_config = BenchmarkConfig.from_yaml(config, model)
        click.echo(f"Config: {config}")
    else:
        click.echo("WARNING: No --config specified, using hardcoded defaults (min_duration=600s)")
        benchmark_config = get_default_config(model)

    benchmark_config.scenario = Scenario(scenario)
    benchmark_config.results_dir = output_dir
    benchmark_config.openvino.device = device.upper() if device else "CPU"

    supported = SUPPORTED_SCENARIOS.get(benchmark_config.model.model_type)
    if supported and benchmark_config.scenario not in supported:
        names = ", ".join(s.value for s in supported)
        click.echo(f"Error: {model} does not support {scenario} scenario.")
        click.echo(f"Supported scenarios for {model}: {names}")
        sys.exit(1)

    if properties:
        from .backends.device_discovery import parse_device_properties, validate_device_properties
        parsed_props = parse_device_properties(properties)
        benchmark_config.openvino.device_properties = parsed_props
        is_valid, warnings = validate_device_properties(parsed_props, device)
        for warning in warnings:
            logger.debug(warning)

    if auto_batch_timeout_ms > 0 or optimal_batch_size > 0:
        if not benchmark_config.openvino.device_properties:
            benchmark_config.openvino.device_properties = {}
        if auto_batch_timeout_ms > 0:
            benchmark_config.openvino.device_properties['AUTO_BATCH_TIMEOUT'] = str(auto_batch_timeout_ms)
        if optimal_batch_size > 0:
            benchmark_config.openvino.device_properties['OPTIMAL_BATCH_SIZE'] = str(optimal_batch_size)
        click.echo(f"AUTO_BATCH: timeout={auto_batch_timeout_ms}ms, optimal_batch={optimal_batch_size}")

    if benchmark_config.openvino.is_accelerator_device():
        _validate_accelerator_device(benchmark_config.openvino.device)

    if num_threads > 0:
        benchmark_config.openvino.num_threads = num_threads

    is_accelerator = benchmark_config.openvino.is_accelerator_device()

    if performance_hint == 'AUTO':
        if is_accelerator:
            actual_hint = None
        elif scenario == 'Offline':
            actual_hint = 'THROUGHPUT'
            if batch_size is None and model not in ('sdxl', 'ssd-resnet34', 'llama3.1-8b'):
                batch_size = 32
        else:
            actual_hint = 'THROUGHPUT'
    else:
        actual_hint = performance_hint

    if batch_size is None:
        batch_size = 1 if model in ('sdxl', 'llama3.1-8b') else 0

    benchmark_config.openvino.num_streams = num_streams
    benchmark_config.openvino.batch_size = batch_size

    if hasattr(benchmark_config.model, 'preprocessing') and benchmark_config.model.preprocessing:
        benchmark_config.model.preprocessing.output_layout = 'NCHW' if nchw else 'NHWC'

    if not is_accelerator and actual_hint:
        benchmark_config.openvino.performance_hint = actual_hint

    if model_path:
        benchmark_config.model.model_path = model_path

    if data_path:
        benchmark_config.dataset.path = data_path

    if count > 0:
        benchmark_config.dataset.num_samples = count

    scenario_config = benchmark_config.get_scenario_config()
    if duration is not None:
        scenario_config.min_duration_ms = duration
    if min_query_count is not None:
        scenario_config.min_query_count = min_query_count
    if target_qps > 0:
        scenario_config.target_qps = target_qps

    if scenario == 'Server':
        if nireq_multiplier is not None:
            scenario_config.nireq_multiplier = nireq_multiplier
        if target_latency_ns > 0:
            scenario_config.target_latency_ns = target_latency_ns
            click.echo(f"Custom target latency: {target_latency_ns / 1e6:.1f}ms (Open Division)")

        if explicit_batching:
            scenario_config.explicit_batching = True
            scenario_config.batch_timeout_us = batch_timeout_us if batch_timeout_us is not None else 2000
            # Default to nireq_multiplier=6 for explicit batching if not overridden
            if nireq_multiplier is None:
                scenario_config.nireq_multiplier = 6
            explicit_batch = batch_size if batch_size > 1 else 8
            scenario_config.explicit_batch_size = explicit_batch
            click.echo(f"Explicit batching: batch={explicit_batch}, timeout={scenario_config.batch_timeout_us}us, nireq={scenario_config.nireq_multiplier}")

    if not benchmark_config.model.model_path:
        click.echo("Error: Model path is required. Use --model-path or download the model first.")
        click.echo(f"Run: mlperf-ov download --model {model}")
        sys.exit(1)

    if not Path(benchmark_config.dataset.path).exists():
        click.echo(f"Error: Dataset path does not exist: {benchmark_config.dataset.path}")
        _print_dataset_help(model)
        sys.exit(1)

    click.echo(f"Model: {benchmark_config.model.name}")
    click.echo(f"Task: {benchmark_config.model.task}")
    click.echo(f"Mode: {mode}")
    if mode == 'performance':
        click.echo(f"Scenario: {benchmark_config.scenario.value}")
    click.echo(f"Device: {benchmark_config.openvino.device}")
    click.echo(f"Batch size: {benchmark_config.openvino.batch_size}")
    if hasattr(benchmark_config.model, 'preprocessing') and benchmark_config.model.preprocessing:
        input_layout = getattr(benchmark_config.model.preprocessing, 'output_layout', 'NCHW')
        click.echo(f"Input layout: {input_layout}")
    click.echo("")

    runner = BenchmarkRunner(benchmark_config)

    if warmup > 0:
        click.echo(f"Warming up ({warmup} iterations)...")
        runner.setup()
        if runner.backend is not None:
            runner.backend.warmup(warmup)
        elif hasattr(runner.sut, 'warmup') and callable(getattr(runner.sut, 'warmup')):
            runner.sut.warmup(warmup)
        else:
            click.echo("Skipping warmup (model uses separate components)")

    if mode == 'accuracy':
        click.echo("Running accuracy test...")
        benchmark_config.test_mode = TestMode.ACCURACY_ONLY
        results = runner.run()
    elif mode == 'performance':
        click.echo("Running performance test...")
        benchmark_config.test_mode = TestMode.PERFORMANCE_ONLY
        results = runner.run()
    else:  # both
        click.echo("Running accuracy test...")
        benchmark_config.test_mode = TestMode.ACCURACY_ONLY
        acc_results = runner.run()

        runner.sut.reset()

        click.echo("Running performance test...")
        benchmark_config.test_mode = TestMode.PERFORMANCE_ONLY
        perf_results = runner.run()

        results = {**perf_results, "accuracy": acc_results.get("accuracy", {})}

    runner.print_summary()
    results_path = runner.save_results()
    click.echo(f"Results saved to: {results_path}")


def _validate_accelerator_device(device: str) -> None:
    try:
        from openvino import Core
        from .backends.device_discovery import validate_accelerator_device

        core = Core()

        is_valid, error = validate_accelerator_device(core, device)
        if not is_valid:
            click.echo(f"Error: {error}")
            click.echo(f"Available devices: {core.available_devices}")
            sys.exit(1)

    except ImportError as e:
        logger.warning(f"Could not validate accelerator device: {e}")
    except Exception as e:
        logger.error(f"Error validating accelerator device: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _print_dataset_help(model: str) -> None:
    if model == 'resnet50':
        click.echo("Please download the ImageNet validation dataset.")
        click.echo("Run: mlperf-ov download-dataset --dataset imagenet")
    elif model == 'bert':
        click.echo("Please download the SQuAD dataset.")
        click.echo("Run: mlperf-ov download-dataset --dataset squad")
    elif model == 'retinanet':
        click.echo("Please download the OpenImages dataset.")
        click.echo("Run: mlperf-ov download-dataset --dataset openimages")
    elif model == 'whisper':
        click.echo("Please download the LibriSpeech dataset.")
        click.echo("Run: mlperf-ov download-dataset --dataset librispeech")
    elif model == 'ssd-resnet34':
        click.echo("Please download the COCO 2017 validation dataset.")
        click.echo("Run: mlperf-ov download-dataset --dataset coco2017")
    elif model == 'sdxl':
        click.echo("Please download the COCO 2014 captions dataset.")
        click.echo("Run: mlperf-ov download-dataset --dataset coco2014")
    elif model == 'llama3.1-8b':
        click.echo("Please download the CNN-DailyMail dataset.")
        click.echo("Run: mlperf-ov download-dataset --dataset cnn-dailymail")


@main.command('download-model')
@click.option('--model', '-m', type=click.Choice(['resnet50', 'bert', 'retinanet', 'whisper', 'sdxl', 'ssd-resnet34', 'llama3.1-8b']),
              default='resnet50', help='Model to download')
@click.option('--output-dir', '-o', type=click.Path(),
              default='./models', help='Output directory')
@click.option('--format', '-f', type=click.Choice(['onnx', 'openvino']),
              default='onnx', help='Model format')
@click.option('--batch-sizes', type=str, default='1,2,4,8',
              help='Batch sizes for RetinaNet (comma-separated, default: 1,2,4,8)')
@click.option('--hf-token', type=str, default=None, envvar='HF_TOKEN',
              help='HuggingFace access token (for gated models like Meta-Llama). Also reads HF_TOKEN env var.')
@click.option('--stateful/--no-stateful', default=False,
              help='Llama export: merge KV-cache into OpenVINO state variables (--stateful) '
                   'or keep past_key_values as explicit graph I/O (--no-stateful, default).')
def download_model_cmd(model: str, output_dir: str, format: str, batch_sizes: str, hf_token: Optional[str], stateful: bool):
    """Download model files."""
    click.echo(f"Downloading {model} model...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        if model == 'whisper':
            from .utils.model_downloader import download_whisper_model
            # Whisper always needs OpenVINO IR format for inference
            if format != 'openvino':
                click.echo("  Note: Whisper requires OpenVINO format, using --format openvino")
            paths = download_whisper_model(
                str(output_path),
                export_to_openvino=True,
            )
            click.echo(f"Model downloaded to: {paths['model_path']}")
            if 'encoder_path' in paths:
                click.echo(f"  Encoder: {paths['encoder_path']}")
                click.echo(f"  Decoder: {paths['decoder_path']}")
            if 'decoder_with_past_path' in paths:
                click.echo(f"  Decoder with KV-cache: {paths['decoder_with_past_path']}")
        elif model == 'sdxl':
            from .utils.model_downloader import download_sdxl_model
            if format != 'openvino':
                click.echo("  Note: SDXL requires OpenVINO format, using --format openvino")
            paths = download_sdxl_model(
                str(output_path),
                export_to_openvino=True,
            )
            click.echo(f"Model downloaded to: {paths['model_path']}")
        elif model == 'retinanet':
            from .utils.model_downloader import download_retinanet_model
            bs_list = [int(x.strip()) for x in batch_sizes.split(',')]
            convert_to_openvino = (format == 'openvino')
            paths = download_retinanet_model(
                str(output_path),
                batch_sizes=bs_list,
                convert_to_openvino=convert_to_openvino,
            )
            click.echo(f"RetinaNet ONNX model: {paths['onnx_path']}")
            if 'batch_models' in paths and paths['batch_models']:
                click.echo("OpenVINO IR models:")
                for bs, path in sorted(paths['batch_models'].items()):
                    click.echo(f"  Batch size {bs}: {path}")
        elif model == 'llama3.1-8b':
            from .utils.model_downloader import download_llama_model
            if format != 'openvino':
                click.echo("  Note: Llama requires OpenVINO format, using --format openvino")
            paths = download_llama_model(
                str(output_path),
                export_to_openvino=True,
                hf_token=hf_token,
                stateful=stateful,
            )
            click.echo(f"Model downloaded to: {paths['model_path']}")
        else:
            model_path = download_model(model, str(output_path), format)
            click.echo(f"Model downloaded to: {model_path}")
    except Exception as e:
        click.echo(f"Error downloading model: {e}")
        sys.exit(1)


@main.command()
def info():
    """Show system and library information."""
    import platform

    click.echo("\nSystem Information:")
    click.echo(f"  Platform: {platform.platform()}")
    click.echo(f"  Python: {platform.python_version()}")
    click.echo(f"  Processor: {platform.processor()}")

    try:
        import psutil
        click.echo(f"  Physical cores: {psutil.cpu_count(logical=False)}")
        click.echo(f"  Logical cores: {psutil.cpu_count(logical=True)}")
        click.echo(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    except ImportError:
        pass

    click.echo("\nLibrary Versions:")

    try:
        import openvino as ov
        click.echo(f"  OpenVINO: {ov.__version__}")
    except ImportError:
        click.echo("  OpenVINO: Not installed")

    try:
        import mlperf_loadgen
        click.echo("  MLPerf LoadGen: Installed")
    except ImportError:
        click.echo("  MLPerf LoadGen: Not installed")

    try:
        import numpy as np
        click.echo(f"  NumPy: {np.__version__}")
    except ImportError:
        pass

    try:
        from PIL import Image
        import PIL
        click.echo(f"  Pillow: {PIL.__version__}")
    except ImportError:
        click.echo("  Pillow: Not installed")

    try:
        import transformers
        click.echo(f"  Transformers: {transformers.__version__}")
    except ImportError:
        click.echo("  Transformers: Not installed")

    click.echo("\nSupported Models:")
    click.echo("  - ResNet50 (Image Classification)")
    click.echo("  - BERT-Large (Question Answering)")
    click.echo("  - RetinaNet (Object Detection)")
    click.echo("  - Whisper (Speech Recognition)")
    click.echo("  - SDXL (Text-to-Image Generation)")
    click.echo("  - Llama 3.1 8B (Text Generation)")


@main.command('download-dataset')
@click.option('--dataset', '-d', type=click.Choice(['imagenet', 'librispeech', 'squad', 'openimages', 'coco2014', 'coco2017', 'cnn-dailymail']),
              required=True, help='Dataset to download')
@click.option('--output-dir', '-o', type=click.Path(),
              default='./data', help='Output directory')
@click.option('--subset', '-s', type=str, default=None,
              help='Dataset subset (e.g., "dev-clean" for librispeech)')
@click.option('--count', '-c', type=int, default=None,
              help='Max images to download (OpenImages only, default=all)')
@click.option('--with-images', is_flag=True,
              help='Download reference images (COCO 2014 only, ~6GB, required for FID computation)')
@click.option('--force', is_flag=True, help='Force re-download')
@click.option('--hf-token', type=str, default=None, envvar='HF_TOKEN',
              help='HuggingFace access token (for CNN-DailyMail tokenizer). Also reads HF_TOKEN env var.')
def download_dataset_cmd(dataset: str, output_dir: str, subset: Optional[str],
                         count: Optional[int], with_images: bool, force: bool,
                         hf_token: Optional[str]):
    """Download dataset files."""
    from .utils.dataset_downloader import download_dataset, get_dataset_info

    click.echo(f"\nDownloading {dataset} dataset...")

    try:
        info = get_dataset_info(dataset)
        click.echo(f"Description: {info.get('description', 'N/A')}")
    except ValueError:
        pass

    if subset:
        click.echo(f"Subset: {subset}")
    if count and dataset == 'openimages':
        click.echo(f"Max images: {count}")
    if with_images and dataset == 'coco2014':
        click.echo("Including reference images (~6GB)")

    click.echo("")

    try:
        if dataset == 'openimages':
            from .utils.dataset_downloader import download_openimages
            paths = download_openimages(output_dir, force=force, max_images=count)
        elif dataset == 'coco2017':
            from .utils.dataset_downloader import download_coco2017
            paths = download_coco2017(output_dir, force=force)
        elif dataset == 'coco2014':
            from .utils.dataset_downloader import download_coco2014
            paths = download_coco2014(output_dir, force=force, download_images=with_images)
        else:
            paths = download_dataset(dataset, output_dir, subset, force, hf_token=hf_token)

        click.echo("\nDataset downloaded successfully!")
        click.echo(f"  Path: {paths.get('data_path', 'N/A')}")

        if 'val_map' in paths:
            click.echo(f"  Val map: {paths['val_map']}")
        if 'transcript_path' in paths:
            click.echo(f"  Transcripts: {paths['transcript_path']}")
        if 'num_samples' in paths:
            click.echo(f"  Samples: {paths['num_samples']}")

    except Exception as e:
        click.echo(f"\nError downloading dataset: {e}")
        sys.exit(1)


@main.command('setup')
@click.option('--model', '-m', type=click.Choice(['resnet50', 'bert', 'retinanet', 'whisper', 'sdxl', 'ssd-resnet34', 'llama3.1-8b']),
              required=True, help='Model to set up')
@click.option('--output-dir', '-o', type=click.Path(),
              default='.', help='Base output directory')
@click.option('--format', '-f', type=click.Choice(['onnx', 'openvino']),
              default='onnx', help='Model format')
@click.option('--hf-token', type=str, default=None, envvar='HF_TOKEN',
              help='HuggingFace access token (for gated models like Meta-Llama). Also reads HF_TOKEN env var.')
def setup_cmd(model: str, output_dir: str, format: str, hf_token: Optional[str]):
    """Download both model and dataset for a benchmark."""
    from .utils.model_downloader import download_model, download_whisper_model, download_sdxl_model
    from .utils.model_downloader import download_retinanet_model, download_llama_model
    from .utils.dataset_downloader import download_dataset

    output_path = Path(output_dir)
    models_dir = output_path / "models"
    data_dir = output_path / "data"

    click.echo(f"\n{'='*60}")
    click.echo(f"Setting up {model} benchmark")
    click.echo(f"{'='*60}\n")

    click.echo("Step 1: Downloading model...")
    try:
        if model == 'whisper':
            # Whisper requires OpenVINO IR format (WhisperOptimumSUT needs .xml/.bin files)
            if format != 'openvino':
                click.echo("  Note: Whisper requires OpenVINO format, using --format openvino")
            model_paths = download_whisper_model(
                str(models_dir),
                export_to_openvino=True,
            )
            model_path = model_paths['model_path']
        elif model == 'sdxl':
            if format != 'openvino':
                click.echo("  Note: SDXL requires OpenVINO format, using --format openvino")
            model_paths = download_sdxl_model(
                str(models_dir),
                export_to_openvino=True,
            )
            model_path = model_paths['model_path']
        elif model == 'retinanet':
            convert_to_openvino = (format == 'openvino')
            model_paths = download_retinanet_model(
                str(models_dir),
                batch_sizes=[1, 2, 4, 8],
                convert_to_openvino=convert_to_openvino,
            )
            model_path = model_paths.get('model_path', model_paths['onnx_path'])
            click.echo(f"  ONNX model: {model_paths['onnx_path']}")
            if 'batch_models' in model_paths and model_paths['batch_models']:
                click.echo("  OpenVINO IR models:")
                for bs, path in sorted(model_paths['batch_models'].items()):
                    click.echo(f"    Batch size {bs}: {path}")
        elif model == 'llama3.1-8b':
            if format != 'openvino':
                click.echo("  Note: Llama requires OpenVINO format, using --format openvino")
            model_paths = download_llama_model(
                str(models_dir),
                export_to_openvino=True,
                hf_token=hf_token,
            )
            model_path = model_paths['model_path']
        else:
            model_path = download_model(model, str(models_dir), format)
        click.echo(f"  Model: {model_path}\n")
    except Exception as e:
        click.echo(f"  Error: {e}")
        sys.exit(1)

    click.echo("Step 2: Downloading dataset...")
    try:
        if model == 'resnet50':
            dataset_paths = download_dataset('imagenet', str(data_dir))
        elif model == 'bert':
            dataset_paths = download_dataset('squad', str(data_dir))
        elif model == 'retinanet':
            dataset_paths = download_dataset('openimages', str(data_dir))
        elif model == 'whisper':
            # MLPerf requires dev-clean + dev-other combined
            dataset_paths = download_dataset('librispeech', str(data_dir), 'mlperf')
        elif model == 'ssd-resnet34':
            from .utils.dataset_downloader import download_coco2017
            dataset_paths = download_coco2017(str(data_dir))
        elif model == 'sdxl':
            dataset_paths = download_dataset('coco2014', str(data_dir))
        elif model == 'llama3.1-8b':
            from .utils.dataset_downloader import download_cnn_dailymail
            dataset_paths = download_cnn_dailymail(str(data_dir), hf_token=hf_token)

        click.echo(f"  Dataset: {dataset_paths.get('data_path', 'N/A')}\n")
    except Exception as e:
        click.echo(f"  Error: {e}")
        sys.exit(1)

    click.echo(f"{'='*60}")
    click.echo("Setup complete! Run benchmark with:")
    click.echo(f"{'='*60}\n")

    data_path = dataset_paths.get('data_path', f'{data_dir}/{model}')

    click.echo(f"  mlperf-ov run --model {model} \\")
    click.echo(f"    --model-path {model_path} \\")
    click.echo(f"    --data-path {data_path}")

    click.echo("")


@main.command('list-models')
def list_models():
    """List all supported models with their details."""
    click.echo("\n" + "="*70)
    click.echo("Supported Models for MLPerf OpenVINO Benchmark")
    click.echo("="*70 + "\n")

    models = [
        {
            'name': 'ResNet50-v1.5',
            'id': 'resnet50',
            'type': ModelType.RESNET50,
            'task': 'Image Classification',
            'dataset': 'ImageNet 2012',
            'metric': 'Top-1 Accuracy',
            'target': '76.46%',
        },
        {
            'name': 'BERT-Large',
            'id': 'bert',
            'type': ModelType.BERT,
            'task': 'Question Answering',
            'dataset': 'SQuAD v1.1',
            'metric': 'F1 Score',
            'target': '90.874%',
        },
        {
            'name': 'RetinaNet',
            'id': 'retinanet',
            'type': ModelType.RETINANET,
            'task': 'Object Detection',
            'dataset': 'OpenImages',
            'metric': 'mAP',
            'target': '37.57%',
        },
        {
            'name': 'Whisper Large v3',
            'id': 'whisper',
            'type': ModelType.WHISPER,
            'task': 'Speech Recognition',
            'dataset': 'LibriSpeech',
            'metric': 'Word Accuracy',
            'target': '97.93%',
        },
        {
            'name': 'SSD-ResNet34',
            'id': 'ssd-resnet34',
            'type': ModelType.SSD_RESNET34,
            'task': 'Object Detection',
            'dataset': 'COCO 2017',
            'metric': 'mAP',
            'target': '20.0%',
        },
        {
            'name': 'Stable Diffusion XL',
            'id': 'sdxl',
            'type': ModelType.SDXL,
            'task': 'Text-to-Image',
            'dataset': 'COCO 2014',
            'metric': 'CLIP Score / FID',
            'target': '31.69-31.81 / 23.01-23.95',
        },
        {
            'name': 'Llama 3.1 8B',
            'id': 'llama3.1-8b',
            'type': ModelType.LLAMA3_1_8B,
            'task': 'Text Generation',
            'dataset': 'CNN-DailyMail v3.0.0',
            'metric': 'ROUGE-1/2/L/Lsum',
            'target': '38.7792 / 15.9075 / 24.4957 / 35.793',
        },
    ]

    for m in models:
        scenarios = SUPPORTED_SCENARIOS.get(m['type'], [])
        scenario_str = ", ".join(s.value for s in scenarios)
        click.echo(f"{m['name']} ({m['id']})")
        click.echo(f"  Task:      {m['task']}")
        click.echo(f"  Dataset:   {m['dataset']}")
        click.echo(f"  Metric:    {m['metric']} (Target: {m['target']})")
        click.echo(f"  Scenarios: {scenario_str}")
        click.echo("")

    click.echo("Quick start:")
    click.echo("  mlperf-ov setup --model <model_id>")
    click.echo("  mlperf-ov run --model <model_id> --mode both")
    click.echo("")


if __name__ == "__main__":
    main()
