"""
Command Line Interface for MLPerf OpenVINO Benchmark.

Supports: ResNet50, BERT, RetinaNet, Whisper
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import click
import numpy as np

from .core.config import BenchmarkConfig, Scenario, TestMode, ModelType
from .core.benchmark_runner import BenchmarkRunner
from .utils.model_downloader import download_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.2.0", prog_name="mlperf-ov")
def main():
    """
    MLPerf v5.1 OpenVINO Benchmark Tool

    A benchmark tool for measuring CPU inference performance using OpenVINO,
    compatible with MLPerf Inference v5.1 specifications.

    Supported models:
    - ResNet50 (Image Classification on ImageNet)
    - BERT-Large (Question Answering on SQuAD)
    - RetinaNet (Object Detection on OpenImages)
    - Whisper (Speech Recognition on LibriSpeech)
    """
    pass


def get_default_config(model: str) -> BenchmarkConfig:
    """Get default configuration for a model."""
    if model == 'resnet50':
        return BenchmarkConfig.default_resnet50()
    elif model == 'bert':
        return BenchmarkConfig.default_bert()
    elif model == 'retinanet':
        return BenchmarkConfig.default_retinanet()
    elif model == 'whisper':
        return BenchmarkConfig.default_whisper()
    else:
        return BenchmarkConfig.default_resnet50()


@main.command()
@click.option('--model', '-m', type=click.Choice(['resnet50', 'bert', 'retinanet', 'whisper']),
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
@click.option('--num-threads', type=int, default=0,
              help='Number of threads (0 = auto)')
@click.option('--num-streams', type=str, default='AUTO',
              help='Number of inference streams')
@click.option('--batch-size', '-b', type=int, default=1,
              help='Inference batch size')
@click.option('--performance-hint', type=click.Choice(['THROUGHPUT', 'LATENCY', 'AUTO']),
              default='AUTO', help='Performance hint (AUTO selects based on scenario)')
@click.option('--duration', type=int, default=60000,
              help='Minimum test duration in ms')
@click.option('--target-qps', type=float, default=0,
              help='Target QPS (queries per second)')
@click.option('--count', type=int, default=0,
              help='Number of samples to use (0 = all)')
@click.option('--warmup', type=int, default=10,
              help='Number of warmup iterations')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def run(model: str, scenario: str, mode: str, model_path: Optional[str],
        data_path: Optional[str], output_dir: str, config: Optional[str],
        num_threads: int, num_streams: str, batch_size: int, performance_hint: str,
        duration: int, target_qps: float, count: int, warmup: int, verbose: bool):
    """
    Run MLPerf benchmark.

    Examples:

        # ResNet50 benchmark
        mlperf-ov run --model resnet50 --scenario Offline --mode performance

        # BERT benchmark
        mlperf-ov run --model bert --model-path ./models/bert.onnx --data-path ./data/squad

        # RetinaNet benchmark
        mlperf-ov run --model retinanet --mode accuracy

        # Whisper benchmark
        mlperf-ov run --model whisper --data-path ./data/librispeech
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    click.echo(f"\n{'='*60}")
    click.echo("MLPerf v5.1 OpenVINO Benchmark")
    click.echo(f"{'='*60}\n")

    # Load or create configuration based on model
    if config:
        click.echo(f"Loading configuration from: {config}")
        benchmark_config = BenchmarkConfig.from_yaml(config, model)
    else:
        click.echo(f"Using default configuration for {model}")
        benchmark_config = get_default_config(model)

    # Override with CLI options
    benchmark_config.scenario = Scenario(scenario)
    benchmark_config.results_dir = output_dir

    if num_threads > 0:
        benchmark_config.openvino.num_threads = num_threads

    # Auto-select optimal settings based on scenario
    if performance_hint == 'AUTO':
        if scenario == 'Offline':
            # Offline: optimize for maximum THROUGHPUT
            actual_hint = 'THROUGHPUT'
            if batch_size == 1:  # User didn't specify batch size
                click.echo("AUTO: Using optimized settings for Offline (THROUGHPUT)")
                batch_size = 32  # Larger batch for throughput
        else:
            # Server: optimize for minimum LATENCY
            # Server mode measures latency-bounded throughput
            # LATENCY hint minimizes single-sample inference time
            actual_hint = 'LATENCY'
            click.echo("AUTO: Using optimized settings for Server (LATENCY)")
            # Keep batch_size=1 for Server (each query = 1 sample)
    else:
        actual_hint = performance_hint

    benchmark_config.openvino.num_streams = num_streams
    benchmark_config.openvino.batch_size = batch_size
    benchmark_config.openvino.performance_hint = actual_hint

    if model_path:
        benchmark_config.model.model_path = model_path

    if data_path:
        benchmark_config.dataset.path = data_path

    if count > 0:
        benchmark_config.dataset.num_samples = count

    # Update scenario config
    scenario_config = benchmark_config.get_scenario_config()
    scenario_config.min_duration_ms = duration
    if target_qps > 0:
        scenario_config.target_qps = target_qps
        click.echo(f"Server mode: target_qps={target_qps}")
    elif scenario == 'Server':
        # For Server mode, set high target_qps if not specified
        # This allows LoadGen to send queries as fast as the system can handle
        scenario_config.target_qps = 50000.0  # Very high to not be the bottleneck
        click.echo(f"Server mode: target_qps={scenario_config.target_qps} (use --target-qps to set)")
        click.echo("NOTE: Server mode measures latency-bounded throughput.")
        click.echo("      For maximum throughput, use --scenario Offline")

    # Validate configuration
    if not benchmark_config.model.model_path:
        click.echo("Error: Model path is required. Use --model-path or download the model first.")
        click.echo(f"Run: mlperf-ov download --model {model}")
        sys.exit(1)

    if not Path(benchmark_config.dataset.path).exists():
        click.echo(f"Error: Dataset path does not exist: {benchmark_config.dataset.path}")
        _print_dataset_help(model)
        sys.exit(1)

    # Print configuration summary
    click.echo(f"Model: {benchmark_config.model.name}")
    click.echo(f"Task: {benchmark_config.model.task}")
    click.echo(f"Scenario: {benchmark_config.scenario.value}")
    click.echo(f"Device: {benchmark_config.openvino.device}")
    click.echo(f"Threads: {benchmark_config.openvino.num_threads or 'auto'}")
    click.echo(f"Streams: {benchmark_config.openvino.num_streams}")
    click.echo(f"Batch size: {benchmark_config.openvino.batch_size}")
    click.echo(f"Performance hint: {benchmark_config.openvino.performance_hint}")
    click.echo("")

    # Create runner
    runner = BenchmarkRunner(benchmark_config)

    # Warmup
    if warmup > 0:
        click.echo(f"Warming up ({warmup} iterations)...")
        runner.setup()
        runner.backend.warmup(warmup)

    # Run benchmark
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

    # Print summary
    runner.print_summary()

    # Save results
    results_path = runner.save_results()
    click.echo(f"Results saved to: {results_path}")


def _print_dataset_help(model: str) -> None:
    """Print dataset download help for a model."""
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


@main.command()
@click.option('--model', '-m', type=click.Choice(['resnet50', 'bert', 'retinanet', 'whisper']),
              default='resnet50', help='Model to download')
@click.option('--output-dir', '-o', type=click.Path(),
              default='./models', help='Output directory')
@click.option('--format', '-f', type=click.Choice(['onnx', 'openvino']),
              default='onnx', help='Model format')
def download(model: str, output_dir: str, format: str):
    """
    Download model files.

    Examples:

        mlperf-ov download --model resnet50 --output-dir ./models

        mlperf-ov download --model bert --format onnx

        mlperf-ov download --model whisper --format openvino
    """
    click.echo(f"Downloading {model} model...")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        if model == 'whisper':
            from .utils.model_downloader import download_whisper_model
            export_to_openvino = (format == 'openvino')
            paths = download_whisper_model(
                str(output_path),
                export_to_openvino=export_to_openvino
            )
            click.echo(f"Model downloaded to: {paths['model_path']}")
            if 'encoder_path' in paths:
                click.echo(f"  Encoder: {paths['encoder_path']}")
                click.echo(f"  Decoder: {paths['decoder_path']}")
        else:
            model_path = download_model(model, str(output_path), format)
            click.echo(f"Model downloaded to: {model_path}")
    except Exception as e:
        click.echo(f"Error downloading model: {e}")
        sys.exit(1)


@main.command()
@click.option('--model-path', '-m', type=click.Path(exists=True), required=True,
              help='Path to model file')
@click.option('--iterations', '-n', type=int, default=100,
              help='Number of iterations')
@click.option('--warmup', type=int, default=10,
              help='Number of warmup iterations')
@click.option('--num-threads', type=int, default=0,
              help='Number of threads (0 = auto)')
def benchmark_latency(model_path: str, iterations: int, warmup: int, num_threads: int):
    """
    Run quick latency benchmark (without LoadGen).

    Example:

        mlperf-ov benchmark-latency -m ./models/resnet50.onnx -n 100
    """
    from .backends.openvino_backend import OpenVINOBackend
    from .core.config import OpenVINOConfig

    click.echo(f"Loading model: {model_path}")

    config = OpenVINOConfig(
        num_threads=num_threads,
        performance_hint="LATENCY",
    )

    backend = OpenVINOBackend(model_path, config)
    backend.load()

    click.echo(f"Running benchmark ({warmup} warmup + {iterations} iterations)...")
    results = backend.benchmark(num_iterations=iterations, warmup_iterations=warmup)

    click.echo("\nResults:")
    click.echo(f"  Mean latency:   {results['mean_latency_ms']:.2f} ms")
    click.echo(f"  Median latency: {results['median_latency_ms']:.2f} ms")
    click.echo(f"  Min latency:    {results['min_latency_ms']:.2f} ms")
    click.echo(f"  Max latency:    {results['max_latency_ms']:.2f} ms")
    click.echo(f"  P90 latency:    {results['p90_latency_ms']:.2f} ms")
    click.echo(f"  P99 latency:    {results['p99_latency_ms']:.2f} ms")
    click.echo(f"  Throughput:     {results['throughput_fps']:.2f} FPS")


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


@main.command('download-dataset')
@click.option('--dataset', '-d', type=click.Choice(['imagenet', 'librispeech', 'squad', 'openimages']),
              required=True, help='Dataset to download')
@click.option('--output-dir', '-o', type=click.Path(),
              default='./data', help='Output directory')
@click.option('--subset', '-s', type=str, default=None,
              help='Dataset subset (e.g., "dev-clean" for librispeech)')
@click.option('--force', is_flag=True, help='Force re-download')
def download_dataset_cmd(dataset: str, output_dir: str, subset: Optional[str], force: bool):
    """
    Download dataset files.

    Examples:

        # Download ImageNet validation subset
        mlperf-ov download-dataset --dataset imagenet

        # Download SQuAD v1.1
        mlperf-ov download-dataset --dataset squad

        # Download OpenImages validation set
        mlperf-ov download-dataset --dataset openimages

        # Download LibriSpeech dev-clean
        mlperf-ov download-dataset --dataset librispeech --subset dev-clean
    """
    from .utils.dataset_downloader import download_dataset, get_dataset_info

    click.echo(f"\nDownloading {dataset} dataset...")

    # Show dataset info
    try:
        info = get_dataset_info(dataset)
        click.echo(f"Description: {info.get('description', 'N/A')}")
    except ValueError:
        pass

    if subset:
        click.echo(f"Subset: {subset}")

    click.echo("")

    try:
        paths = download_dataset(dataset, output_dir, subset, force)

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
@click.option('--model', '-m', type=click.Choice(['resnet50', 'bert', 'retinanet', 'whisper']),
              required=True, help='Model to set up')
@click.option('--output-dir', '-o', type=click.Path(),
              default='.', help='Base output directory')
@click.option('--format', '-f', type=click.Choice(['onnx', 'openvino']),
              default='onnx', help='Model format')
def setup_cmd(model: str, output_dir: str, format: str):
    """
    Download both model and dataset for a benchmark.

    This is a convenience command that downloads everything needed
    to run a benchmark.

    Examples:

        # Set up ResNet50 benchmark
        mlperf-ov setup --model resnet50

        # Set up BERT benchmark
        mlperf-ov setup --model bert

        # Set up RetinaNet benchmark
        mlperf-ov setup --model retinanet

        # Set up Whisper benchmark with OpenVINO format
        mlperf-ov setup --model whisper --format openvino
    """
    from .utils.model_downloader import download_model, download_whisper_model
    from .utils.dataset_downloader import download_dataset

    output_path = Path(output_dir)
    models_dir = output_path / "models"
    data_dir = output_path / "data"

    click.echo(f"\n{'='*60}")
    click.echo(f"Setting up {model} benchmark")
    click.echo(f"{'='*60}\n")

    # Download model
    click.echo("Step 1: Downloading model...")
    try:
        if model == 'whisper':
            export_to_openvino = (format == 'openvino')
            model_paths = download_whisper_model(
                str(models_dir),
                export_to_openvino=export_to_openvino
            )
            model_path = model_paths['model_path']
        else:
            model_path = download_model(model, str(models_dir), format)
        click.echo(f"  Model: {model_path}\n")
    except Exception as e:
        click.echo(f"  Error: {e}")
        sys.exit(1)

    # Download dataset
    click.echo("Step 2: Downloading dataset...")
    try:
        if model == 'resnet50':
            dataset_paths = download_dataset('imagenet', str(data_dir))
        elif model == 'bert':
            dataset_paths = download_dataset('squad', str(data_dir))
        elif model == 'retinanet':
            dataset_paths = download_dataset('openimages', str(data_dir))
        elif model == 'whisper':
            dataset_paths = download_dataset('librispeech', str(data_dir), 'dev-clean')

        click.echo(f"  Dataset: {dataset_paths.get('data_path', 'N/A')}\n")
    except Exception as e:
        click.echo(f"  Error: {e}")
        sys.exit(1)

    # Print usage instructions
    click.echo(f"{'='*60}")
    click.echo("Setup complete! Run benchmark with:")
    click.echo(f"{'='*60}\n")

    data_path = dataset_paths.get('data_path', f'{data_dir}/{model}')

    click.echo(f"  mlperf-ov run --model {model} \\")
    click.echo(f"    --model-path {model_path} \\")
    click.echo(f"    --data-path {data_path}")

    click.echo("")


@main.command('perf-tips')
@click.option('--model', '-m', type=click.Choice(['resnet50', 'bert', 'retinanet', 'whisper']),
              default='resnet50', help='Model to show tips for')
def perf_tips(model: str):
    """
    Show performance optimization tips for maximum throughput.

    IMPORTANT: For maximum performance, use Offline mode, not Server mode!
    """
    click.echo("\n" + "="*70)
    click.echo("PERFORMANCE OPTIMIZATION TIPS")
    click.echo("="*70 + "\n")

    click.echo("1. USE OFFLINE MODE FOR MAXIMUM THROUGHPUT")
    click.echo("   " + "-"*50)
    click.echo("   Server mode is for LATENCY testing (rate-limited)")
    click.echo("   Offline mode is for THROUGHPUT testing (maximum speed)")
    click.echo("")
    click.echo("   BAD (slow):  --scenario Server")
    click.echo("   GOOD (fast): --scenario Offline")
    click.echo("")

    click.echo("2. INCREASE BATCH SIZE")
    click.echo("   " + "-"*50)
    click.echo("   Larger batch size = better throughput (more parallelism)")
    click.echo("")
    if model == 'resnet50':
        click.echo("   Recommended: --batch-size 8 to 64 (depends on RAM)")
    elif model == 'bert':
        click.echo("   Recommended: --batch-size 4 to 16")
    elif model == 'retinanet':
        click.echo("   Recommended: --batch-size 2 to 8 (large model)")
    elif model == 'whisper':
        click.echo("   Recommended: --batch-size 1 to 4 (sequential decoding)")
    click.echo("")

    click.echo("3. USE OPTIMAL NUMBER OF STREAMS")
    click.echo("   " + "-"*50)
    click.echo("   Streams allow parallel inference requests")
    click.echo("")
    click.echo("   --num-streams AUTO   (recommended, auto-detect)")
    click.echo("   --num-streams <N>    (N = number of physical cores)")
    click.echo("")

    click.echo("4. USE PERFORMANCE HINT")
    click.echo("   " + "-"*50)
    click.echo("   --performance-hint THROUGHPUT  (for max throughput)")
    click.echo("   --performance-hint LATENCY     (for min latency)")
    click.echo("")

    click.echo("5. CONSIDER LOWER PRECISION (if accuracy allows)")
    click.echo("   " + "-"*50)
    click.echo("   FP32 = highest accuracy, slowest")
    click.echo("   FP16 = good accuracy, 1.5-2x faster")
    click.echo("   INT8 = acceptable accuracy, 2-4x faster")
    click.echo("")

    click.echo("="*70)
    click.echo("RECOMMENDED COMMAND FOR MAXIMUM THROUGHPUT:")
    click.echo("="*70 + "\n")

    if model == 'resnet50':
        batch = 32
    elif model == 'bert':
        batch = 8
    elif model == 'retinanet':
        batch = 4
    else:
        batch = 2

    click.echo(f"  mlperf-ov run --model {model} \\")
    click.echo("    --scenario Offline \\")
    click.echo("    --mode performance \\")
    click.echo(f"    --batch-size {batch} \\")
    click.echo("    --num-streams AUTO \\")
    click.echo("    --performance-hint THROUGHPUT \\")
    click.echo("    --model-path ./models/<model> \\")
    click.echo("    --data-path ./data/<dataset>")
    click.echo("")

    click.echo("="*70)
    click.echo("WHY SERVER MODE IS SLOW:")
    click.echo("="*70 + "\n")
    click.echo("  Server mode simulates real-world server load with:")
    click.echo("  - Rate-limited query arrival (target_qps)")
    click.echo("  - Latency requirements (target_latency_ns)")
    click.echo("  - This INTENTIONALLY limits throughput!")
    click.echo("")
    click.echo("  Use Server mode only to test latency compliance,")
    click.echo("  NOT for maximum throughput benchmarks.")
    click.echo("")


@main.command('benchmark-throughput')
@click.option('--model-path', '-m', type=click.Path(exists=True), required=True,
              help='Path to model file')
@click.option('--batch-size', '-b', type=int, default=1,
              help='Batch size for inference')
@click.option('--num-streams', type=str, default='AUTO',
              help='Number of inference streams')
@click.option('--iterations', '-n', type=int, default=100,
              help='Number of iterations')
@click.option('--warmup', type=int, default=10,
              help='Number of warmup iterations')
@click.option('--num-threads', type=int, default=0,
              help='Number of threads (0 = auto)')
def benchmark_throughput(model_path: str, batch_size: int, num_streams: str,
                         iterations: int, warmup: int, num_threads: int):
    """
    Run quick throughput benchmark using async inference.

    This is the fastest way to measure your hardware's maximum throughput
    without LoadGen overhead.

    Example:

        mlperf-ov benchmark-throughput -m ./models/resnet50.onnx -b 32 -n 200
    """
    import time
    from .backends.openvino_backend import OpenVINOBackend
    from .core.config import OpenVINOConfig

    click.echo(f"\nLoading model: {model_path}")

    config = OpenVINOConfig(
        num_threads=num_threads,
        num_streams=num_streams,
        performance_hint="THROUGHPUT",
    )

    backend = OpenVINOBackend(model_path, config)
    backend.load()

    click.echo(f"Device: CPU")
    click.echo(f"Streams: {backend.num_streams}")
    click.echo(f"Batch size: {batch_size}")

    # Create dummy inputs
    dummy_inputs = []
    for _ in range(batch_size):
        inputs = {}
        for name, shape in backend.input_shapes.items():
            inputs[name] = np.random.randn(*shape).astype(np.float32)
        dummy_inputs.append(inputs)

    # Warmup
    click.echo(f"\nWarming up ({warmup} iterations)...")
    for _ in range(warmup):
        backend.predict_batch(dummy_inputs)

    # Benchmark with async inference
    click.echo(f"Running throughput benchmark ({iterations} iterations x {batch_size} batch)...")

    total_samples = 0
    start_time = time.perf_counter()

    for _ in range(iterations):
        backend.predict_batch(dummy_inputs)
        total_samples += batch_size

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    throughput = total_samples / elapsed
    latency_per_sample = (elapsed / total_samples) * 1000

    click.echo("\n" + "="*50)
    click.echo("THROUGHPUT RESULTS:")
    click.echo("="*50)
    click.echo(f"  Total samples:     {total_samples}")
    click.echo(f"  Total time:        {elapsed:.2f} seconds")
    click.echo(f"  Throughput:        {throughput:.2f} samples/sec")
    click.echo(f"  Latency/sample:    {latency_per_sample:.2f} ms")
    click.echo("="*50 + "\n")


@main.command('list-models')
def list_models():
    """List all supported models with their details."""
    click.echo("\n" + "="*70)
    click.echo("Supported Models for MLPerf v5.1 OpenVINO Benchmark")
    click.echo("="*70 + "\n")

    models = [
        {
            'name': 'ResNet50-v1.5',
            'id': 'resnet50',
            'task': 'Image Classification',
            'dataset': 'ImageNet 2012',
            'metric': 'Top-1 Accuracy',
            'target': '76.46%',
        },
        {
            'name': 'BERT-Large',
            'id': 'bert',
            'task': 'Question Answering',
            'dataset': 'SQuAD v1.1',
            'metric': 'F1 Score',
            'target': '90.874%',
        },
        {
            'name': 'RetinaNet',
            'id': 'retinanet',
            'task': 'Object Detection',
            'dataset': 'OpenImages',
            'metric': 'mAP',
            'target': '37.57%',
        },
        {
            'name': 'Whisper Large v3',
            'id': 'whisper',
            'task': 'Speech Recognition',
            'dataset': 'LibriSpeech',
            'metric': 'Word Accuracy',
            'target': '97.93%',
        },
    ]

    for m in models:
        click.echo(f"{m['name']} ({m['id']})")
        click.echo(f"  Task:    {m['task']}")
        click.echo(f"  Dataset: {m['dataset']}")
        click.echo(f"  Metric:  {m['metric']} (Target: {m['target']})")
        click.echo("")

    click.echo("Quick start:")
    click.echo("  mlperf-ov setup --model <model_id>")
    click.echo("  mlperf-ov run --model <model_id> --mode both")
    click.echo("")


if __name__ == "__main__":
    main()
