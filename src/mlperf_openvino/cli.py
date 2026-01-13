"""
Command Line Interface for MLPerf OpenVINO Benchmark.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import click

from .core.config import BenchmarkConfig, Scenario, TestMode
from .core.benchmark_runner import BenchmarkRunner
from .utils.model_downloader import download_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0", prog_name="mlperf-ov")
def main():
    """
    MLPerf v5.1 OpenVINO Benchmark Tool
    
    A benchmark tool for measuring CPU inference performance using OpenVINO,
    compatible with MLPerf Inference v5.1 specifications.
    """
    pass


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
@click.option('--performance-hint', type=click.Choice(['THROUGHPUT', 'LATENCY']),
              default='THROUGHPUT', help='Performance hint')
@click.option('--duration', type=int, default=60000,
              help='Minimum test duration in ms')
@click.option('--count', type=int, default=0,
              help='Number of samples to use (0 = all)')
@click.option('--warmup', type=int, default=10,
              help='Number of warmup iterations')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def run(model: str, scenario: str, mode: str, model_path: Optional[str],
        data_path: Optional[str], output_dir: str, config: Optional[str],
        num_threads: int, num_streams: str, performance_hint: str,
        duration: int, count: int, warmup: int, verbose: bool):
    """
    Run MLPerf benchmark.
    
    Example usage:
    
        mlperf-ov run --model resnet50 --scenario Offline --mode performance
        
        mlperf-ov run -m resnet50 -s Server --model-path ./models/resnet50.onnx
        
        mlperf-ov run -m whisper --data-path ./data/librispeech
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
        click.echo("Using default configuration")
        if model == 'whisper':
            benchmark_config = BenchmarkConfig.default_whisper()
        else:
            benchmark_config = BenchmarkConfig.default_resnet50()
    
    # Override with CLI options
    benchmark_config.scenario = Scenario(scenario)
    benchmark_config.results_dir = output_dir
    
    if num_threads > 0:
        benchmark_config.openvino.num_threads = num_threads
    benchmark_config.openvino.num_streams = num_streams
    benchmark_config.openvino.performance_hint = performance_hint
    
    if model_path:
        benchmark_config.model.model_path = model_path
    
    if data_path:
        benchmark_config.dataset.path = data_path
    
    if count > 0:
        benchmark_config.dataset.num_samples = count
    
    # Update duration
    scenario_config = benchmark_config.get_scenario_config()
    scenario_config.min_duration_ms = duration
    
    # Validate configuration
    if not benchmark_config.model.model_path:
        click.echo("Error: Model path is required. Use --model-path or download the model first.")
        click.echo(f"Run: mlperf-ov download --model {model}")
        sys.exit(1)
    
    if not Path(benchmark_config.dataset.path).exists():
        click.echo(f"Error: Dataset path does not exist: {benchmark_config.dataset.path}")
        if model == 'whisper':
            click.echo("Please download the LibriSpeech dataset.")
        else:
            click.echo("Please download the ImageNet validation dataset.")
        sys.exit(1)
    
    # Print configuration summary
    click.echo(f"Model: {benchmark_config.model.name}")
    click.echo(f"Scenario: {benchmark_config.scenario.value}")
    click.echo(f"Device: {benchmark_config.openvino.device}")
    click.echo(f"Threads: {benchmark_config.openvino.num_threads or 'auto'}")
    click.echo(f"Streams: {benchmark_config.openvino.num_streams}")
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
    
    Example:
    
        mlperf-ov download --model resnet50 --output-dir ./models
        
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


@main.command('download-dataset')
@click.option('--dataset', '-d', type=click.Choice(['imagenet', 'librispeech']),
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
    
        # Download ImageNet validation subset (1000 images)
        mlperf-ov download-dataset --dataset imagenet
        
        # Download LibriSpeech dev-clean
        mlperf-ov download-dataset --dataset librispeech --subset dev-clean
        
        # Download LibriSpeech test-clean
        mlperf-ov download-dataset --dataset librispeech --subset test-clean
    """
    from .utils.dataset_downloader import download_dataset, get_dataset_info
    
    click.echo(f"\nDownloading {dataset} dataset...")
    
    # Show dataset info
    info = get_dataset_info(dataset)
    click.echo(f"Description: {info['description']}")
    
    if subset:
        if subset in info:
            subset_info = info[subset]
            click.echo(f"Subset: {subset}")
            click.echo(f"Size: ~{subset_info.get('size_mb', 'unknown')} MB")
            click.echo(f"Samples: {subset_info.get('num_samples', 'unknown')}")
        else:
            click.echo(f"Warning: Unknown subset '{subset}', using default")
    
    click.echo("")
    
    try:
        paths = download_dataset(dataset, output_dir, subset, force)
        
        click.echo("\nDataset downloaded successfully!")
        click.echo(f"  Path: {paths['data_path']}")
        
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
@click.option('--model', '-m', type=click.Choice(['resnet50', 'whisper']),
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
        elif model == 'whisper':
            dataset_paths = download_dataset('librispeech', str(data_dir), 'dev-clean')
        
        click.echo(f"  Dataset: {dataset_paths['data_path']}\n")
    except Exception as e:
        click.echo(f"  Error: {e}")
        sys.exit(1)
    
    # Print usage instructions
    click.echo(f"{'='*60}")
    click.echo("Setup complete! Run benchmark with:")
    click.echo(f"{'='*60}\n")
    
    if model == 'resnet50':
        click.echo(f"  mlperf-ov run --model resnet50 \\")
        click.echo(f"    --model-path {model_path} \\")
        click.echo(f"    --data-path {dataset_paths['data_path']}")
    elif model == 'whisper':
        click.echo(f"  mlperf-ov run --model whisper \\")
        click.echo(f"    --model-path {model_path} \\")
        click.echo(f"    --data-path {dataset_paths['data_path']}")
    
    click.echo("")


if __name__ == "__main__":
    main()
