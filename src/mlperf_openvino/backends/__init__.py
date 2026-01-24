"""Inference backends for MLPerf OpenVINO Benchmark."""

from .base import BaseBackend
from .openvino_backend import OpenVINOBackend
from .multi_device_backend import MultiDeviceBackend
from .device_discovery import (
    discover_accelerator_devices,
    is_accelerator_die,
    is_simulator,
    validate_accelerator_device,
    get_card_and_die,
    get_die_number,
    parse_device_properties,
)

__all__ = [
    "BaseBackend",
    "OpenVINOBackend",
    "MultiDeviceBackend",
    "discover_accelerator_devices",
    "is_accelerator_die",
    "is_simulator",
    "validate_accelerator_device",
    "get_card_and_die",
    "get_die_number",
    "parse_device_properties",
]
