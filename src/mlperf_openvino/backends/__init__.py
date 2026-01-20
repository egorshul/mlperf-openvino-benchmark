"""Inference backends for MLPerf OpenVINO Benchmark."""

from .base import BaseBackend
from .openvino_backend import OpenVINOBackend
from .multi_device_backend import MultiDeviceBackend
from .device_discovery import (
    # Generic functions (preferred)
    discover_accelerator_devices,
    is_accelerator_die,
    is_simulator,
    validate_accelerator_device,
    get_card_and_die,
    get_die_number,
    parse_device_properties,
    # Backward compatible aliases
    discover_x_devices,
    is_x_die,
    is_x_simulator,
    validate_x_device,
)

__all__ = [
    "BaseBackend",
    "OpenVINOBackend",
    "MultiDeviceBackend",
    # Generic functions (preferred)
    "discover_accelerator_devices",
    "is_accelerator_die",
    "is_simulator",
    "validate_accelerator_device",
    "get_card_and_die",
    "get_die_number",
    "parse_device_properties",
    # Backward compatible aliases
    "discover_x_devices",
    "is_x_die",
    "is_x_simulator",
    "validate_x_device",
]
