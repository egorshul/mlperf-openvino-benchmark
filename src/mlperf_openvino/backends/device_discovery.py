"""
Device discovery utilities for X accelerator devices.

This module provides functions to discover and manage X neural network
accelerator devices in an OpenVINO-based inference system.

X Device Topology:
- 1 card = 2 dies (compute units)
- Devices appear as: X.0, X.1, X.2, X.3, ...
- Simulators appear as: X.FuncSimulator, X.Simulator

Example:
    >>> from openvino import Core
    >>> core = Core()
    >>> devices = discover_x_devices(core)
    >>> print(devices)  # ['X.0', 'X.1', 'X.2', 'X.3']
"""

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Device name constants
X_DEVICE_PREFIX = "X"
X_SIMULATOR_SUFFIXES = ("FuncSimulator", "Simulator")

# Pattern for valid X die: X.<number>
X_DIE_PATTERN = re.compile(r"^X\.(\d+)$")


def discover_x_devices(core: "Core") -> List[str]:
    """
    Discover all available X accelerator dies.

    Queries OpenVINO Core for available devices and filters for
    physical X dies (X.0, X.1, etc.), excluding simulators.

    Args:
        core: OpenVINO Core instance

    Returns:
        Sorted list of X device names (e.g., ['X.0', 'X.1', 'X.2', 'X.3'])

    Example:
        >>> from openvino import Core
        >>> core = Core()
        >>> dies = discover_x_devices(core)
        >>> print(f"Found {len(dies)} X dies: {dies}")
    """
    try:
        all_devices = core.available_devices
    except Exception as e:
        logger.error(f"Failed to get available devices: {e}")
        return []

    x_dies = []
    simulators_found = []

    for device in all_devices:
        if not device.startswith(X_DEVICE_PREFIX):
            continue

        # Check if it's a simulator
        if is_x_simulator(device):
            simulators_found.append(device)
            continue

        # Check if it's a valid die (X.<number>)
        if is_x_die(device):
            x_dies.append(device)

    # Sort by die number for consistent ordering
    x_dies = sort_x_devices(x_dies)

    if simulators_found:
        logger.debug(f"Found X simulators (excluded): {simulators_found}")

    logger.info(f"Discovered {len(x_dies)} X dies: {x_dies}")
    return x_dies


def is_x_simulator(device_name: str) -> bool:
    """
    Check if the device is an X simulator.

    Args:
        device_name: Device name from OpenVINO

    Returns:
        True if device is a simulator (X.FuncSimulator or X.Simulator)
    """
    if not device_name.startswith(X_DEVICE_PREFIX + "."):
        return False

    suffix = device_name[len(X_DEVICE_PREFIX) + 1:]
    return suffix in X_SIMULATOR_SUFFIXES


def is_x_die(device_name: str) -> bool:
    """
    Check if the device is a valid X die.

    Valid dies have format X.<number>, e.g., X.0, X.1, X.2

    Args:
        device_name: Device name from OpenVINO

    Returns:
        True if device is a valid X die
    """
    return X_DIE_PATTERN.match(device_name) is not None


def get_die_number(device_name: str) -> Optional[int]:
    """
    Extract die number from device name.

    Args:
        device_name: Device name (e.g., 'X.2')

    Returns:
        Die number (e.g., 2) or None if not a valid die
    """
    match = X_DIE_PATTERN.match(device_name)
    if match:
        return int(match.group(1))
    return None


def sort_x_devices(devices: List[str]) -> List[str]:
    """
    Sort X device names by die number.

    Args:
        devices: List of X device names

    Returns:
        Sorted list of device names
    """
    def sort_key(device: str) -> int:
        num = get_die_number(device)
        return num if num is not None else float('inf')

    return sorted(devices, key=sort_key)


def get_x_device_info(core: "Core", device_name: str) -> dict:
    """
    Get detailed information about an X device.

    Args:
        core: OpenVINO Core instance
        device_name: Device name (e.g., 'X.0')

    Returns:
        Dictionary with device properties
    """
    info = {
        "name": device_name,
        "is_die": is_x_die(device_name),
        "is_simulator": is_x_simulator(device_name),
        "die_number": get_die_number(device_name),
    }

    try:
        info["full_name"] = core.get_property(device_name, "FULL_DEVICE_NAME")
    except Exception:
        info["full_name"] = device_name

    return info


def get_card_and_die(device_name: str) -> Optional[Tuple[int, int]]:
    """
    Get card and die indices from device name.

    X device topology: 1 card = 2 dies
    - X.0, X.1 -> Card 0
    - X.2, X.3 -> Card 1
    - etc.

    Args:
        device_name: Device name (e.g., 'X.2')

    Returns:
        Tuple of (card_index, die_index_on_card) or None
        Example: X.2 -> (1, 0) meaning Card 1, Die 0 on that card
    """
    die_num = get_die_number(device_name)
    if die_num is None:
        return None

    card_index = die_num // 2
    die_on_card = die_num % 2
    return (card_index, die_on_card)


def validate_x_device(core: "Core", device_name: str) -> Tuple[bool, str]:
    """
    Validate that an X device is available and usable.

    Args:
        core: OpenVINO Core instance
        device_name: Device name to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check basic format
    if device_name == "X":
        # User wants all X devices - check if any exist
        devices = discover_x_devices(core)
        if not devices:
            return False, "No X devices found. Available devices: " + ", ".join(core.available_devices)
        return True, ""

    # Check for specific die (X.<N>)
    if device_name.startswith("X."):
        if is_x_simulator(device_name):
            return False, f"'{device_name}' is a simulator, not a physical die"

        if not is_x_die(device_name):
            return False, f"Invalid X device format: '{device_name}'. Expected X.<number>"

        # Check if device exists
        all_devices = core.available_devices
        if device_name not in all_devices:
            available_x = [d for d in all_devices if d.startswith("X.") and is_x_die(d)]
            if available_x:
                return False, f"Device '{device_name}' not found. Available X dies: {available_x}"
            else:
                return False, f"Device '{device_name}' not found. No X devices available."
        return True, ""

    return False, f"Invalid device specification: '{device_name}'"


def parse_device_properties(properties_str: str) -> dict:
    """
    Parse device properties from CLI string format.

    Format: "KEY=VALUE" or "KEY=VALUE,KEY2=VALUE2,..."

    Args:
        properties_str: Properties string from CLI

    Returns:
        Dictionary of property key-value pairs

    Example:
        >>> parse_device_properties("PERFORMANCE_MODE=THROUGHPUT,NUM_STREAMS=4")
        {'PERFORMANCE_MODE': 'THROUGHPUT', 'NUM_STREAMS': '4'}
    """
    if not properties_str:
        return {}

    properties = {}
    pairs = properties_str.split(",")

    for pair in pairs:
        pair = pair.strip()
        if not pair:
            continue

        if "=" not in pair:
            logger.warning(f"Invalid property format (missing '='): '{pair}'")
            continue

        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            logger.warning(f"Empty property key in: '{pair}'")
            continue

        properties[key] = value

    return properties


def validate_device_properties(properties: dict, device_type: str) -> Tuple[bool, List[str]]:
    """
    Validate device properties for a given device type.

    Args:
        properties: Dictionary of properties to validate
        device_type: Device type ("CPU", "X", or specific like "X.0")

    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []

    # Common valid properties
    common_props = {
        "NUM_STREAMS",
        "PERFORMANCE_HINT",
        "INFERENCE_NUM_THREADS",
        "ENABLE_PROFILING",
        "CACHE_DIR",
    }

    # X-specific properties
    x_props = {
        "PERFORMANCE_MODE",
        "NUM_STREAMS",
        "OPTIMIZATION_LEVEL",
        "EXECUTION_MODE_HINT",
    }

    # CPU-specific properties
    cpu_props = {
        "AFFINITY",
        "ENABLE_CPU_PINNING",
        "INFERENCE_THREADS_PER_STREAM",
        "ENABLE_HYPER_THREADING",
    }

    is_x_device = device_type.startswith("X")

    for key in properties:
        if key in common_props:
            continue
        if is_x_device and key in x_props:
            continue
        if not is_x_device and key in cpu_props:
            continue

        # Unknown property - warn but don't fail
        warnings.append(f"Unknown property '{key}' for device '{device_type}'")

    return True, warnings
