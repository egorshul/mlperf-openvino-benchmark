"""
Device discovery utilities for accelerator devices.

Topology: 1 card = 2 dies. Devices appear as: NPU.0, NPU.1, NPU.2, etc.
"""

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Simulator suffixes (excluded from physical dies)
SIMULATOR_SUFFIXES = ("FuncSimulator", "Simulator")


def _get_die_pattern(device_prefix: str) -> re.Pattern:
    """Get regex pattern for device dies."""
    return re.compile(rf"^{re.escape(device_prefix)}\.(\d+)$")


def discover_accelerator_devices(core: "Core", device_prefix: str) -> List[str]:
    """Discover all available accelerator dies for a device prefix."""
    try:
        all_devices = core.available_devices
    except Exception as e:
        logger.error(f"Failed to get available devices: {e}")
        return []

    dies = []
    simulators_found = []
    pattern = _get_die_pattern(device_prefix)

    for device in all_devices:
        if not device.startswith(device_prefix):
            continue

        # Check if it's a simulator
        if is_simulator(device, device_prefix):
            simulators_found.append(device)
            continue

        # Check if it's a valid die (DEVICE.<number>)
        if pattern.match(device):
            dies.append(device)

    # Sort by die number for consistent ordering
    dies = sort_devices(dies, device_prefix)

    if simulators_found:
        logger.debug(f"Found {device_prefix} simulators (excluded): {simulators_found}")

    logger.debug(f"Discovered {len(dies)} {device_prefix} dies: {dies}")
    return dies


def is_simulator(device_name: str, device_prefix: str) -> bool:
    """Check if the device is a simulator."""
    if not device_name.startswith(device_prefix + "."):
        return False

    suffix = device_name[len(device_prefix) + 1:]
    return suffix in SIMULATOR_SUFFIXES


def is_accelerator_die(device_name: str, device_prefix: str) -> bool:
    """Check if the device is a valid accelerator die (DEVICE.N format)."""
    pattern = _get_die_pattern(device_prefix)
    return pattern.match(device_name) is not None


def get_die_number_for_device(device_name: str, device_prefix: str) -> Optional[int]:
    """Extract die number from device name."""
    pattern = _get_die_pattern(device_prefix)
    match = pattern.match(device_name)
    if match:
        return int(match.group(1))
    return None


def get_die_number(device_name: str) -> Optional[int]:
    """Extract die number (auto-detects prefix)."""
    if "." not in device_name:
        return None
    prefix = device_name.split(".")[0]
    return get_die_number_for_device(device_name, prefix)


def sort_devices(devices: List[str], device_prefix: str) -> List[str]:
    """Sort device names by die number."""
    def sort_key(device: str) -> int:
        num = get_die_number_for_device(device, device_prefix)
        return num if num is not None else float('inf')

    return sorted(devices, key=sort_key)


def get_device_info(core: "Core", device_name: str) -> dict:
    """Get information about a device."""
    prefix = device_name.split(".")[0] if "." in device_name else device_name

    info = {
        "name": device_name,
        "prefix": prefix,
        "is_die": is_accelerator_die(device_name, prefix) if "." in device_name else False,
        "is_simulator": is_simulator(device_name, prefix) if "." in device_name else False,
        "die_number": get_die_number(device_name),
    }

    try:
        info["full_name"] = core.get_property(device_name, "FULL_DEVICE_NAME")
    except Exception:
        info["full_name"] = device_name

    return info


def get_card_and_die(device_name: str) -> Optional[Tuple[int, int]]:
    """Get card and die indices (1 card = 2 dies). Returns (card_idx, die_on_card)."""
    die_num = get_die_number(device_name)
    if die_num is None:
        return None

    card_index = die_num // 2
    die_on_card = die_num % 2
    return (card_index, die_on_card)


def validate_accelerator_device(core: "Core", device_name: str) -> Tuple[bool, str]:
    """Validate that an accelerator device is available. Returns (is_valid, error_message)."""
    all_devices = core.available_devices

    # Determine prefix
    if "." in device_name:
        prefix = device_name.split(".")[0]
    else:
        prefix = device_name

    # Check if it's just the prefix (user wants all dies)
    if device_name == prefix:
        # User wants all devices of this type - check if any exist
        devices = discover_accelerator_devices(core, prefix)
        if not devices:
            available = ", ".join(all_devices)
            return False, f"No {prefix} devices found. Available devices: {available}"
        return True, ""

    # Check for specific die (PREFIX.<N>)
    if device_name.startswith(prefix + "."):
        if is_simulator(device_name, prefix):
            return False, f"'{device_name}' is a simulator, not a physical die"

        if not is_accelerator_die(device_name, prefix):
            return False, f"Invalid {prefix} device format: '{device_name}'. Expected {prefix}.<number>"

        # Check if device exists
        if device_name not in all_devices:
            available_dies = [d for d in all_devices
                            if d.startswith(prefix + ".") and is_accelerator_die(d, prefix)]
            if available_dies:
                return False, f"Device '{device_name}' not found. Available {prefix} dies: {available_dies}"
            else:
                return False, f"Device '{device_name}' not found. No {prefix} devices available."
        return True, ""

    return False, f"Invalid device specification: '{device_name}'"


def parse_device_properties(properties_str: str) -> dict:
    """Parse device properties from CLI string (KEY=VALUE,KEY2=VALUE2,...)."""
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
    """Validate device properties. Returns (is_valid, warnings)."""
    warnings = []

    common_props = {"NUM_STREAMS", "PERFORMANCE_HINT", "INFERENCE_NUM_THREADS",
                    "ENABLE_PROFILING", "CACHE_DIR"}
    accelerator_props = {"PERFORMANCE_MODE", "NUM_STREAMS", "OPTIMIZATION_LEVEL",
                         "EXECUTION_MODE_HINT"}
    cpu_props = {"AFFINITY", "ENABLE_CPU_PINNING", "INFERENCE_THREADS_PER_STREAM",
                 "ENABLE_HYPER_THREADING"}

    is_accelerator = device_type.upper() not in ("CPU", "GPU")

    for key in properties:
        if key in common_props:
            continue
        if is_accelerator and key in accelerator_props:
            continue
        if not is_accelerator and key in cpu_props:
            continue
        warnings.append(f"Unknown property '{key}' for device '{device_type}'")

    return True, warnings
