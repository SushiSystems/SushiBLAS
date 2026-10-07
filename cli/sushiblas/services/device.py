# device.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Names the devices `--device` accepts and puts their selector in a child environment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .errors import InvalidDeviceError

K_DEVICE_VARIABLE = "ONEAPI_DEVICE_SELECTOR"
K_SELECTOR_PREFIX = "selector:"

#: Maps each device name to its ONEAPI_DEVICE_SELECTOR value; see cli/sushiblas/README.md.
K_DEVICE_SELECTOR = {
    "cpu": "opencl:cpu",
    "cuda": "cuda:*",
}


@dataclass(frozen=True, slots=True)
class DeviceEnvironment:
    """Holds a child environment after `--device` was applied to it.

    Args:
        variables: The environment the child process receives.
        replaced: The selector the caller had set and `--device` overrode; None when none was.
    """

    variables: dict[str, str]
    replaced: str | None


def device_selector(device: str) -> str:
    """Returns the ONEAPI_DEVICE_SELECTOR value *device* stands for.

    Args:
        device: A name of K_DEVICE_SELECTOR in any case, or `selector:<text>` for <text> verbatim.

    Raises:
        InvalidDeviceError: *device* is neither, or the text after `selector:` is blank.
    """
    if device.startswith(K_SELECTOR_PREFIX):
        selector = device[len(K_SELECTOR_PREFIX):]
        if not selector.strip():
            raise InvalidDeviceError(device, sorted(K_DEVICE_SELECTOR))
        return selector
    try:
        return K_DEVICE_SELECTOR[device.strip().lower()]
    except KeyError as error:
        raise InvalidDeviceError(device, sorted(K_DEVICE_SELECTOR)) from error


def with_device(env: Mapping[str, str], device: str | None) -> DeviceEnvironment:
    """Returns a copy of *env* whose device selector is the one *device* names.

    Args:
        env: The environment the child would otherwise receive; it is not modified.
        device: A `--device` value; None keeps whatever selector *env* holds.

    Raises:
        InvalidDeviceError: *device* is not a value `device_selector` accepts.
    """
    variables = dict(env)
    if device is None:
        return DeviceEnvironment(variables, None)
    selector = device_selector(device)
    replaced = variables.get(K_DEVICE_VARIABLE)
    variables[K_DEVICE_VARIABLE] = selector
    return DeviceEnvironment(variables, replaced)
