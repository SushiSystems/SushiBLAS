# test_device.py
# SushiBLAS - https://github.com/SushiSystems/SushiBLAS
# Copyright (c) 2026-present Mustafa Garip & Sushi Systems
# Licensed under PolyForm Noncommercial 1.0.0. See LICENSE.
# Commercial use requires a licence from Sushi Systems.
"""Tests for the mapping from a `--device` value to ONEAPI_DEVICE_SELECTOR.

See cli/tests/README.md.
"""

from __future__ import annotations

import pytest

from sushiblas.services.device import (
    K_DEVICE_SELECTOR,
    K_DEVICE_VARIABLE,
    DeviceEnvironment,
    device_selector,
    with_device,
)
from sushiblas.services.errors import K_EXIT_USAGE, InvalidDeviceError

K_CALLER_SELECTOR = "level_zero:0"


@pytest.mark.parametrize(
    ("device", "selector"),
    [("cpu", "opencl:cpu"), ("cuda", "cuda:*")],
)
def test_device_selector_maps_each_name(device, selector):
    """Each device name stands for one fixed selector."""
    assert device_selector(device) == selector


def test_device_selector_covers_every_name_in_the_table():
    """The cases above are the whole table, so a new name cannot arrive untested."""
    assert sorted(K_DEVICE_SELECTOR) == ["cpu", "cuda"]


def test_device_selector_ignores_the_case_of_a_name():
    """CPU and cpu name the same device."""
    assert device_selector("CPU") == "opencl:cpu"


@pytest.mark.parametrize(
    ("device", "selector"),
    [
        ("selector:level_zero:gpu", "level_zero:gpu"),
        ("selector:cuda:0;opencl:cpu", "cuda:0;opencl:cpu"),
        ("selector:*:GPU", "*:GPU"),
        ("selector: cuda:0 ", " cuda:0 "),
    ],
)
def test_device_selector_passes_the_text_after_the_prefix_through(device, selector):
    """selector:<text> yields <text> as written: colons, semicolons, case and blanks."""
    assert device_selector(device) == selector


@pytest.mark.parametrize(
    "device",
    ["gpu", "", "opencl:cpu", "selector:", "selector:  ", "Selector:x"],
)
def test_device_selector_rejects_an_unknown_value(device):
    """A value that is neither a name nor a non-blank passthrough is a usage error."""
    with pytest.raises(InvalidDeviceError) as raised:
        device_selector(device)
    assert raised.value.exit_code == K_EXIT_USAGE
    assert repr(device) in str(raised.value)
    assert "cpu, cuda or selector:<text>" in str(raised.value)


def test_with_device_leaves_the_environment_alone_without_a_device():
    """No --device and no variable: the child gets what it would have got."""
    env = {"PATH": "/bin"}
    assert with_device(env, None) == DeviceEnvironment({"PATH": "/bin"}, None)


def test_with_device_keeps_the_selector_of_the_caller_without_a_device():
    """No --device: a selector already in the environment stays, and nothing is reported."""
    env = {K_DEVICE_VARIABLE: K_CALLER_SELECTOR}
    assert with_device(env, None) == DeviceEnvironment(env, None)


def test_with_device_sets_the_selector_when_the_caller_had_none():
    """--device alone sets the variable and reports nothing replaced."""
    chosen = with_device({"PATH": "/bin"}, "cpu")
    assert chosen == DeviceEnvironment({"PATH": "/bin", K_DEVICE_VARIABLE: "opencl:cpu"}, None)


def test_with_device_outranks_the_selector_of_the_caller():
    """--device wins over the environment and hands back the value it replaced."""
    chosen = with_device({K_DEVICE_VARIABLE: K_CALLER_SELECTOR}, "cuda")
    assert chosen == DeviceEnvironment({K_DEVICE_VARIABLE: "cuda:*"}, K_CALLER_SELECTOR)


def test_with_device_does_not_modify_the_environment_it_was_given():
    """The caller's mapping is copied, so only the child sees the selector."""
    env = {K_DEVICE_VARIABLE: K_CALLER_SELECTOR}
    with_device(env, "cpu")
    assert env == {K_DEVICE_VARIABLE: K_CALLER_SELECTOR}


def test_with_device_rejects_an_unknown_value():
    """An invalid --device raises before any environment is produced."""
    with pytest.raises(InvalidDeviceError):
        with_device({}, "gpu")
