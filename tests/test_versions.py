"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

A release has one version number, shared by the package, the Rust core and the verifier (constitution, Versions;
AGENTS.md, Releases). Cargo spells a pre-release with a hyphen, 2.0.0-dev.0 for 2.0.0.dev0; packaging reads both.
"""

import pathlib
import tomllib

from packaging.version import Version

ROOT = pathlib.Path(__file__).resolve().parents[1]


def version(path: str, table: str) -> Version:
    return Version(tomllib.loads((ROOT / path).read_text())[table]['version'])


def test_one_version():
    package = version('pyproject.toml', 'project')
    assert version('verification/pyproject.toml', 'project') == package
    assert version('rust/Cargo.toml', 'package') == package
