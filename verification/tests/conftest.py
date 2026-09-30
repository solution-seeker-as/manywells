"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Shared fixtures: the frozen v1.0.0 graph, and v1.0.0 roots from verification/build/make_test_fixtures.py
(sol-1 well 977 with its stable root at p0 = 169.00 bar and trickle root at 208.02 bar, a one-root
well, well 977 past the fold with no root, and a convergence group at N = 50, 100, 200).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from manywells_verify.cases import Case, Root
from manywells_verify.residual import ResidualGraph, Variant

FIXTURES = Path(__file__).parent / 'data' / 'fixtures.npz'


@pytest.fixture(scope='session')
def graph():
    return ResidualGraph('v1.0.0')


@pytest.fixture(scope='session')
def fixtures():
    """case_id -> (Case, reference roots with v1's dense-Jacobian labels)."""
    with np.load(FIXTURES) as data:
        out = {}
        for e in json.loads(str(data['meta'])):
            case = Case(e['case_id'], e['params'], e['n_cells'], Variant(*e['variant']), group=e['group'])
            out[e['case_id']] = case, [Root(data[r['key']], r['label']) for r in e['roots']]
        return out
