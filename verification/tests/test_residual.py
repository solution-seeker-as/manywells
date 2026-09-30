"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Tests of the frozen v1.0.0 residual graph. The golden vectors were computed by
verification/build/build_graph.py in the v1.0.0 environment (casadi 3.6.4), so these tests
also catch changes from a newer CasADi, and check the generated C against the same values.
"""

import ctypes
import shutil
import subprocess

import numpy as np
import pytest

from manywells_verify.residual import GRAPH_ROOT, ResidualGraph, Variant, choke_row

GRAPH_DIR = GRAPH_ROOT / 'v1.0.0'
RTOL = 1e-13


@pytest.fixture(scope='module')
def golden():
    with np.load(GRAPH_DIR / 'golden.npz') as data:
        return dict(data)


def assert_close(actual, expected):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    ok = ~np.isnan(expected)
    scale = np.maximum(1.0, np.abs(expected[ok]))
    assert np.max(np.abs(actual[ok] - expected[ok]) / scale, initial=0.0) <= RTOL


def golden_cases(golden, name):
    n_in = sum(1 for k in golden if k.startswith(f'{name}.in'))
    n_out = sum(1 for k in golden if k.startswith(f'{name}.out'))
    inputs = [golden[f'{name}.in{j}'] for j in range(n_in)]
    outputs = [golden[f'{name}.out{j}'] for j in range(n_out)]
    return inputs, outputs


def test_manifest_lists_every_block(graph):
    names = set(graph.manifest['blocks'])
    assert names == {p.stem for p in GRAPH_DIR.glob('*.casadi')}
    assert graph.manifest['built_with']['casadi'] == '3.6.4'


@pytest.mark.parametrize('name', sorted(p.stem for p in GRAPH_DIR.glob('*.casadi')))
def test_blocks_match_golden_vectors(graph, golden, name):
    f = graph._blocks[name]
    inputs, outputs = golden_cases(golden, name)
    assert len(inputs[0]) >= 2
    for k in range(len(inputs[0])):
        result = f(*[inp[k] for inp in inputs])
        for j, expected in enumerate(outputs):
            assert_close(np.array(result[j]), expected[k])


def stack_cases(golden):
    k = 0
    while f'stack{k}.x' in golden:
        yield {key.split('.', 1)[1]: golden[key] for key in golden if key.startswith(f'stack{k}.')}
        k += 1


def test_stacked_residual_matches_v1(graph, golden):
    cases = list(stack_cases(golden))
    assert {int(c['N']) for c in cases} == {100, 200}
    for c in cases:
        N, variant = int(c['N']), Variant(*[str(s) for s in c['variant']])
        r, J = graph.evaluate(c['x'], c['P'], N, variant)
        assert_close(r, c['r'])
        assert_close(J @ c['v'], c['Jv'])
        assert J.shape == (7 * (N + 1),) * 2


def test_v1_solution_satisfies_frozen_residual(graph, golden):
    for c in stack_cases(golden):
        if str(c['kind']) != 'solution':
            continue
        N, variant = int(c['N']), Variant(*[str(s) for s in c['variant']])
        r = graph.evaluate(c['x'], c['P'], N, variant, jacobian=False)
        assert np.all(np.isfinite(r))
        assert np.abs(r[choke_row(N)]) < 1e-6


def test_evaluate_rejects_wrong_state_length(graph, golden):
    c = next(stack_cases(golden))
    with pytest.raises(ValueError):
        graph.evaluate(c['x'][:-7], c['P'], int(c['N']))


# Generated C, called through ctypes: the same golden vectors must come out
CASADI_INT = ctypes.c_longlong
REAL_P = ctypes.POINTER(ctypes.c_double)


@pytest.fixture(scope='module')
def c_library(tmp_path_factory):
    cc = shutil.which('cc') or shutil.which('gcc')
    if cc is None:
        pytest.skip('no C compiler')
    lib = tmp_path_factory.mktemp('c') / 'residual.so'
    source = GRAPH_DIR / graph_c_file()
    subprocess.run([cc, '-O2', '-shared', '-fPIC', '-o', str(lib), str(source), '-lm'], check=True)
    return ctypes.CDLL(str(lib))


def graph_c_file():
    return ResidualGraph('v1.0.0').manifest['c_source']['file']


def call_c(lib, c_name, inputs, out_shapes):
    fn = getattr(lib, c_name)
    work = getattr(lib, f'{c_name}_work')
    sizes = [CASADI_INT() for _ in range(4)]
    assert work(*[ctypes.byref(s) for s in sizes]) == 0
    n_arg, n_res, n_iw, n_w = (s.value for s in sizes)

    in_bufs = [np.ascontiguousarray(np.atleast_1d(np.asarray(v, dtype=float)).ravel()) for v in inputs]
    out_bufs = [np.zeros(int(np.prod(shape))) for shape in out_shapes]
    arg = (REAL_P * n_arg)(*[b.ctypes.data_as(REAL_P) for b in in_bufs])
    res = (REAL_P * n_res)(*[b.ctypes.data_as(REAL_P) for b in out_bufs])
    iw = (CASADI_INT * max(n_iw, 1))()
    w = (ctypes.c_double * max(n_w, 1))()
    assert fn(arg, res, iw, w, 0) == 0
    # CasADi stores matrices column by column
    return [b.reshape(shape[::-1]).T if len(shape) == 2 else b for b, shape in zip(out_bufs, out_shapes)]


@pytest.mark.parametrize('name', sorted(p.stem for p in GRAPH_DIR.glob('*.casadi')))
def test_generated_c_matches_golden_vectors(c_library, graph, golden, name):
    c_name = graph.manifest['blocks'][name]['c_name']
    inputs, outputs = golden_cases(golden, name)
    for k in range(len(inputs[0])):
        shapes = [expected[k].shape for expected in outputs]
        result = call_c(c_library, c_name, [inp[k] for inp in inputs], shapes)
        for got, expected in zip(result, outputs):
            assert_close(got.reshape(expected[k].shape), expected[k])
