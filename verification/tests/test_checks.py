"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Tests of the verifier's checks on the v1.0.0 fixtures (see conftest.py).
"""

from manywells_verify.cases import Root
from manywells_verify.checks import Tolerances, check_convergence, verify_case


def unlabelled(roots, operating_point):
    return [Root(r.x, None, operating_point=(k == operating_point)) for k, r in enumerate(roots)]


def test_fixture_roots_are_the_known_ones(fixtures):
    _, roots = fixtures['sol1-977']
    assert [(round(r.x[0], 2), r.label) for r in roots] == [(169.0, 'stable'), (208.02, 'unstable')]
    assert fixtures['sol1-977-past-fold'][1] == []


def test_graph_label_matches_v1_dense_label(graph, fixtures):
    for case, roots in fixtures.values():
        res = verify_case(graph, case, unlabelled(roots, None), roots)
        for rr, ref in zip(res.roots, roots):
            assert rr.label == ref.label, case.case_id
    trickle = verify_case(graph, *fixtures['sol1-977']).roots[1]
    assert trickle.slope > 1e3


def test_v1_roots_pass_residuals_and_invariants(graph, fixtures):
    for case, roots in fixtures.values():
        res = verify_case(graph, case, roots, roots)
        for name in ('residuals', 'invariants', 'stability'):
            assert res.checks[name].status in ('pass', 'n/a'), (case.case_id, name, res.checks[name].detail)


def test_perturbed_root_fails_residuals(graph, fixtures):
    case, roots = fixtures['sol1-one-root']
    x = roots[0].x.copy()
    x[7 * 50] += 0.1                                  # 0.1 bar at point 50
    res = verify_case(graph, case, [Root(x, operating_point=True)], roots)
    assert res.checks['residuals'].failed
    assert 'momentum' in res.checks['residuals'].detail
    assert res.checks['operating_point'].failed       # an invalid root cannot be the operating point


def test_trickle_root_as_operating_point_fails(graph, fixtures):
    case, roots = fixtures['sol1-977']
    stable_pick = verify_case(graph, case, unlabelled(roots, 0), roots)
    trickle_pick = verify_case(graph, case, unlabelled(roots, 1), roots)
    assert stable_pick.checks['operating_point'].status == 'pass'
    assert trickle_pick.checks['operating_point'].failed
    assert 'nearest reference root is unstable' in trickle_pick.checks['operating_point'].detail
    assert stable_pick.n_stable_reference == 1


def test_root_set_and_labels(graph, fixtures):
    case, roots = fixtures['sol1-977']
    labelled = [Root(r.x, r.label, operating_point=(r.label == 'stable')) for r in roots]
    full = verify_case(graph, case, labelled, roots)
    assert full.checks['root_set'].status == 'pass'
    assert full.checks['stability'].status == 'pass'

    only_trickle = verify_case(graph, case, [Root(roots[1].x, 'unstable')], roots)
    assert only_trickle.checks['root_set'].failed
    assert 'stable root at p0 = 169.000' in only_trickle.checks['root_set'].detail

    swapped = [Root(roots[0].x, 'unstable', operating_point=True), Root(roots[1].x, 'stable')]
    wrong = verify_case(graph, case, swapped, roots)
    assert wrong.checks['stability'].failed
    assert wrong.checks['root_set'].failed

    unlabelled_set = verify_case(graph, case, unlabelled(roots, 0), roots)
    assert unlabelled_set.checks['root_set'].status == 'pass'


def test_extra_valid_root_is_a_finding(graph, fixtures):
    case, roots = fixtures['sol1-977']
    res = verify_case(graph, case, [Root(r.x, r.label) for r in roots], roots[:1])
    assert any('not in the reference root set' in f for f in res.findings)


def test_past_the_fold(graph, fixtures):
    case, roots = fixtures['sol1-977-past-fold']
    assert verify_case(graph, case, [], roots).checks['operating_point'].status == 'pass'
    _, stable_977 = fixtures['sol1-977']
    wrong = verify_case(graph, case, [Root(stable_977[0].x, operating_point=True)], roots)
    assert wrong.checks['operating_point'].failed
    assert wrong.checks['residuals'].failed           # p_r differs, so the inflow rows do not hold


def test_invariants(graph, fixtures):
    case, roots = fixtures['sol1-one-root']
    x = roots[0].x.copy()
    x[7 * 10 + 3] = 1.2                               # alpha at point 10
    assert 'alpha outside [0, 1]' in verify_case(graph, case, [Root(x)], roots).checks['invariants'].detail

    X = roots[0].x.reshape(-1, 7)
    choked = case.params['p_s'] <= case.params['cpr'] * X[-1, 0]
    ok = verify_case(graph, case, [Root(roots[0].x, choked=choked)], roots)
    bad = verify_case(graph, case, [Root(roots[0].x, choked=not choked)], roots)
    assert ok.checks['invariants'].status == 'pass'
    assert 'CHOKED' in bad.checks['invariants'].detail


def test_wrong_state_length(graph, fixtures):
    case, roots = fixtures['sol1-one-root']
    res = verify_case(graph, case, [Root(roots[0].x[:-7], operating_point=True)], roots)
    assert res.checks['residuals'].failed and 'length' in res.checks['residuals'].detail


def test_convergence_is_first_order(fixtures):
    members = [fixtures[f'conv-one-root-N{n}'] for n in (200, 50, 100)]   # order does not matter
    check = check_convergence([(case, roots[0]) for case, roots in members], Tolerances())
    assert check.status == 'pass', check.detail
    assert 0.9 < check.value < 1.1
    two = check_convergence([(case, roots[0]) for case, roots in members[:2]], Tolerances())
    assert two.status == 'n/a'
    missing = check_convergence([(case, None if k == 0 else roots[0]) for k, (case, roots) in enumerate(members)],
                                Tolerances())
    assert missing.failed
