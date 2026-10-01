"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

develop's root sets on the verifier's case set, in the v1.0.0 configuration (specs/verification.md), as a
candidate file for manywells-verify. Run from the project root:

    uv run python -m scripts.verification.develop_candidate verification/data/develop.parquet
    uv run manywells-verify verification/data/develop.parquet --data verification/data \
        --expected-failures verification/expected_failures.csv --name "develop (v1.0.0 configuration)"

Each case is mapped to develop's inputs by manywells.configurations.v1_well, and every root the search finds is
written with its label, CHOKED flag and whether it is the operating point. It imports both manywells and
manywells_verify, so it is a script, not part of either package (specs/architecture.md).
"""

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

from manywells_verify.cases import Root as CandidateRoot, read_cases, write_roots

from manywells.choke import BernoulliChokeModel, SimpsonChokeModel
from manywells.configurations import v1_well
from manywells.inflow import ProductivityIndex, Vogel
from manywells.simulator import BoundaryConditions, SSDFSimulator

DATA = Path('verification/data')


def well_of(case):
    """develop's WellProperties and BoundaryConditions in the v1.0.0 configuration for a verifier case."""
    prm, variant = case.params, case.variant
    inflow = Vogel(w_l_max=prm['w_l_max']) if variant.inflow == 'vogel' else ProductivityIndex(k_l=prm['k_l'])
    choke = (SimpsonChokeModel if variant.choke == 'simpson' else BernoulliChokeModel)(K_c=prm['K_c'],
                                                                                     chk_profile=variant.profile)
    if abs(choke.cpr - prm['cpr']) > 1e-12:
        raise ValueError(f'{case.case_id}: cpr {prm["cpr"]} differs from develop\'s {choke.cpr}')
    wp = v1_well(L=prm['L'], D=prm['D'], rho_l=prm['rho_l'], R_s=prm['R_s'], cp_g=prm['cp_g'], cp_l=prm['cp_l'],
                 f_D=prm['f_D'], h=prm['h'], f_g=prm['f_g'], inflow=inflow, choke=choke, n_cells=case.n_cells)
    bc = BoundaryConditions(p_r=prm['p_r'], p_s=prm['p_s'], T_r=prm['T_r'], T_s=prm['T_s'], u=prm['u'], w_lg=prm['w_lg'])
    return wp, bc


def solve_case(case):
    """The root set of one case, as candidate roots, and timings (s)."""
    wp, bc = well_of(case)
    t0 = time.perf_counter()
    sim = SSDFSimulator(wp)
    t1 = time.perf_counter()
    rs = sim.root_set(bc)
    t2 = time.perf_counter()
    roots = [CandidateRoot(r.x, r.label, operating_point=r is rs.operating_point, choked=r.choked,
                           info={'slope': r.slope}) for r in rs.roots]
    search = [{'start': a.start, 'p_0': a.p_0, 'outcome': a.outcome, 'seconds': a.seconds, 'iterations': a.iterations} for a in rs.search]
    return case.case_id, roots, {'build': t1 - t0, 'search': t2 - t1, 'attempts': search}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[1])
    parser.add_argument('out', type=Path, help='candidate parquet file to write')
    parser.add_argument('--data', type=Path, default=DATA, help='directory with cases.parquet')
    parser.add_argument('--cases', nargs='*', help='only these case ids')
    parser.add_argument('--processes', type=int, default=os.cpu_count())
    parser.add_argument('--log', type=Path, help='JSON file for timings and every start of every search')
    args = parser.parse_args()

    cases = read_cases(args.data / 'cases.parquet')
    todo = [c for k, c in cases.items() if not args.cases or k in args.cases]
    t0 = time.perf_counter()
    with multiprocessing.Pool(args.processes) as pool:
        results = pool.map(solve_case, todo, chunksize=1)
    wall = time.perf_counter() - t0

    write_roots({cid: roots for cid, roots, _ in results}, args.out)
    build = sum(r[2]['build'] for r in results)
    search = sum(r[2]['search'] for r in results)
    print(f'{len(results)} cases in {wall:.1f} s wall time ({args.processes} processes); per case: '
          f'build {build / len(results):.2f} s, search {search / len(results):.2f} s')
    if args.log:
        args.log.write_text(json.dumps({cid: info for cid, _, info in results}, indent=1))


if __name__ == '__main__':
    main()
