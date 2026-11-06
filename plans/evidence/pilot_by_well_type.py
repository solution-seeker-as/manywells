"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 05 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

R6's sol-2 pilot by well type, for the sampler redesign's checkpoint (D1, D2, F1): the core wells' median-sample
rates against the rate brackets of the per-well analyses kept outside the repo, and the
lifted share. OUT is the pilot's output folder of sampler_prototype.py (wells.parquet, samples.parquet).

    .venv/bin/python plans/evidence/pilot_by_well_type.py OUT
"""

import sys
from pathlib import Path

import pandas as pd

OILS = ('black_oil', 'volatile_oil')
GOR_OIL_WELL = 1068.0  # Sm3/Sm3, as sampler_prototype_report.py's oil wells


def main(out):
    s = pd.read_parquet(out / 'samples.parquet')
    w = pd.read_parquet(out / 'wells.parquet').set_index('ID')
    s = s.assign(q_oe=24 * (s.QOIL + s.QGAS / 1000), q_g=24 * s.QGAS)
    w['q_oe'], w['q_g'] = s.groupby('ID').q_oe.median(), s.groupby('ID').q_g.median()
    core = w[w.coverage_tail == 'core']
    print('Core wells by well type: median-sample rates, Sm3/d (oil equivalent for oil wells, gas for gas wells)')
    for t in ('platform', 'subsea', 'onshore'):
        o = core[core.fluid_type.isin(OILS) & (core.well_type == t) & (core.gor_nom <= GOR_OIL_WELL)]
        g = core[~core.fluid_type.isin(OILS) & (core.well_type == t)]
        lifted = core[core.fluid_type.isin(OILS) & (core.well_type == t)].lift.eq('gas_lift').mean()
        print(
            f'  {t}: oil wells {len(o)}, P50 {o.q_oe.median():.0f}, above 1018 {(o.q_oe > 1018).mean():.0%}, '
            f'below 16 {(o.q_oe < 16).mean():.0%}, oil wells lifted {lifted:.0%}; gas wells {len(g)}, '
            f'P50 {g.q_g.median():.3g}, above 270,000 {(g.q_g > 2.7e5).mean():.0%}'
        )


if __name__ == '__main__':
    main(Path(sys.argv[1]))
