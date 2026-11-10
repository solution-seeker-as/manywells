"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The files of a v2 dataset, manywells-sol-2 or manywells-nsol-2 (specs/sampling.md, SMP-108 and SMP-110;
plans/sampler-redesign/datasets-draft.md, Sections 2 and 3): the labels' codes and the columns' types, the records
of the generation (the code version, the lock file's hash, the platform), the writer of a folder of tables with its
metadata.json, and the re-solve of a sample from the dataset's rows alone. The tables themselves are assembled from a
run's results by manywells.sampling.v2.assemble.

A dataset is a folder of four tables, `wells`, `samples`, `geometry` and `pvt`, with its logs `drawn_wells` and
`redraws` in a companion folder next to it, each with a metadata.json. A sample's row, its well's row, the well's
`geometry` rows and, for a table fluid, its table's `pvt` rows give the sample's WellProperties and
BoundaryConditions in the v2.0.0 configuration (sample_case), and the Rust core solves them again to the sample's
features (resolve_sample).
"""

import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from manywells.choke import CHOKE_PROFILES, SimpsonChokeModel
from manywells.configurations import V2, V2_SLIP, check
from manywells.datasets import schema
from manywells.datasets.io import code_version
from manywells.datasets.rows import root_features
from manywells.friction import RoughnessFriction
from manywells.geometry import WellGeometry
from manywells.inflow import Forchheimer, ProductivityIndex, Vogel
from manywells.pvt.black_oil import BlackOil
from manywells.pvt.fluid import FluidModel
from manywells.pvt.table import OPTIONAL_COLUMNS, TEXT_KEYS, PVTTable, PVTTableSet
from manywells.simulator import BoundaryConditions, SSDFSimulator, WellProperties
from manywells.slip import HASAN_REGIMES
from manywells.thermal import ThermalModel

_DATASETS = {'sol': 'manywells-sol-2', 'nsol': 'manywells-nsol-2'}
_TABLES = ('wells', 'samples', 'geometry', 'pvt')  # A dataset's tables  # spec: SMP-110
_LOGS = ('drawn_wells', 'redraws')  # Its logs, in the companion folder  # spec: SMP-108
# The operating point's columns of a sample: SMP-30's features but the inputs
FEATURES = tuple(key for key in schema.FEATURES if key not in ('CHK', 'PDC', 'WGL'))
# A sample's inputs that vary, by the attempt's key: both datasets, and nsol-2's besides
_SAMPLE_INPUTS = {'u': 'CHK', 'p_s': 'PDC', 'w_lg': 'WGL', 'p_r': 'PRES', 'gor': 'GOR', 'wlr': 'WLR'}
_NSOL_INPUTS = {'lift_on': 'LIFT', 'h': 'HTC', 'roughness': 'ROUGH', 'k_l': 'KL', 'w_l_max': 'WLMAX'}
_NSOL_INPUTS |= {'forchheimer_a': 'FORCHA', 'forchheimer_b': 'FORCHB'}
_NSOL_TIME = {'life_stage': 'STAGE', 'test': 'TEST', 'sweep': 'SWEEP'}  # With WEEKS, the date in weeks
_WEEKS_PER_YEAR = 52.1775
_STAGES = ('initial', 'mid_life', 'late')
_REASONS = ('p_r_le_p_s', 'no_root', 'exception', 'invalid_row', 'incomplete_scan', 'unstable_only')  # spec: SMP-104
_WELL_OUTCOMES = (  # spec: SMP-106; a crash of the generator is logged as a well's outcome too
    'accepted',
    'not_producible',
    'best_rate_solver_failure',
    'invalid_input',
    'slot_limit',
    'redraw_limit',
    'solver_failure_limit',
    'dead_date',
    'short_life',
    'table_failed',
    'crash',
)
# Each label's values, in the order of their codes (datasets-draft.md, Section 4, with SMP-52's values)
_CODES = {
    'coverage_tail': ('core', 'tail'),
    'well_type': ('platform', 'subsea', 'onshore'),
    'fluid_type': ('black_oil', 'volatile_oil', 'gas_condensate', 'gas'),
    'fluid_class': ('black_oil', 'volatile_oil', 'gas_condensate', 'wet_gas', 'dry_gas'),
    'class_rule': ('mccain_1990',),
    'drive': ('depletion', 'pressure_support'),
    'life_stage': _STAGES,
    'STAGE': _STAGES,
    'life_state': ('initial', 'mid_life', 'depleted', 'watered_out'),
    'pressure_class': ('under', 'normal', 'over'),
    'saturation': ('undersaturated', 'saturated', 'single_phase_gas'),
    'geometry_class': ('vertical', 'slightly_deviated', 'deviated', 'highly_deviated'),
    'geometry_family': ('vertical', 'build_and_hold', 's_shaped', 'high_angle'),
    'pvt_model': ('black_oil', 'table'),
    'fluid_source': (
        'synthetic',
        'wb2000_gas_condensate',
        'pedersen2015_t1_3',
        'pedersen2015_t3_8',
        'wb2000_near_critical_oil',
        'wb2000_volatile_oil',
        'pedersen2015_t1_4',
    ),
    'saturation_kind': ('bubble', 'dew', 'none'),
    'saturation_rule': ('matched', 'gor_moved', 'equilibrium_phase'),
    'inflow': ('vogel', 'productivity_index', 'forchheimer'),
    'tubing': ('2-7/8', '3-1/2', '4-1/2', '5-1/2', '7', '9-5/8'),
    'choke_model': ('simpson',),
    'chk_profile': CHOKE_PROFILES,
    'lift': ('natural', 'gas_lift'),
    'lift_start_rule': ('from_start', 'static_check', 'end_of_natural_flow', 'drawn_time', 'no_natural_flow'),
    'split': ('train', 'test'),
    'SWEEP': ('choke', 'lift'),
    'FRBH': ('bubbly', 'dispersed-bubbly', 'slug', 'churn', 'annular'),
    'FRWH': ('bubbly', 'dispersed-bubbly', 'slug', 'churn', 'annular'),
    'not_producible_reason': ('water_cut', 'below_floor', 'no_operating_point', 'solver_failure'),
    'reason': _REASONS,
}
_OUTCOMES = {  # The outcome's codes: an attempt's in `redraws`, a well's in `drawn_wells`
    'redraws': ('operating_point', 'no_operating_point', 'solver_failure'),
    'drawn_wells': _WELL_OUTCOMES,
}
_INTEGERS = {  # Integer columns: nullable where a row may hold none
    'ID': 'int32',
    'n_cells': 'int16',
    'draw_index': 'int64',
    'slot': 'int16',
    'attempt': 'int16',
    'station': 'int16',
    'TEST': 'Int8',
    'n_roots': 'int8',
    'n_stable': 'int8',
}
_COUNTS = ('n_attempts', 'n_redraws', 'n_redraws_no_op', 'n_redraws_solver')  # A well's counts of its slots' attempts
_COUNTS += tuple(f'n_redraws_{reason}' for reason in _REASONS) + ('n_several_stable',)
_INTEGERS |= dict.fromkeys(_COUNTS, 'int32')
_BOOLEANS = ('CHOKED', 'LIFT')
assert set(_CODES['FRBH']) == set(HASAN_REGIMES)  # SLIP-16's five regimes


# --- Types ------------------------------------------------------------------------------------------------------------
def _typed(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """
    A table's columns in their types (datasets-draft.md, Section 2): labels as categories with the codes of CODES (and
    OUTCOMES for an `outcome`), IDs, slots and counts as integers, `ID` nullable in the logs, flags as booleans; the
    other numbers float64, a value that does not apply null.

    :raises ValueError: for a label's value without a code
    """
    df = df.copy()
    for col in df.columns:
        codes = _OUTCOMES[table] if col == 'outcome' else _CODES.get(col)
        if codes is not None:
            unknown = set(df[col].dropna()) - set(codes)
            if unknown:
                raise ValueError(f'{table}.{col}: values without a code: {sorted(unknown)}')
            df[col] = pd.Categorical(df[col], categories=list(codes))
        elif col in _INTEGERS:
            dtype = _INTEGERS[col]
            if col == 'ID' and table in _LOGS:
                dtype = 'Int32'
            if col in ('n_roots', 'n_stable') and len(df) and df[col].max() > 127:
                raise ValueError(f'{table}.{col} above 127')
            df[col] = df[col].astype(dtype)
        elif col in _BOOLEANS:
            df[col] = df[col].astype(bool)
    return df


# --- Records ----------------------------------------------------------------------------------------------------------
def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def code_record() -> dict:
    """
    The code's record: `version` (datasets.io.code_version), the full `commit` of the working tree (None outside a
    git checkout) and whether the tree has changes, `dirty`.
    """
    out = dict(version=code_version(), commit=None, dirty=None)
    try:
        run = dict(cwd=_root(), capture_output=True, text=True, check=True)
        out['commit'] = subprocess.run(['git', 'rev-parse', 'HEAD'], **run).stdout.strip()
        out['dirty'] = bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], **run).stdout)
    except (OSError, subprocess.CalledProcessError):
        pass
    return out


def _lock_hash() -> str | None:
    """The SHA-256 of the lock file, uv.lock, that fixes the environment; None where there is none."""
    path = _root() / 'uv.lock'
    return _sha256(path) if path.exists() else None


def _platform_record() -> dict:
    return dict(python=sys.version, platform=platform.platform(), machine=platform.machine())


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


# --- Files ------------------------------------------------------------------------------------------------------------
def _write_tables(folder, tables: dict, metadata: dict) -> dict:  # spec: SMP-110
    """
    A folder of tables: one parquet file per table, then metadata.json, which holds metadata and each file's row count
    and SHA-256 under `files`. A folder without metadata.json is incomplete.

    :param folder: The folder, made if missing; an older metadata.json in it is removed first
    :param tables: {name: DataFrame}, each typed (_typed)
    :param metadata: The records to write
    :return: The metadata written
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    meta_path = folder / 'metadata.json'
    meta_path.unlink(missing_ok=True)
    files = {}
    for name, df in tables.items():
        path = folder / f'{name}.parquet'
        df.to_parquet(path, index=False)
        files[path.name] = dict(rows=len(df), sha256=_sha256(path))
    text = json.dumps(dict(metadata, files=files), indent=1, default=_json_default)
    tmp = folder / 'metadata.json.tmp'
    tmp.write_text(text)
    tmp.rename(meta_path)
    return json.loads(text)


def _json_default(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return str(v)


def read_dataset(folder) -> dict:
    """A dataset's tables, {name: DataFrame}, and its records under `metadata`. The companion folder's logs, where
    given in place of the dataset's, are read the same way."""
    folder = Path(folder)
    out = {p.stem: pd.read_parquet(p) for p in sorted(folder.glob('*.parquet'))}
    out['metadata'] = json.loads((folder / 'metadata.json').read_text())
    return out


# --- Re-solving a sample ----------------------------------------------------------------------------------------------
def pvt_from_rows(rows: pd.DataFrame):  # spec: SMP-110
    """
    A table well's PVT model from its `pvt` rows: a PVTTableSet where the rows carry several GORs, else a PVTTable
    with the rows' one GOR, or none, as PVTTable.read_csv reads the table's file, to the bit. A column that is null in
    every row of the table is one the table does not have.
    """
    rows = rows.reset_index(drop=True)
    first = rows.iloc[0]
    common = dict(rho_o=float(first['rho_o']), rho_g=float(first['rho_g']))
    common |= {key: first[key] for key in TEXT_KEYS if key in rows and isinstance(first[key], str)}
    columns = ['p', 'T', 'R_so', 'B_o', 'Z'] + [c for c in OPTIONAL_COLUMNS if c in rows and rows[c].notna().any()]
    gors = rows['gor'] if 'gor' in rows else pd.Series(math.nan, index=rows.index)
    if gors.nunique() > 1:
        tables = tuple(
            PVTTable.from_frame(d[columns], 'pvt rows', gor=float(g), **common)
            for g, d in rows.groupby('gor', sort=True)
        )
        return PVTTableSet(tables=tables)
    gor = float(gors.iloc[0]) if gors.notna().all() else None
    return PVTTable.from_frame(rows[columns], 'pvt rows', gor=gor, **common)


def _value(row, key):
    v = row[key]
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def sample_case(well, sample, survey, pvt=None) -> tuple:  # spec: SMP-110, SMP-111
    """
    A sample's WellProperties and BoundaryConditions in the v2.0.0 configuration, from the dataset's rows alone, as
    the generator built them (manywells.sampling.v2.mapping.WellBuilder): the well's `geometry` stations, its black
    oil or its table set, and its inputs, the varying ones from the sample (nsol-2: the heat-transfer coefficient, the
    roughness, the inflow's coefficients and whether lift is installed too).

    :param well: The well's row of `wells` (a pandas Series or a dict)
    :param sample: The sample's row of `samples`
    :param survey: The well's rows of `geometry`, in the order of `station`, with `md` and `tvd`
    :param pvt: A table well's rows of `pvt` (those of its `pvt_table_id`), or None for a black oil
    :return: (WellProperties, BoundaryConditions), checked against the v2.0.0 configuration
    """
    nsol = 'HTC' in sample
    if well['pvt_model'] == 'table':
        if pvt is None or len(pvt) == 0:
            raise ValueError(f'well {well["ID"]} is a table fluid: its pvt rows are needed')
        hc = pvt_from_rows(pvt)
    else:
        hc = BlackOil(rho_o=well['rho_o'], rho_g=well['rho_g'], gas_law='dak', p_sep=well['p_sep'], T_sep=well['T_sep'])
    inputs = sample if nsol else well
    keys = _NSOL_INPUTS if nsol else {key: key for key in _NSOL_INPUTS}
    coef = {key: inputs[keys[key]] for key in ('k_l', 'w_l_max', 'forchheimer_a', 'forchheimer_b')}
    if well['inflow'] == 'forchheimer':
        inflow = Forchheimer(a=coef['forchheimer_a'], b=coef['forchheimer_b'])
    elif well['inflow'] == 'vogel':
        inflow = Vogel(w_l_max=coef['w_l_max'])
    else:
        inflow = ProductivityIndex(k_l=coef['k_l'])
    lifted = bool(sample['LIFT']) if nsol else well['lift'] == 'gas_lift'
    md, tvd = np.asarray(survey['md'], dtype=float), np.asarray(survey['tvd'], dtype=float)
    fluid = FluidModel(
        pvt=hc,
        rho_w=well['rho_w'],
        gor=sample['GOR'],
        wlr=sample['WLR'],
        cp_g=well['cp_g'],
        cp_o=well['cp_o'],
        cp_w=well['cp_w'],
    )
    wp = WellProperties(
        geometry=WellGeometry.from_survey(md, tvd, int(well['n_cells']), D=well['D']),
        fluid=fluid,
        friction=RoughnessFriction(roughness=inputs[keys['roughness']], correlation='chen'),
        thermal=ThermalModel(h=inputs[keys['h']]),
        slip=V2_SLIP,
        inflow=inflow,
        choke=SimpsonChokeModel(K_c=well['K_c'], chk_profile=well['chk_profile']),
        injection_md=_value(well, 'injection_md') if lifted else None,
    )
    bc = BoundaryConditions(
        p_r=sample['PRES'],
        p_s=sample['PDC'],
        T_r=well['T_r'],
        T_s=well['T_s'],
        T_lg=_value(well, 'T_lg') if lifted else None,
        u=sample['CHK'],
        w_lg=sample['WGL'] if lifted else 0.0,
    )
    check(wp, V2)
    return wp, bc


def _sample_features(root, bc, fluid) -> dict:  # spec: SMP-30
    """The operating point's features of a sample (FEATURES): the root's, and the stream's mass fractions."""
    x_w = 1 - fluid.f_o_in_liquid
    row = root_features(root, bc, fluid.rho_o, fluid.rho_w, fluid.rho_g, x_w)
    f_g = fluid.f_g
    row |= {'FGAS': f_g, 'FOIL': (1 - f_g) * (1 - x_w), 'FWAT': (1 - f_g) * x_w}
    return {key: row[key] for key in FEATURES}


def resolve_sample(well, sample, survey, pvt=None, backend='rust') -> dict | None:  # spec: SMP-110
    """
    A sample solved again from its rows (sample_case): its features (_sample_features) at the operating point, or None
    where the solve finds none. With the dataset's code version and backend on the generator's platform, they are the
    sample's to the bit (SMP-110).
    """
    wp, bc = sample_case(well, sample, survey, pvt)
    rs = SSDFSimulator(wp, backend=backend).root_set(bc)
    if rs.operating_point is None:
        return None
    return _sample_features(rs.operating_point, bc, wp.fluid)


def sample_rows(tables: dict, ID: int, slot: int) -> tuple:
    """
    The rows that re-solve a sample of a dataset (read_dataset): its well's row, its row, its survey and, for a table
    fluid, its table's rows (else None).
    """
    wells, samples = tables['wells'], tables['samples']
    well = wells.loc[wells['ID'] == ID].iloc[0]
    sample = samples.loc[(samples['ID'] == ID) & (samples['slot'] == slot)].iloc[0]
    survey = tables['geometry'].loc[tables['geometry']['ID'] == ID].sort_values('station')
    pvt = None
    if well['pvt_model'] == 'table':
        pvt = tables['pvt'].loc[tables['pvt']['pvt_table_id'] == well['pvt_table_id']]
    return well, sample, survey, pvt
