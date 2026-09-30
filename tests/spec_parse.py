"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Reads the model spec for the spec tests: equation IDs, coverage tables, test vectors and code tags
(specs/model/README.md).
"""

import json
import pathlib
import re
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPECS = ROOT / 'specs'
MODEL = SPECS / 'model'
ROW_VECTORS = MODEL / 'vectors' / 'v1_rows.json'
CODE_DIRS = [ROOT / 'src', ROOT / 'rust']

ID = r'[A-Z]+(?:-[A-Z]+)*-\d+'
DEFINITION = re.compile(rf'^### ({ID}) · (.+)$')
TAG = re.compile(rf'(?:#|//)\s*spec:\s*({ID}(?:\s*,\s*{ID})*)')
BEGIN, END = '<!-- vectors:begin', '<!-- vectors:end'

NAMESPACES = {
    'BAL': 'model/balances.md', 'DISC': 'model/discretization.md', 'SOL': 'model/solution.md',
    'GEO': 'model/geometry.md', 'PVT-GAS': 'model/pvt/gas.md', 'PVT-OIL': 'model/pvt/oil.md',
    'PVT-WAT': 'model/pvt/water.md', 'PVT-MIX': 'model/pvt/mixture.md', 'SLIP': 'model/slip.md',
    'FRIC': 'model/friction.md', 'THM': 'model/thermal.md', 'INF': 'model/inflow.md',
    'CHK': 'model/choke.md', 'SMO': 'model/smoothing.md', 'SMP': 'sampling.md',
}
CHECK_KINDS = ('vectors', 'rows', 'verifier:', 'property:', 'spec-only:')


def namespace(eq_id):
    return eq_id.rsplit('-', 1)[0]


def spec_files():
    files = sorted(MODEL.rglob('*.md'))
    if (SPECS / 'sampling.md').exists():
        files.append(SPECS / 'sampling.md')
    return files


def rel(path):
    return path.relative_to(SPECS).as_posix()


def outside_vectors(text):
    """The lines of a spec file outside its generated vector block."""
    inside = False
    for line in text.splitlines():
        if line.startswith(BEGIN):
            inside = True
        elif line.startswith(END):
            inside = False
        elif not inside:
            yield line


def definitions():
    """Every definition heading, as (ID, file, title); an ID defined twice appears twice."""
    found = []
    for path in spec_files():
        for line in outside_vectors(path.read_text()):
            m = DEFINITION.match(line)
            if m:
                found.append((m.group(1), rel(path), m.group(2)))
    return found


def table_rows(lines):
    """The body rows of the first Markdown table in lines, as lists of stripped cells."""
    rows, started = [], False
    for line in lines:
        if line.startswith('|'):
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            if started and not set(''.join(cells)) <= set('-: '):
                rows.append(cells)
            started = True
        elif started:
            break
    return rows


def section(text, title):
    """The lines of the level-2 section with this title."""
    lines, inside = [], False
    for line in text.splitlines():
        if line.startswith('## '):
            inside = line[3:].strip() == title
        elif inside:
            lines.append(line)
    return lines


@dataclass(frozen=True)
class CoverageRow:
    eq_id: str
    file: str
    paper: str
    code: str
    checked_by: tuple


def coverage():
    """Every row of every coverage table."""
    found = []
    for path in spec_files():
        for cells in table_rows(section(path.read_text(), 'Coverage')):
            if len(cells) != 4:
                raise ValueError(f'{rel(path)}: coverage row with {len(cells)} cells: {cells}')
            checks = tuple(c.strip() for c in cells[3].split(';') if c.strip())
            found.append(CoverageRow(cells[0], rel(path), cells[1], cells[2], checks))
    return found


def parse_value(cell):
    if cell in ('true', 'false'):
        return cell == 'true'
    try:
        return float(cell)
    except ValueError:
        return cell


@dataclass(frozen=True)
class VectorTable:
    file: str
    heading: str
    ids: tuple
    inputs: tuple
    outputs: tuple
    rows: tuple      # each row: (inputs dict, outputs dict)


def vector_tables():
    """Every component vector table in the generated blocks."""
    tables = []
    for path in spec_files():
        text = path.read_text()
        if BEGIN not in text:
            continue
        block = text[text.index(BEGIN):text.index(END)].splitlines()
        starts = [k for k, line in enumerate(block) if line.startswith('### ')]
        for k, start in enumerate(starts):
            heading = block[start][4:].strip()
            lines = block[start + 1:starts[k + 1] if k + 1 < len(starts) else len(block)]
            header = next(line for line in lines if line.startswith('|'))
            cols = [c.strip() for c in header.strip().strip('|').split('|')]
            outputs = tuple(c[1:].strip() for c in cols if c.startswith('→'))
            inputs = tuple(c for c in cols if not c.startswith('→'))
            rows = []
            for cells in table_rows(lines):
                values = [parse_value(c) for c in cells]
                rows.append((dict(zip(inputs, values[:len(inputs)])), dict(zip(outputs, values[len(inputs):]))))
            ids = tuple(re.findall(ID, heading.split('(')[0]))
            tables.append(VectorTable(rel(path), heading, ids, inputs, outputs, tuple(rows)))
    return tables


def row_vectors():
    return json.loads(ROW_VECTORS.read_text())


def code_tags():
    """Every `spec:` tag in the code, as (file, line number, ID)."""
    found = []
    for base in CODE_DIRS:
        for path in sorted(base.rglob('*')):
            if path.suffix not in ('.py', '.rs') or '__pycache__' in path.parts:
                continue
            for n, line in enumerate(path.read_text().splitlines(), 1):
                for m in TAG.finditer(line):
                    for eq_id in re.findall(ID, m.group(1)):
                        found.append((path.relative_to(ROOT).as_posix(), n, eq_id))
    return found


def retired_ids():
    """IDs listed after 'Retired IDs:' in specs/model/README.md."""
    for line in (MODEL / 'README.md').read_text().splitlines():
        if line.startswith('Retired IDs:'):
            return set(re.findall(ID, line))
    raise ValueError('specs/model/README.md has no "Retired IDs:" line')
