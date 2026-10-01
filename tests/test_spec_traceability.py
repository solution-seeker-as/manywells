"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 30 September 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Traceability between the model spec and the code (principle 1 of plans/manywells-v2-plan.md):
every equation ID is defined once and covered, every `# spec:` tag in the code names an ID, and
every ID is tagged in code or marked spec-only (specs/model/README.md).
"""

import collections

from .spec_parse import (CHECK_KINDS, NAMESPACES, code_tags, coverage, definitions, namespace, retired_ids,
                         row_vectors, vector_tables)


def defined():
    return {eq_id: path for eq_id, path, _ in definitions()}


def test_ids_are_defined_once():
    counts = collections.Counter(eq_id for eq_id, _, _ in definitions())
    assert counts, 'no equation IDs found in specs/'
    assert not [k for k, n in counts.items() if n > 1], 'defined more than once'


def test_ids_are_in_their_namespace_file():
    wrong = [(eq_id, path) for eq_id, path in defined().items() if NAMESPACES.get(namespace(eq_id)) != path]
    assert not wrong, 'IDs defined outside their namespace file (or in an unknown namespace)'


def test_retired_ids_are_not_defined():
    assert not retired_ids() & set(defined())


def test_every_id_has_one_coverage_row():
    ids = defined()
    rows = coverage()
    counts = collections.Counter(r.eq_id for r in rows)
    assert not [k for k, n in counts.items() if n > 1], 'IDs with more than one coverage row'
    assert not [r.eq_id for r in rows if r.eq_id not in ids], 'coverage rows for undefined IDs'
    assert not [r.eq_id for r in rows if r.file != ids[r.eq_id]], 'coverage rows outside the defining file'
    assert not sorted(set(ids) - set(counts)), 'IDs without a coverage row'


def test_checked_by_is_valid():
    """Every ID has test vectors, a spot check or a property, or is spec-only with a reason (Step 4, "Done when")."""
    bad = []
    for r in coverage():
        if not r.checked_by:
            bad.append((r.eq_id, 'empty'))
        for check in r.checked_by:
            kind = next((k for k in CHECK_KINDS if check == k or check.startswith(k + ' (')
                         or (k.endswith(':') and check.startswith(k))), None)
            if kind is None or (kind.endswith(':') and not check[len(kind):].strip()):
                bad.append((r.eq_id, check))
    assert not bad


def test_vector_claims_are_backed():
    tables = vector_tables()
    in_tables = collections.defaultdict(set)
    for t in tables:
        for eq_id in t.ids:
            in_tables[eq_id].add(t.file)
    rows = coverage()
    claims = {r.eq_id: r for r in rows}
    assert not [r.eq_id for r in rows if 'vectors' in r.checked_by and r.file not in in_tables[r.eq_id]], \
        'coverage claims vectors, but the file has no table for the ID'
    assert not [i for i in in_tables if i not in claims or 'vectors' not in claims[i].checked_by], \
        'vector tables for IDs whose coverage does not claim vectors'
    assert all(t.rows for t in tables), 'empty vector table'


def test_row_vector_ids_are_defined_and_claimed():
    claims = {r.eq_id: r for r in coverage()}
    ids = {eq_id for well in row_vectors()['wells'] for point in well['rows'] for eq_id, _ in point}
    assert ids
    assert not [i for i in ids if i not in claims or 'rows' not in claims[i].checked_by]


def test_code_tags_name_defined_ids():
    ids = set(defined())
    unknown = [(f, n, eq_id) for f, n, eq_id in code_tags() if eq_id not in ids]
    assert not unknown, 'spec: tags that name no defined ID'


def test_every_id_is_tagged_or_spec_only():
    tagged = {eq_id for _, _, eq_id in code_tags()}
    spec_only = {r.eq_id for r in coverage() if any(c.startswith('spec-only:') for c in r.checked_by)}
    missing = sorted(set(defined()) - tagged - spec_only)
    assert not missing, f'{len(missing)} IDs neither tagged in code nor spec-only: {", ".join(missing)}'
