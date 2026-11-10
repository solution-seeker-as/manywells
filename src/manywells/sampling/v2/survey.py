"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 10 November 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The survey of a v2 well (specs/sampling.md, SMP-60): stations (MD, TVD) from the wellhead to the bottom of the
modelled flow path, by the well's geometry family. Straight sections, and circular builds and drops with one station
per degree. A survey takes N_NUMBERS numbers of its generator, whatever its family and whether it fits (SMP-51).
"""

import math

import numpy as np

from manywells.sampling.v2._distributions import unused

N_NUMBERS = 6  # A survey's numbers: five of the shape and the kickoff's, whatever the family  # spec: SMP-51
FAMILIES = ('vertical', 'build_and_hold', 's_shaped', 'high_angle')  # spec: SMP-59
_KICKOFF = (0.05, 0.75)  # The kickoff's share of the TVD below the mudline
_BUILD = (0.3, 3.6)  # Build rate (degrees per 30 m)
_DROP, _S_BOTTOM = (0.2, 2.2), (0.0, 35.0)  # An S well's drop rate (degrees per 30 m) and bottom inclination (degrees)
_S_HOLD, _S_END = (15.0, 45.0), (100.0, 400.0)  # An S well's hold inclination (degrees) and end section (m of TVD)
_TANGENT, _LANDING = (10.0, 65.0), (1.5, 7.0)  # A drain well's tangent before landing (degrees) and landing build
_HEEL_TANGENT = (50.0, 300.0)  # A drain well's tangent at the heel's inclination (m of MD)


def _arc(st, th1, rate):
    """Append stations along a circular arc from the last station's inclination to th1 (degrees), at a build or drop
    rate (degrees per 30 m), one station per degree."""
    md0, tvd0, th0 = st[-1]
    kappa = math.radians(rate) / 30.0
    n = max(1, int(abs(th1 - th0)))
    for th in np.linspace(th0, th1, n + 1)[1:]:
        md = md0 + abs(math.radians(th - th0)) / kappa
        tvd = tvd0 + abs(math.sin(math.radians(th)) - math.sin(math.radians(th0))) / kappa
        st.append((md, tvd, th))


def _hold_to(st, tvd_end):
    """Append a station straight on from the last, at its inclination, to the TVD tvd_end."""
    md0, tvd0, th = st[-1]
    if tvd_end > tvd0:
        st.append((md0 + (tvd_end - tvd0) / math.cos(math.radians(th)), tvd_end, th))


def _arc_tvd(th0, th1, rate):
    """The TVD of an arc from th0 to th1 (degrees) at a rate (degrees per 30 m)."""
    return abs(math.sin(math.radians(th1)) - math.sin(math.radians(th0))) * 30.0 / math.radians(rate)


def _raise_rate(rate, top, need, room):
    """Raise a build or drop rate by steps of 20%, to at most top, until the TVD need(rate) fits in room."""
    while need(rate) > room and rate < top:
        rate = min(top, 1.2 * rate)
    return rate


def survey(rng, family, L, theta=None, d_mud=0.0):  # spec: SMP-60
    """
    One draw of a well's survey: (md, tvd, parameters), or None where the shape does not fit the well's depth, and
    the caller redraws it.

    Each shape parameter is uniform over its range, and the kickoff over the part of 0.05 to 0.75 of the depth below
    the mudline that leaves room for the rest of the shape. A build or drop too gentle for the well's depth is raised
    towards its range's top; the angles are never lowered. Every draw takes N_NUMBERS numbers of the generator: a
    family draws the numbers that it does not use, and a shape that does not fit its kickoff's (SMP-51).

    :param rng: The geometry block's generator
    :param family: One of FAMILIES
    :param L: True vertical depth of the bottom of the modelled flow path below the wellhead (m)
    :param theta: The class's range (degrees) of the hold inclination (build_and_hold) or of the heel's (high_angle)
    :param d_mud: The mudline's TVD in the modelled well (m): a platform well's water depth, else 0
    :return: The stations' measured depths and true vertical depths (m), from the wellhead, and the shape's parameters
    """
    prm = {}
    if family == 'vertical':
        unused(rng, N_NUMBERS)
        return [0.0, L], [0.0, L], prm
    z = L - d_mud
    room = L - d_mud - _KICKOFF[0] * z  # The TVD below the shallowest kickoff

    def kickoff(below):
        """Uniform over the part of the kickoff's range that leaves the TVD below for the rest of the shape."""
        f_max = min(_KICKOFF[1], (L - below - d_mud) / z)
        if f_max < _KICKOFF[0]:
            unused(rng)  # The kickoff's number
            return None
        return d_mud + z * rng.uniform(_KICKOFF[0], f_max)

    st = [(0.0, 0.0, 0.0)]
    if family == 'build_and_hold':
        th, b = rng.uniform(*theta), rng.uniform(*_BUILD)
        unused(rng, 3)  # The other families' three further numbers
        b = _raise_rate(b, _BUILD[1], lambda x: _arc_tvd(0, th, x), room - 50)
        K = kickoff(50 + _arc_tvd(0, th, b))
        if K is None:
            return None
        st.append((K, K, 0.0))
        _arc(st, th, b)
        _hold_to(st, L)
        prm.update(kickoff=K, theta_hold=th, build_rate=b)
    elif family == 's_shaped':
        th, b, d = rng.uniform(*_S_HOLD), rng.uniform(*_BUILD), rng.uniform(*_DROP)
        th_e, end_tvd = min(rng.uniform(*_S_BOTTOM), th), min(rng.uniform(*_S_END), 0.25 * L)
        left = room - 100 - end_tvd
        b = _raise_rate(b, _BUILD[1], lambda x: _arc_tvd(0, th, x) + _arc_tvd(th, th_e, d), left)
        d = _raise_rate(d, _DROP[1], lambda x: _arc_tvd(0, th, b) + _arc_tvd(th, th_e, x), left)
        K = kickoff(100 + end_tvd + _arc_tvd(0, th, b) + _arc_tvd(th, th_e, d))
        if K is None:
            return None
        st.append((K, K, 0.0))
        _arc(st, th, b)
        _hold_to(st, L - end_tvd - _arc_tvd(th, th_e, d))
        _arc(st, th_e, d)
        _hold_to(st, L)
        prm.update(kickoff=K, theta_hold=th, theta_end=th_e, build_rate=b, drop_rate=d, end_section_tvd=end_tvd)
    elif family == 'high_angle':
        th, th_heel = rng.uniform(*_TANGENT), rng.uniform(*theta)
        th = min(th, th_heel)  # The tangent is never steeper than the heel
        b1, b2, tan_md = rng.uniform(*_BUILD), rng.uniform(*_LANDING), rng.uniform(*_HEEL_TANGENT)

        def below(x1, x2):
            return _arc_tvd(0, th, x1) + _arc_tvd(th, th_heel, x2) + tan_md * math.cos(math.radians(th_heel))

        b1 = _raise_rate(b1, _BUILD[1], lambda x: below(x, b2), room - 100)
        b2 = _raise_rate(b2, _LANDING[1], lambda x: below(b1, x), room - 100)
        K = kickoff(100 + below(b1, b2))
        if K is None:
            return None
        st.append((K, K, 0.0))
        _arc(st, th, b1)
        _hold_to(st, L - _arc_tvd(th, th_heel, b2) - tan_md * math.cos(math.radians(th_heel)))
        _arc(st, th_heel, b2)
        _hold_to(st, L)
        prm.update(kickoff=K, theta_hold=th, theta_heel=th_heel, build_rate=b1, build_rate_2=b2, tangent_md=tan_md)
    else:
        raise ValueError(f'unknown geometry family {family!r}')
    md = [s[0] for s in st]
    tvd = [s[1] for s in st]
    keep = [0] + [i for i in range(1, len(md)) if md[i] > md[i - 1] + 1e-6]
    return [md[i] for i in keep], [tvd[i] for i in keep], prm
