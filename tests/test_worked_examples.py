"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 29 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Checks develop's correlations against worked examples published with them or in textbooks (specs/model/README.md,
Test vectors, Independent checks; plans/model-lock-plan.md, Step L3).

The develop vectors in specs/model/ come from develop itself, and the Rust core implements the same spec, so neither
checks the values against the sources. These do: each example names its source and gives the value as printed, and
develop's value must round to it, that is, lie within half a unit of its last digit. The inputs are the example's,
converted to the interface's units with the example's own conversions; where the example carries an intermediate
unrounded, or rounds it before the next step, the inputs do the same. Each example is checked on the CasADi
backend's component functions and, where the test adapter rust() reaches the same function, on the Rust core.

A value develop does not reproduce, after checking units and inputs, is a finding: it is recorded as open in the
spec file's Test vectors section and is a strict expected failure here, so that the entry has to go once the finding
is ruled on. A value whose example disagrees with its own equation at its own inputs is the example's error, not a
finding; those are listed in EXAMPLE_ERRORS and not checked. The comparisons with fluids and with transcriptions of
the sources' equations, for the correlations without a worked example, are in plans/evidence/independent_checks.py.
"""

import math
from dataclasses import dataclass

import casadi as ca
import pytest

from manywells.geometry import WellGeometry
from manywells.inflow import Vogel
from manywells.pvt import density_from_api, gas_density_from_sg, mixture_viscosity
from manywells.pvt.black_oil import BlackOil, BlackOilPVT, live_oil_viscosity
from manywells.pvt.dead_oil import dead_oil_viscosity
from manywells.pvt.fluid import FluidModel
from manywells.pvt.gas import gas_viscosity, sutton_pseudo_critical
from manywells.slip import HASAN_REGIMES, SlipModel
from manywells.units import CF_BAR, CF_CP, CF_PSI, CF_RS, T_REF, fahrenheit_to_kelvin

from .test_spec_vectors import rust

AHMED = 'Ahmed (2006), Reservoir engineering handbook, 3rd ed.'
HK = 'Hasan and Kabir (2018), Fluid flow and heat transfer in wellbores, 2nd ed.'
FT, LBFT3, LBS2 = 0.3048, 16.018463, 0.45359237     # ft -> m, lbm/ft3 -> kg/m3, lbm/s2 -> N/m


def num(x):
    return float(ca.DM(x)) if isinstance(x, (ca.DM, ca.SX, ca.MX)) else float(x)


def bar(psia):
    return psia * CF_PSI / CF_BAR


@dataclass(frozen=True)
class Example:
    ids: str                # the spec IDs it checks
    source: str             # reference, example and page
    quantity: str           # what the example prints, and its unit
    printed: str            # the value as printed: its last digit sets the tolerance
    python: object          # () -> the CasADi backend's value, in the printed unit
    rust: object = None     # () -> the Rust core's value, or None where the adapter does not reach the function
    finding: str = None     # an open finding: develop does not reproduce the printed value


def agrees(value, printed):
    """Whether value rounds to printed: a label must be equal, a number within half a unit of the last digit."""
    if not printed[0].isdigit():
        return value == printed
    decimals = len(printed.partition('.')[2])
    return abs(num(value) - float(printed)) <= 0.5 * 10 ** -decimals * (1 + 1e-9)


# ---------------------------------------------------------------------------------------------------------------
# Vazquez and Beggs (PVT-OIL-5, PVT-OIL-6, PVT-OIL-8) and Beggs and Robinson (PVT-OIL-10, PVT-OIL-11): the six oils
# of Ahmed's Example 2-17

@dataclass(frozen=True)
class Oil:
    T: float            # reservoir temperature (°F)
    p_b: float          # bubble point (psig)
    R_s: float          # solution gas-oil ratio at p_b (scf/STB)
    p_sep: float        # first-stage separator (psig)
    T_sep: float        # (°F)
    api: float
    sg_gas: float       # gas gravity at the separator
    gamma_gs: float     # Vazquez and Beggs's Eq. (1) at the separator, unrounded, as Examples 2-18 and 2-29 carry it

    @property
    def p_b_psia(self):
        return self.p_b + 14.7      # psig + 14.7, as Ahmed's Example 2-23 gives the measured bubble points

    def black_oil(self, gor=200.0):
        """A fluid with develop's black oil of this oil, from its densities and its separator: PVT-OIL-5 to PVT-OIL-8
        in turn. Its gas-oil ratio (Sm3/Sm3) caps the fluid's R_s and B_o (PVT-OIL-15), not those of its pvt."""
        pvt = BlackOil(rho_o=density_from_api(self.api), rho_g=gas_density_from_sg(self.sg_gas),
                       p_sep=bar(self.p_sep + 14.7), T_sep=fahrenheit_to_kelvin(self.T_sep))
        return FluidModel(pvt=pvt, gor=gor)


# Ahmed (2006), Example 2-17, p. 80. Example 2-29 repeats them with oil 5's bubble point as 3065 psig, which no
# result below uses.
OILS = {
    1: Oil(250, 2377, 751, 150, 60, 47.1, 0.851, 0.873341),
    2: Oil(220, 2620, 768, 100, 75, 40.7, 0.855, 0.855),
    3: Oil(260, 2051, 693, 100, 72, 48.6, 0.911, 0.911),
    4: Oil(237, 2884, 968, 60, 120, 40.5, 0.898, 0.849946),
    5: Oil(218, 3045, 943, 200, 60, 44.2, 0.781, 0.814339),
    6: Oil(180, 4239, 807, 85, 173, 27.3, 0.848, 0.833588),
}


def vazquez_beggs(api, gamma_gs, gor=None):
    """BlackOilPVT, Vazquez and Beggs's correlations, at API gravity api with the separator gas gravity gamma_gs.

    BlackOilPVT refuses API gravities above 40 (pvt/oil.md, Safeguards), a guard of the model and not part of the
    correlations (specs/model/validity.md, decision 2, proposes 16 to 58), and five of the six oils have 40.5 to
    48.6. So it is built at an API gravity in range and in the same branch of the coefficients, with the separator at
    114.7 psia, where PVT-OIL-5 leaves the gravity as given, and takes api afterwards."""
    pvt = BlackOilPVT(api=min(api, 40.0), sg_gas=gamma_gs, p_sep=114.7 * CF_PSI, T_sep=T_REF, gor=gor)
    pvt.api = api
    return pvt


def in_range(oil):
    return 10 <= oil.api <= 40


def rs_scf(oil, fluid=None):
    """R_s at the bubble point (scf/STB): of BlackOilPVT at the oil's gamma_gs, or of the given fluid's black oil."""
    p, T = oil.p_b_psia * CF_PSI, fahrenheit_to_kelvin(oil.T)
    if fluid is None:
        return num(vazquez_beggs(oil.api, oil.gamma_gs).rs(p, T)) / CF_RS
    return num(fluid.pvt.rs(p / CF_BAR, T)) / CF_RS


def bo_at_measured_rs(oil, fluid=None):
    """B_o at the oil's measured R_s, as Example 2-29 takes it: the solution gas-oil ratio capped at it (PVT-OIL-15),
    at twice the bubble point, where Vazquez and Beggs's R_s is more than twice the cap."""
    p, T = 2 * oil.p_b_psia * CF_PSI, fahrenheit_to_kelvin(oil.T)
    if fluid is None:
        return num(vazquez_beggs(oil.api, oil.gamma_gs, gor=oil.R_s * CF_RS).bo(p, T))
    return num(fluid.bo(p / CF_BAR, T))


def black_oil_examples():
    out = []
    # Example 2-25, p. 89: the separator gas gravity to three decimals (Example 2-18 prints oil 1's as 0.8731)
    for k, printed in {1: '0.873', 2: '0.855', 3: '0.911', 4: '0.850', 5: '0.814', 6: '0.834'}.items():
        oil = OILS[k]
        p_sep, T_sep = (oil.p_sep + 14.7) * CF_PSI, fahrenheit_to_kelvin(oil.T_sep)
        out.append(Example(
            'PVT-OIL-5', f'{AHMED}, Example 2-25, p. 89, oil {k}', 'gamma_gs', printed,
            (lambda o=oil, p=p_sep, T=T_sep: BlackOilPVT(o.api, o.sg_gas, p, T).sg_gas_corr) if in_range(oil) else None,
            lambda o=oil, p=p_sep, T=T_sep: rust('separator_gravity', o.api, o.sg_gas, p / CF_BAR, T)))
    # Example 2-18, p. 82: R_s at the bubble point (scf/STB)
    for k, printed in {1: '779', 2: '733', 3: '702', 5: '947', 6: '841'}.items():
        oil = OILS[k]
        out.append(Example('PVT-OIL-6', f'{AHMED}, Example 2-18, p. 82, oil {k}', 'R_s (scf/STB)', printed,
                           lambda o=oil: rs_scf(o)))
        if in_range(oil):
            out.append(Example(
                'PVT-OIL-5, PVT-OIL-6', f'{AHMED}, Example 2-18, p. 82, oil {k}, from the separator',
                'R_s (scf/STB)', printed, lambda o=oil: rs_scf(o, o.black_oil()),
                lambda o=oil: rust('black_oil', bar(o.p_b_psia), fahrenheit_to_kelvin(o.T),
                                   fluid=o.black_oil())[0] / CF_RS))
    # Example 2-29, pp. 97-98, Method 2: B_o at the bubble point, from the measured R_s
    for k, printed in {1: '1.474', 2: '1.450', 3: '1.451', 5: '1.546', 6: '1.389'}.items():
        oil = OILS[k]
        out.append(Example('PVT-OIL-8', f'{AHMED}, Example 2-29, p. 98, Method 2, oil {k}', 'B_o', printed,
                           lambda o=oil: bo_at_measured_rs(o)))
        if in_range(oil):
            out.append(Example(
                'PVT-OIL-5, PVT-OIL-8, PVT-OIL-15', f'{AHMED}, Example 2-29, p. 98, Method 2, oil {k}, from the '
                'separator', 'B_o', printed, lambda o=oil: bo_at_measured_rs(o, o.black_oil(o.R_s * CF_RS)),
                lambda o=oil: rust('bo', 2 * bar(o.p_b_psia), fahrenheit_to_kelvin(o.T),
                                   fluid=o.black_oil(o.R_s * CF_RS))))
    # Example 2-35, pp. 119-120: Beggs and Robinson's dead-oil viscosity at the reservoir temperature (cP), and the
    # saturated oil's from the measured dead-oil viscosity and R_s (the table's footnote, "Using the measured mu_od",
    # marks oil 1; every row's value is the one from the measured mu_od). Oil 2's measured mu_od is 1.286, printed
    # 0.286 in the solution's table.
    measured_mu_od = {1: 0.765, 2: 1.286, 3: 0.686, 4: 1.014, 5: 1.009, 6: 4.166}
    dead = {1: '0.568', 2: '1.020', 3: '0.493', 4: '0.917', 5: '0.829', 6: '4.246'}
    live = {1: '0.287', 2: '0.377', 3: '0.279', 4: '0.297', 5: '0.300', 6: '0.689'}
    for k, oil in OILS.items():
        T = fahrenheit_to_kelvin(oil.T)
        out.append(Example('PVT-OIL-10', f'{AHMED}, Example 2-35, p. 120, oil {k}', 'mu_od (cP)', dead[k],
                           lambda o=oil, T=T: num(dead_oil_viscosity(o.api, T)) / CF_CP,
                           lambda o=oil, T=T: rust('dead_oil_viscosity', o.api, T) / CF_CP))
        mu = measured_mu_od[k] * CF_CP
        out.append(Example('PVT-OIL-11', f'{AHMED}, Example 2-35, p. 120, oil {k}', 'mu_ob (cP)', live[k],
                           lambda o=oil, mu=mu: num(live_oil_viscosity(mu, o.R_s)) / CF_CP,
                           lambda o=oil, mu=mu: rust('live_oil_viscosity', mu, o.R_s) / CF_CP))
    return out


# ---------------------------------------------------------------------------------------------------------------
# The gas: Lee, Gonzalez and Eakin (PVT-GAS-7) and Sutton (PVT-GAS-5)

def gas_examples():
    # Ahmed (2006), Example 2-14, pp. 74-75: T = 600 °R, M_a = 20.85, rho_g = 8.3 lb/ft3, which Eq. (2-63) divides
    # by 62.4 for g/cm3
    T, rho_g, M = 600.0 / 1.8, 8.3 / 62.4 * 1000.0, 20.85
    out = [Example('PVT-GAS-7', f'{AHMED}, Example 2-14, p. 75', 'mu_g (cP)', '0.0173',
                   lambda: num(gas_viscosity(T, rho_g, M)) / CF_CP,
                   lambda: rust('gas_viscosity', T, rho_g, M) / CF_CP)]
    # Whitson and Brulé (2000), Appendix B, Problem 6, p. 174: Sutton's correlations at gamma_g = 0.65
    src = 'Whitson and Brulé (2000), Phase behavior, Appendix B, Problem 6, p. 174'
    out += [Example('PVT-GAS-5', src, 'T_pc (°R)', '365', lambda: sutton_pseudo_critical(0.65)[1] * 1.8,
                    lambda: rust('sutton_pseudo_critical', 0.65)[1] * 1.8),
            Example('PVT-GAS-5', src, 'p_pc (psia)', '670', lambda: sutton_pseudo_critical(0.65)[0] / CF_PSI,
                    lambda: rust('sutton_pseudo_critical', 0.65)[0] / CF_PSI)]
    # Sutton (1985), Eq. (14), p. 3: p_pc = 756.8 - 131.0 gamma_g - 3.6 gamma_g^2. develop's coefficient of gamma_g is
    # (p_pc(-1) - p_pc(1)) / 2 in psia: 131.07 until Bjarne's ruling of 2026-10-29 (pvt/gas.md, PVT-GAS-5)
    out.append(Example(
        'PVT-GAS-5', 'Sutton (1985), SPE 14265, Eq. (14), p. 3', 'coefficient of gamma_g in p_pc (psia)', '131.0',
        lambda: (sutton_pseudo_critical(-1.0)[0] - sutton_pseudo_critical(1.0)[0]) / 2 / CF_PSI,
        lambda: (rust('sutton_pseudo_critical', -1.0)[0] - rust('sutton_pseudo_critical', 1.0)[0]) / 2 / CF_PSI))
    return out


# ---------------------------------------------------------------------------------------------------------------
# The five flow regimes (SLIP-12 to SLIP-16) and the rise velocities (SLIP-4, SLIP-5): Hasan and Kabir's field
# examples. Their void fractions are not checked: the examples blend the rise velocities within slug and churn flow,
# (A-13) of Hasan et al. (2010), which SLIP-15 leaves out, and two of the three points lie within a few percent of a
# transition, where SLIP-13 smooths it (specs/features/019-five-flow-regimes.md).

@dataclass(frozen=True)
class FlowPoint:
    v_gs: float         # m/s
    v_ls: float
    rho_g: float        # kg/m3
    rho_l: float
    sigma: float        # N/m
    D: float            # m
    cos_incl: float

    def state(self, alpha=0.5):
        """A state at these superficial velocities: the five regimes depend on alpha only through them."""
        return dict(v_g=self.v_gs / alpha, v_l=self.v_ls / (1 - alpha), alpha=alpha, rho_g=self.rho_g,
                    rho_l=self.rho_l, sigma=self.sigma, D=self.D, cos_incl=self.cos_incl)

    def regime(self):
        return SlipModel().flow_regime(**self.state())

    def rust_regime(self):
        s = self.state()
        k = rust('regime', s['v_g'], s['v_l'], s['alpha'], s['rho_g'], s['rho_l'], s['sigma'], s['cos_incl'],
                 slip=SlipModel(), geometry=WellGeometry.vertical(length=1000.0, n_cells=10, D=self.D))
        return HASAN_REGIMES[int(k)]

    def c_0(self):
        return SlipModel().identify_parameters(**self.state())[0]

    def rust_c_0(self):
        s = self.state()
        return rust('slip_parameters', *s.values(), slip=SlipModel())[0]


# §3.4, Field Example 3.1: a vertical well at its wellhead, 505 psig. v_sg is 2.784 ft/s in the data, and 2.845 in the
# solution; the regime is slug flow with either.
HK_3_1 = FlowPoint(2.784 * FT, 1.601 * FT, 2.17 * LBFT3, 55.042 * LBFT3, 0.0696 * LBS2, 2.99 * 0.0254, 1.0)
# §4.2.4, Field Example 4.1: 17.5° from vertical, the wellhead at 361 psig
HK_4_1 = FlowPoint(17.62 * FT, 7.691 * FT, 1.08 * LBFT3, 53.3 * LBFT3, 0.064 * LBS2, 0.203 * FT,
                   math.cos(math.radians(17.5)))
# Appendix 7-E, §7-E.2: a geothermal well, 2.87° from vertical
HK_7_E_2 = FlowPoint(20.044 * FT, 2.222 * FT, 0.315 * LBFT3, 55.57 * LBFT3, 0.0936 * LBS2, 8.819 * 0.0254,
                     math.cos(math.radians(2.87)))


def harmathy_ft_s(f):
    return num(SlipModel.harmathy_rise_velocity(f.rho_g, f.rho_l, f.sigma)) / FT


def taylor_ft_s(f):
    return num(SlipModel.taylor_rise_velocity(f.rho_g, f.rho_l, f.D)) / FT


def slip_examples():
    out = []
    for f, where, printed in ((HK_3_1, '§3.4, Field Example 3.1', 'slug'),
                              (HK_4_1, '§4.2.4, Field Example 4.1', 'churn'),
                              (HK_7_E_2, 'Appendix 7-E, §7-E.2', 'annular')):
        out.append(Example('SLIP-12, SLIP-13, SLIP-14, SLIP-16', f'{HK}, {where}', 'flow regime', printed,
                           f.regime, f.rust_regime))
    out += [
        Example('SLIP-15', f'{HK}, §3.4, Field Example 3.1', 'C_0 of slug flow', '1.2', HK_3_1.c_0, HK_3_1.rust_c_0),
        Example('SLIP-4', f'{HK}, §3.4, Field Example 3.1, Eq. 3.6', 'v_inf (ft/s)', '0.680',
                lambda: harmathy_ft_s(HK_3_1),
                lambda: rust('harmathy_rise_velocity', HK_3_1.rho_g, HK_3_1.rho_l, HK_3_1.sigma) / FT),
        Example('SLIP-4', f'{HK}, §4.2.4, Field Example 4.1, Eq. 3.6', 'v_inf (ft/s)', '0.675',
                lambda: harmathy_ft_s(HK_4_1),
                lambda: rust('harmathy_rise_velocity', HK_4_1.rho_g, HK_4_1.rho_l, HK_4_1.sigma) / FT),
        Example('SLIP-5', f'{HK}, §3.4, Field Example 3.1, Eq. 3.8', 'v_inf_T (ft/s)', '0.971',
                lambda: taylor_ft_s(HK_3_1),
                lambda: rust('taylor_rise_velocity', HK_3_1.rho_g, HK_3_1.rho_l, HK_3_1.D) / FT),
    ]
    return out


# ---------------------------------------------------------------------------------------------------------------
# The mixture viscosity (PVT-MIX-9), and Vogel's inflow (INF-1, a well option)

def other_examples():
    # Hasan and Kabir (2018), §3.4, Field Example 3.1, Eq. 2.25: mu_m = x mu_g + (1 - x) mu_L with x the flowing mass
    # fraction v_sg rho_g / (v_sg rho_g + v_sL rho_L) = 0.066, as Hasan et al. (2010) define it for (A-3); develop
    # takes the in-situ mass fraction, at the example's gas volume fraction f_g = 0.457 (Eq. 3.27)
    mu_l, mu_g, alpha, rho_l, rho_g = 13.09, 0.019, 0.457, 55.042, 2.17     # cP, lbm/ft3
    out = [Example(
        'PVT-MIX-9', f'{HK}, §3.4, Field Example 3.1, Eq. 2.25', 'mu_m (cP)', '12.2',
        lambda: num(mixture_viscosity(mu_l, mu_g, alpha, rho_l, rho_g)),
        lambda: rust('mixture_viscosity', mu_l, mu_g, alpha, rho_l, rho_g),
        finding='PVT-MIX-9 weights by the in-situ mass fraction of gas, Hasan et al. (2010) by the flowing one '
                '(pvt/mixture.md, Test vectors, Independent checks: open finding)')]
    # Guo, Lyons and Ghalambor (2007), Example Problem 3.2, p. 3/34: Vogel's IPR at a reservoir pressure of 5,651 psia,
    # with q_max = J p / 1.8 (Eq. 3.21). The example prints J as 0.1968 stb/d-psi and q_max as 618 stb/d, and its table
    # carries both unrounded (with 618, two of its rows round the other way). Rates in stb/d at p_wf in psia: INF-1
    # takes their ratio, and w_l_max in any unit.
    J = 8.2 * 53 / (141.2 * 1.1 * 1.7 * (math.log(2980 / 0.328) - 0.75))
    q_max = J * 5651 / 1.8
    src = 'Guo et al. (2007), Petroleum production engineering, Example Problem 3.2, p. 3/34'
    rows = {5651: '0', 5000: '122', 4500: '206', 4000: '283', 3500: '352', 3000: '413', 2500: '466', 2000: '512',
            1500: '550', 1000: '580', 500: '603', 0: '618'}
    for p, printed in rows.items():
        out.append(Example('INF-1', f'{src}, p_wf = {p} psia', 'q_o (stb/d)', printed,
                           lambda p=p: num(Vogel(q_max).liquid_mass_flow_rate(float(p), 5651.0)),
                           lambda p=p: rust('liquid_rate', float(p), 5651.0, inflow=Vogel(q_max))))
    return out


EXAMPLES = black_oil_examples() + gas_examples() + slip_examples() + other_examples()

# Values the examples print that their own equations do not give at their own inputs, by hand: not checked
EXAMPLE_ERRORS = {
    f'{AHMED}, Example 2-18, p. 82, oil 4': 'R_s printed 820 scf/STB; Eq. (2-71) at its inputs gives 782 (develop '
                                            '782.3)',
    f'{AHMED}, Example 2-29, p. 98, Method 2, oil 4': 'B_o printed 1.542; Eq. (2-86) at the measured R_s of 968 '
                                                      'gives 1.556 (develop 1.5557)',
    f'{AHMED}, Example 2-18, p. 82, oil 1': 'gamma_gs printed 0.8731; Eq. (2-72) gives 0.87334, which Example 2-25 '
                                            'prints as 0.873 and Example 2-18 carries into R_s',
    f'{HK}, §3.4, Field Example 3.1, Eq. 3.12': 'v_sg for annular flow printed 6.939 ft/s; its own expression gives '
                                                '6.942 (develop 6.942)',
    f'{HK}, §4.2.4, Field Example 4.1, Eq. 4.33': 'the deviation factor printed 1.208 and v_inf_T 1.125 ft/s; Eq. '
                                                  '4.33 at 72.5° from horizontal gives 1.339 (develop SLIP-10: '
                                                  '1.3388), and 0.886 times 1.208 is 1.070',
    f'{HK}, §4.2.4, Field Example 4.1, Eq. 3.12': 'v_sg for annular flow printed 9.595 ft/s; its own expression '
                                                  'gives 9.613',
}


def params(backend):
    out = []
    for e in EXAMPLES:
        fn = e.python if backend == 'python' else e.rust
        if fn is None:
            continue
        marks = [pytest.mark.xfail(strict=True, reason=e.finding)] if e.finding else []
        test_id = f'{e.ids}: {e.source}: {e.quantity}'.replace('§', 'sec. ').replace('é', 'e').replace('°', 'deg ')
        out.append(pytest.param(e, fn, id=test_id, marks=marks))
    return out


@pytest.mark.parametrize('example, fn', params('python'))
def test_worked_example(example, fn):
    value = fn()
    assert agrees(value, example.printed), f'{example.quantity}: develop {value!r}, printed {example.printed}'


@pytest.mark.parametrize('example, fn', params('rust'))
def test_worked_example_rust(example, fn):
    value = fn()
    assert agrees(value, example.printed), f'{example.quantity}: the core {value!r}, printed {example.printed}'


def test_agrees_reads_the_printed_precision():
    assert agrees(0.56849, '0.568') and not agrees(0.56851, '0.568')
    assert agrees(778.6, '779') and not agrees(778.4, '779')
    assert agrees('slug', 'slug') and not agrees('churn', 'slug')


def test_every_example_has_a_source_and_a_backend():
    for e in EXAMPLES:
        assert e.source and e.ids and (e.python or e.rust), e
