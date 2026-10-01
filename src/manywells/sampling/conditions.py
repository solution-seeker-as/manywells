"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Operating-point draws (specs/sampling.md): the stationary samples of sol-1 around a well's nominal values
(SMP-18 to SMP-22), and the weekly samples of nsol-1, whose reservoir pressure decays and whose fractions follow a
random walk (SMP-23 to SMP-27).
"""

from dataclasses import dataclass

import numpy as np

from manywells.simulator import BoundaryConditions


def nominal_conditions(draw, u: float = 0.5) -> BoundaryConditions:
    """The well's nominal operating point at choke position u, without lift gas (SMP-28, step 1)."""
    return BoundaryConditions(p_r=draw.p_r, p_s=draw.p_s, T_r=draw.T_r, T_s=draw.T_s, u=u, w_lg=0.0)


def sample_conditions(draw, rng):
    """
    One stationary sample around the well's nominal values, in v1.0.0's order (`Well.sample_new_conditions`).

    :return: (BoundaryConditions, (f_g, f_o, f_w))
    """
    u = rng.uniform(0.05, 1)                                                     # spec: SMP-18
    w_lg = rng.uniform(0, 5) if draw.has_gas_lift else 0.0                       # spec: SMP-19
    p_s = rng.uniform(0.9 * draw.p_s, 1.1 * draw.p_s)                            # spec: SMP-20
    p_r = rng.uniform(0.98 * draw.p_r, 1.02 * draw.p_r)                          # spec: SMP-21
    f_g, f_o, f_w = draw.fractions                                               # spec: SMP-22
    new_f_g = min(0.99, rng.uniform(0.95 * f_g, 1.05 * f_g))
    wlf = f_w / (f_w + f_o)
    new_wlf = min(1.0, rng.uniform(0.95 * wlf, 1.05 * wlf))
    new_f_w = (1 - new_f_g) * new_wlf
    new_f_o = 1 - new_f_g - new_f_w
    bc = BoundaryConditions(p_r=float(p_r), p_s=float(p_s), T_r=draw.T_r, T_s=draw.T_s, u=float(u), w_lg=float(w_lg))
    return bc, (float(new_f_g), float(new_f_o), float(new_f_w))


@dataclass
class NonStationaryBehavior:
    """
    A well's evolution over its lifetime, in weeks from its first sample, as v1.0.0's `NonStationaryBehavior`.
    The state (the decay rate and the fractions) changes at every attempt, so it is a mutable object of the generator.
    """
    lifetime: float          # Years, SMP-23
    p_r_init: float          # Reservoir pressure at the first sample (bar)
    p_r_conv: float          # Reservoir pressure it decays towards (bar), SMP-25
    decay_rate: float        # gamma_pr, SMP-25
    decay_rate_noise: float  # gamma_pr(t_0) / 20
    p_s_init: float          # Nominal separator pressure (bar), SMP-26
    decay_g: float           # Gas fraction drift per year, SMP-24
    decay_o: float           # Oil fraction drift per year, SMP-24
    fractions: tuple         # Current (f_g, f_o, f_w)

    @classmethod
    def draw(cls, well, rng):
        """The well's behaviour, in v1.0.0's order of draws (`NonStationaryBehavior.__init__`)."""
        lifetime = rng.uniform(10, 20)                                           # spec: SMP-23
        p_r = well.p_r
        p_r_conv = p_r - rng.uniform(0.2 * p_r, 0.4 * p_r)                       # spec: SMP-25
        decay_rate = 1 - 0.01 ** (1 / lifetime)
        f_g, f_o, f_w = well.fractions
        decay_g = rng.uniform(f_g / 2, f_g) / lifetime                           # spec: SMP-24
        decay_o = rng.uniform(f_o / 2, f_o) / lifetime
        return cls(lifetime=float(lifetime), p_r_init=p_r, p_r_conv=float(p_r_conv), decay_rate=float(decay_rate),
                   decay_rate_noise=float(decay_rate / 20), p_s_init=well.p_s, decay_g=float(decay_g),
                   decay_o=float(decay_o), fractions=well.fractions)

    def _step(self, rng):  # spec: SMP-24
        """One week of the fractions' random walk."""
        f_g, f_o, _ = self.fractions
        g, o = 1.0, 1.0
        while g + o > 0.999:
            g = min(0.99, max(f_g - self.decay_g / 52 + rng.normal(0, 0.015), 0.002))
            o = min(0.99, max(f_o - self.decay_o / 52 + rng.normal(0, 0.015), 0.002))
        self.fractions = (float(g), float(o), float(1 - g - o))

    def update(self, well, week: int, week_prev: int, rng) -> BoundaryConditions:  # spec: SMP-26, SMP-27
        """
        The conditions of an attempt at week `week`, the last accepted sample having been at week `week_prev`, in
        v1.0.0's order (`NonStationaryWell.update_conditions`). The fractions take max(1, week - week_prev) steps
        from their current values, at every attempt (SMP-29).
        """
        for _ in range(max(1, week - week_prev)):
            self._step(rng)
        self.decay_rate = min(0.9, max(0.1, self.decay_rate + rng.uniform(-self.decay_rate_noise, self.decay_rate_noise)))
        p_r = (self.p_r_init - self.p_r_conv) * (1 - self.decay_rate) ** (week / 52) + self.p_r_conv   # spec: SMP-25
        p_s = rng.uniform(0.9 * self.p_s_init, 1.1 * self.p_s_init)
        u = rng.uniform(0.05, 1)
        w_lg = rng.uniform(0, 5) if well.has_gas_lift else 0.0
        return BoundaryConditions(p_r=float(p_r), p_s=float(p_s), T_r=well.T_r, T_s=well.T_s, u=float(u),
                                  w_lg=float(w_lg))
