"""
Author: Oskar Feed Jakobsen (oskar@solutionseeker.no)
Date: 2026-07-21

This script shows how to set up and solve an optimization problem with
the rust simulator as a "black box". The following problem is considered:
 
Let

    F(w_m, w_lg) = (w_m - w_ref)**2 + c*w_lg**2,

for some reference total flow w_ref and scaling factor c.
Let G(x; w_lg) be the discretized DAE that the rust simulator solves, with w_lg as a parameter
and x as the solution vector containing [p, v_g, v_l, alpha, rho_g, rho_l, T] for all the cells.
The problem is:

    min F(w_m, w_lg)
    s.t. G(x, w_lg) = 0

The total flux w_m = w_l + w_g is given by the solution to G(x, w_lg) = 0. So, the optimization problem
reduces to a 1D one:
    
    min_{w_lg} F(w_m(w_lg), w_lg)

The paper adds the control variable s that parametrizes the choke u_c and the gas lift w_lg as
    
    u_c = min(1, s)
    w_lg = max(0, s - 1)

Thus, the 1D minimization problem is a problem in s instead.
It is solved in this script with scipy.optimize.minimize_scalar.
"""


from scipy.optimize import minimize_scalar
from manywells_rs import SSDFSimulator, SimError
from manywells.simulator import WellProperties, BoundaryConditions


STD_GRAVITY = 9.80665   # Standard acceleration of gravity (m/s²)
CF_PRES = 1e5           # Conversion factor for pressure: from bar to Pa

def pick_solution(solutions):
    """The new rust simulator may return multiple solutions.
    This function decides which one to pick."""
    assert len(solutions) != 0, "No solutions to pick from."
    assert isinstance(solutions[0], list), "solutions has to be a list containing the solution(s)"

    if len(solutions) == 1:
        # No choice
        return solutions[0]
    else:
        # What to do when there are multiple solutions?
        # solutions are sorted with highest PBH to lowest.
        return solutions[1]

def read_w_g_and_w_l_from_sol(x, simulator: SSDFSimulator):
    """extract w_g and w_l from first cell of the solution vector x.
    This is safe since these are constant across the entire well."""
    # state: [p, v_g, v_l, alpha, rho_g, rho_l, T]
    alpha, v_g, v_l, rho_g, rho_l, A = x[3], x[1], x[2], x[4], x[5], simulator.wp.A
    w_g = A*alpha*rho_g*v_g
    w_l = A*(1-alpha)*rho_l*v_l
    return w_g, w_l

def _objective(w_m, w_ref, w_lg, c):
    """The objective function used in the generation of the manywells-nscl dataset."""
    return  (w_m - w_ref) ** 2 + c * w_lg ** 2

def objective_wrapper(s_cv, w_ref, c, simulator: SSDFSimulator):
    """We wrap the _objective function to do the necessary compute.
    In the paper a control variable s was introduced in order to only have one control variable,
    and not two: the choke and the gas lift. The well is simulated, w_g and w_l is calculated and the
    "actual" objective function is calculated."""
    u_c = min(s_cv, 1.0)
    w_lg = max(s_cv - 1.0, 0.0)

    # update gas lift value and choke
    simulator.bc.w_lg = w_lg
    simulator.bc.u = u_c

    try:
        solutions = simulator.simulate()
        x = pick_solution(solutions) # TODO: How to decide which path to take?
    except SimError:
        return 1e12 # basically infinity!

    w_g, w_l = read_w_g_and_w_l_from_sol(x, simulator)
    w_m = w_g + w_l
    return _objective(w_m, w_ref, w_lg, c)


def solve_optimization_problem(simulator: SSDFSimulator, w_ref, has_gas_lift: True):
    c = 1
    s_bounds = (0, 6) if has_gas_lift else (0, 1)

    result = minimize_scalar(
        lambda s: objective_wrapper(s, w_ref, c, simulator),
        bounds=s_bounds,
        method="bounded"
    )

    s = result.x

    u_c = min(s, 1.0)
    w_lg = max(s - 1.0, 0.0)

    return u_c, w_lg


if __name__ == "__main__":
    # Set up the simulator

    well_properties = WellProperties()
    boundary_conditions = BoundaryConditions(u=0.9)

    simulator = SSDFSimulator(well_properties, boundary_conditions, n_cells = int(well_properties.L / 10))

    # Solve the optimization problem in the paper
    CHK, WLG = solve_optimization_problem(simulator, w_ref=15.0, has_gas_lift=boundary_conditions.w_lg > 0.0)
    

    # print
    print('---------------')
    print('CHK')
    print(CHK)
    print('----')
    print('WLG')
    print(WLG)
    print('---------------')