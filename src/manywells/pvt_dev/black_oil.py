from math import log, exp


################################################
# Black Oil PVT model
################################################


class BlackOil: 
    def __init__(self, api, sg_gas, p_bubble, p_sep, t_sep):
        """
        Black Oil model

        :param api: Oil API (degrees)
        :param sg_gas: Gas specific gravity (relative to air)
        :param p_bubble: Bubble point pressure (psia)
        :param p_sep: Separator pressure (psia)
        :param t_sep: Separator temperature (Fahrenheit)
        """
        self.api = api
        self.sg_gas = sg_gas
        self.p_bubble = p_bubble
        self.p_sep = p_sep
        self.t_sep = t_sep

        if not (10 <= api <= 40):
            raise ValueError(f"Oil API {api} outside range of 10 to 40")

    def vazquez_beggs_pvt(self, p, t):
        return vazquez_beggs_pvt(p, t, self.api, self.sg_gas, self.p_sep, self.t_sep)


def vazquez_beggs_pvt(p, t, api, sg_gas, p_sep, t_sep):
    """
    Calculates Rs and Bo using Vazquez and Beggs correlations

    Reference: Vasquez, M., and H.D. Beggs. "Correlations for Fluid Physical Property Prediction." 
               J Pet Technol 32 (1980): 968–970. doi: https://doi.org/10.2118/6719-PA

    p: Pressure (psia)
    t: Temperature (Fahrenheit)
    api: Oil gravity (API)
    sg_gas: Gas specific gravity (at separator conditions)
    p_sep: Separator pressure (psia)
    t_sep: Separator temperature (Fahrenheit)
    """
    # 1. Correct Gas Gravity to Reference Separator Conditions (100 psig)
    sg_gas_corr = sg_gas * (1 + 5.912e-5 * api * t_sep * log(p_sep / 114.7))
    
    # 2. Determine Coefficients based on API gravity
    if api <= 30:
        c1, c2, c3 = 0.0362, 1.0937, 25.7240
        f1, f2, f3 = 4.677e-4, 1.751e-5, -1.811e-8
    else:
        c1, c2, c3 = 0.0178, 1.1870, 23.9310
        f1, f2, f3 = 4.670e-4, 1.100e-5, 1.337e-9

    # 3. Calculate Solution Gas-Oil Ratio (Rs) in scf/STB
    rs = c1 * sg_gas_corr * (p**c2) * exp(c3 * api / (t + 460))
    
    # 4. Calculate Oil Formation Volume Factor (Bo) in bbl/STB
    # This formula is for pressures at or below bubble point
    # bo = 1 + (f1 * rs) + (f2 * (t - 60) * (api / sg_gas_corr)) + (f3 * rs * (t - 60) * (api / sg_gas_corr))
    bo = 1 + (f1 * rs) + (f2 + f3 * rs) * (t - 60) * (api / sg_gas_corr)
    
    # TODO: Add Bo calculation for case where pressure is above bubble point
    # when p >= p_b, where p_b is the bubble point pressure
    # bo = bo_ref * exp(c0 * (p_b - p))
    # bo_ref is the oil formation volume factor at the bubble point pressure
    # c0 = (a1 + a2 * rs + a3 * t + a4 * sg_gas_corr + a5 * api) / (a6 * p)
    # a1 = -1433.0
    # a2 = 5.0
    # a3 = 17.2
    # a4 = -1180.0
    # a5 = 12.61
    # a6 = 10000.0

    return rs, bo




def live_oil_density(oil_api: float, gas_sg: float, rs: float, bo: float) -> float:
    """Live Oil Density, lbm/ft3

    Calculate the live density of the oil.

    Args:
        oil_api (float): Oil API Degrees
        gas_sg (float): Gas Specific Gravity, relative to air
        rs (float): Solubility of Gas in the oil, scf/stb
        bo (float): Oil FVF, rb/stb

    Returns:
        rho_oil (float): density of the oil, lbm/ft3

    References:
        - Applied Multiphase Flow in Pipes (2017) Al-Safran and Brill, Page 288, Eqn B-34
    """
    oil_sg = 141.5 / (oil_api + 131.5)  # oil specific gravity
    rho_oil = (62.42796 * oil_sg + (0.0136 * gas_sg * rs)) / bo
    return rho_oil


if __name__ == "__main__":
    # Example Usage:
    pres = 2000    # psia
    temp = 150     # deg F
    oil_api = 35   # 35 API (uses the >30 coefficients)
    gas_sg = 0.65
    p_s = 100      # separator pressure
    t_s = 70       # separator temp

    rs_val, bo_val = vazquez_beggs_pvt(pres, temp, oil_api, gas_sg, p_s, t_s)

    print(f"Solution GOR (Rs): {rs_val:.2f} scf/STB")
    print(f"Oil Formation Volume Factor (Bo): {bo_val:.4f} bbl/STB")

    rho_oil = live_oil_density(oil_api, gas_sg, rs_val, bo_val)

    print(f"Live Oil Density: {rho_oil:.4f} lbm/ft3")

    rho_oil_si = rho_oil * 0.453592 / 0.0283168  # From lbm/ft3 to kg/m3
    print(f"Live Oil Density: {rho_oil_si:.4f} kg/m3")
