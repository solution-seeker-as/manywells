"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 8 August 2024
Erlend Lundby, erlend@solutionseeker.no
"""

import numpy as np


class InitGuess:
    def __init__(self, downscale_pr=350, min_rate=3.0, min_distance_threshold=0.05):
        self.downscale_pr = downscale_pr
        self.x0_candidates = list()
        self.minimum_flowrate = min_rate
        self.min_distance_threshold = min_distance_threshold


    def list_fg_pr_scaled_coordinates(self):
        """
        :return: List of np array(fg,pr/self.downscale_pr) coordinates
        """
        lst = list()
        for c in self.x0_candidates:
            lst.append(np.array((c['f_g'], c['p_r']/self.downscale_pr)))
        return lst


    def closest_candidate(self,  well):
        """
        Find the x0 candidate closest to wells current gas fraction f_g and
        the downscaled reservoir pressure p_r, namely p_r/self.downscale_pr
        :param well: well object
        :return: x_guess of closest candidate if x0_candidates is not empty
        """
        if self.x0_candidates:
            closest_candidate = min(self.x0_candidates, key=lambda candidate:
                                    np.linalg.norm(
                                        np.array((candidate['f_g'], candidate['p_r']/self.downscale_pr)) -
                                        np.array((well.fractions[0], well.bc.p_r/self.downscale_pr))
                                    ))
            return closest_candidate['x_guess']
        else:
            return None


    def minimum_euclidian_distance(self, candidate):
        """
        Minimal euclidean distance in  (f_g, p_r/self.downscale_pr) coordinates from
        current well to set of x0_candidates
        :param candidate: New possible candidate to be included in x0_candidate
        :return: distance
        """
        if self.x0_candidates:
            well_coord = np.array((candidate['f_g'], candidate['p_r']/self.downscale_pr))
            lst_coordinates = self.list_fg_pr_scaled_coordinates()
            closest = lst_coordinates[min(range(len(lst_coordinates)), key=lambda i:
                                          np.linalg.norm(lst_coordinates[i] - well_coord))]
            return np.linalg.norm(well_coord - closest)
        else:
            return 1000.0


    def init_candidate(self, x, sim):
        """
        Check if x can be evaluated as a candidate for init x based on flow rate
        :param x: data from cells in well
        :param sim: Simulator object
        :return: bool
        """
        df_x = sim.solution_as_df(x)
        df_x['w_g'] = sim.wp.A * df_x['alpha'] * df_x['rho_g'] * df_x['v_g']
        df_x['w_l'] = sim.wp.A * (1 - df_x['alpha']) * df_x['rho_l'] * df_x['v_l']
        w_tot = df_x['w_g'] + df_x['w_l']
        wtot_out = w_tot.iloc[-1]
        if wtot_out < self.minimum_flowrate:
            # print('To low rate data')
            return False
        else:
            return True

    def create_candidate(self, x, sim, well):
        """
        Create candidate to be added to self.x0_candidates
        :param x: init guess candidate
        :param sim: simulator object
        :param well: well object
        :return: candidate if to be added to x0_candidates
        """
        if not self.init_candidate(x, sim):
            return None
        else:
            candidate = {
                'x_guess': x,
                'f_g': well.fractions[0],
                'p_r': well.bc.p_r,
            }
        if not self.x0_candidates:
            return candidate

        min_distance = self.minimum_euclidian_distance(candidate)
        if min_distance > self.min_distance_threshold:
            return candidate
        else:
            return None

    def add_candidate(self, x_guess, sim, well):
        """
        Add candidate to self.x0_candidates if candidate fulfill requirenments
        :param x_guess: Inital guess candidate
        :param sim: Simulator object
        :param well: Well object
        """
        x_guess_candidate = self.create_candidate(x_guess, sim, well)
        if x_guess_candidate is not None:
            self.x0_candidates.append(x_guess_candidate)
