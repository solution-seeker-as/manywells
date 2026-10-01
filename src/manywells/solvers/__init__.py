"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

Solvers for the discretized system of a well (manywells.discretization): an Ipopt adapter (ipopt), the
initial-guess march (march), and the multi-start root search with the stability label (roots). They know nothing
about the physics.
"""
