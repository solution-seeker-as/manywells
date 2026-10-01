"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 01 October 2026
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no

The sampling procedure of the ManyWells datasets (specs/sampling.md): v1.0.0's procedure, ported to develop's model
and extended to its inputs. wells draws wells (SMP-1 to SMP-17, SMP-40 to SMP-44), conditions draws operating points
(SMP-18 to SMP-27), and generate solves them and applies the generators' acceptance rules (SMP-28, SMP-29).
Every draw comes from a generator seeded from the dataset's seed, the well and the draw (SMP-31).
"""
