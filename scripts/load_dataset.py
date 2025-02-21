"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 21 February 2025
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no 
"""

import pandas as pd


# Read manywells-sol-1 dataset
dataset_filename = '../data/manywells-sol/manywells-sol-1.zip'
dataset = pd.read_csv(dataset_filename, compression='zip')
print(dataset)

# Read manywells-sol-1 config
config_filename = '../data/manywells-sol/manywells-sol-1_config.zip'
config = pd.read_csv(config_filename, compression='zip')
print(config)



