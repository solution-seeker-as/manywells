"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 21 February 2025
Bjarne Grimstad, bjarne.grimstad@solutionseeker.no 
"""

import datasets


# Download 'manywells-sol-1'
data = datasets.load_dataset("solution-seeker-as/manywells", name='manywells-sol-1')

# Cast dataset to a Pandas DataFrame
df = data['train'].to_pandas()

# Print the data
print(df)

# Load config
config = datasets.load_dataset("solution-seeker-as/manywells", name='manywells-sol-1-config')

# Print config
print(config)
