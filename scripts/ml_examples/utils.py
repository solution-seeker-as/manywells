"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 26 January 2025
Erlend Lundby, erlend@solutionseeker.no
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def sample_n_fraction_per_task(df, task_col, fraction):
    """
    Randomly sample a fraction of data from each task.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    task_col (str): The column name representing the task.
    fraction (float): The fraction of data to sample from each task (0 < fraction <= 1).

    Returns:
    pd.DataFrame: A DataFrame containing the sampled data.
    """
    # Ensure the fraction is between 0 and 1
    if not (0 < fraction <= 1):
        raise ValueError("Fraction must be between 0 and 1.")

    # Group by the task column and apply sampling
    sampled_df = df.groupby(task_col).apply(lambda x: x.sample(frac=fraction)).reset_index(drop=True)

    return sampled_df


def standard_scaler(df, cols):
    """
    Standard normal scaling of the dataset
    :param df: dataframe
    :param cols: columns to scale
    :return: scaled train and test sets
    """
    scaler = StandardScaler()
    scaler.fit_transform(pd.concat([df[cols]],axis=0))
    df_scaled = df.copy()
    df_scaled[cols] = scaler.transform(df[cols])
    return df_scaled


def split_data_by_task(df, train_frac=0.8, test_frac=0.15, task_col='ID'):
    # Check that the fractions sum to 1
    val_frac = 1 - train_frac - test_frac
    assert train_frac + test_frac + val_frac == 1.0, "Fractions must sum to 1"

    # Create empty DataFrames for train, test, and validation sets
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    val_df = pd.DataFrame()

    # Group the data by the 'task' column
    grouped = df.groupby(task_col)

    for task, group in grouped:
        # Split the data for each task
        train, temp = train_test_split(group, test_size=(1 - train_frac))
        test, val = train_test_split(temp, test_size=(val_frac / (test_frac + val_frac)))

        # Append the splits to the respective DataFrames
        train_df = pd.concat([train_df, train])
        test_df = pd.concat([test_df, test])
        val_df = pd.concat([val_df, val])

    return train_df, test_df, val_df


def split_data_by_time_or_random(df, n1, n2, task_col='ID', time_col='WEEKS'):
    # Create empty DataFrames for the splits
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    # Group the data by the 'task' column
    grouped = df.groupby(task_col)

    for task, group in grouped:
        if time_col in group.columns:
            # Sort the group by the time column if it exists
            sorted_group = group.sort_values(by=time_col)

            # Calculate the maximum starting index for df1 to ensure df2 has at least n2 data points
            max_start_index = len(sorted_group) - n1 - n2

            if max_start_index < 0:
                raise ValueError(f"Not enough data points for task {task} to split into the desired sizes.")

            # Select a random starting index for df1
            start_index = np.random.randint(0, max_start_index + 1)

            # Split the data for each task
            df1_task = sorted_group.iloc[start_index:start_index + n1]
            df2_task = sorted_group.iloc[start_index + n1:start_index + n1 + n2]
        else:
            # Randomly sample n1 and n2 data points if the time column is not present
            if len(group) < n1 + n2:
                raise ValueError(f"Not enough data points for task {task} to split into the desired sizes.")

            # Randomly sample without replacement
            sampled_indices = np.random.choice(group.index, n1 + n2, replace=False)
            df1_task = group.loc[sampled_indices[:n1]]
            df2_task = group.loc[sampled_indices[n1:n1 + n2]]

        # Append the splits to the respective DataFrames
        df1 = pd.concat([df1, df1_task])
        df2 = pd.concat([df2, df2_task])

    return df1, df2


