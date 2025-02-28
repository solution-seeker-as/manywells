"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 26 January 2025
Erlend Lundby, erlend@solutionseeker.no
"""
import os
import statistics

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import numpy as np

from scripts.ml_examples.base_model import LearningAlgorithm
from scripts.ml_examples.data_loader import load_data, load_config
from scripts.ml_examples.utils import standard_scaler, split_data_by_time_or_random
from scripts.ml_examples.noise import add_noise
import scripts.plt_config as plt_config


# Define the columns for input and output
input_cols = ['CHK', 'QGL', 'PWH', 'PDC', 'TWH', 'FOIL', 'FGAS']
output_cols = ['WTOT']

# Define the column for the task identifier
task_col ='ID'

# Define columns to scale
scale_cols = ['QGL', 'PWH', 'PDC', 'TWH']


class SingleTaskRegressionModel(LearningAlgorithm):
    def __init__(self, base_model, base_model_kwargs, input_cols, output_cols, task_col):
        """
        Initialize the RegressionModel with a scikit-learn model.

        Parameters:
        model (object): A scikit-learn regression model instance.
        """
        self.base_model = base_model
        self.base_model_kwargs = base_model_kwargs
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.task_col = task_col
        self.models = {}

    def fit(self, df):
        for task in df[self.task_col].unique():
            task_mask = df[self.task_col] == task
            df_task = df[task_mask]
            input_data = df_task[self.input_cols]
            output_data = df_task[self.output_cols].values.ravel()

            model = self.base_model(**self.base_model_kwargs)
            model.fit(input_data, output_data)
            self.models[task] = model

    def predict(self, df):
        predictions = pd.Series(index=df.index)
        for task in df[self.task_col].unique():
            task_mask = df[self.task_col] == task
            df_task = df[task_mask]
            input_data = df_task[self.input_cols]

            pred = self.models[task].predict(input_data)
            predictions[task_mask] = pred

        return predictions

    def load(self):
        pass

class SingleTaskSVR(SingleTaskRegressionModel):
    def __init__(self, base_model_kwargs,
                 input_cols, output_cols, task_col):
        #base_model_kwargs = {'C': 100.0, 'gamma': 0.4, 'epsilon': 0.1, 'tol': 1e-4, 'cache_size': 1000}
        super().__init__(SVR,
                         base_model_kwargs,
                         input_cols,
                         output_cols,
                         task_col)


class MTLRegressionModel(LearningAlgorithm):
    def __init__(self, base_model, base_model_kwargs, input_cols, output_cols, task_col):
        """
        Initialize the MTLRegressionModel with scikit-learn regression models.
        """
        self.base_model = base_model
        self.base_model_kwargs = base_model_kwargs
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.task_col = task_col
        self.model = None
        self.dummy_cols = None

    def fit(self, df: pd.DataFrame):
        task_dummies = pd.get_dummies(df[self.task_col], prefix='TASK')
        self.dummy_cols = task_dummies.columns.tolist()
        df_train = pd.concat([df[self.input_cols + self.output_cols], task_dummies], axis=1)
        X_train = df_train[self.input_cols + self.dummy_cols]
        y_train = df_train[self.output_cols].values.ravel()
        self.model = self.base_model(**self.base_model_kwargs)
        self.model.fit(X_train, y_train)

    def predict(self, df: pd.DataFrame):
        # Generate dummy variables for the task column
        task_dummies = pd.get_dummies(df[self.task_col], prefix='TASK')

        # Ensure all dummy columns are present
        for col in self.dummy_cols:
            if col not in task_dummies:
                task_dummies[col] = 0

        # Reorder columns to match the training dummy columns
        task_dummies = task_dummies[self.dummy_cols]
        X_pred = pd.concat([df[self.input_cols], task_dummies], axis=1)
        predictions = self.model.predict(X_pred)
        return predictions

    def load(self):
        pass


class MultiTaskSVR(MTLRegressionModel):
    def __init__(self, base_model_kwargs, input_cols, output_cols, task_col):
        #base_model_kwargs = {'C': 100.0, 'gamma': 0.4, 'epsilon': 0.1, 'tol': 1e-4, 'cache_size': 1000}
        super().__init__(SVR,
                         base_model_kwargs,
                         input_cols,
                         output_cols,
                         task_col)


def evaluate(model, df):
    y_pred = model.predict(df)
    y_test = df[model.output_cols]
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {'mean_squared_error': mse, 'r2_score': r2}


def plot_mse(n_tasks, mse_scores):
    plt_config.set_mpl_config()
    fig, ax = plt.subplots()
    ax.plot(n_tasks, mse_scores, marker='o', linestyle='-')
    ax.set_xlabel('Number of tasks')
    ax.set_ylabel('Mean squared error')
    ax.set_title('Performance')
    ax.grid(True)
    return fig


if __name__ == '__main__':

    # Load dataset and dataset config
    df_orig = load_data(version='v9', dataset_name='synth_v9')
    config_orig = load_config(version='v9', dataset_name='synth_v9')

    # Add gaussian noise to the data
    df_noise, _ = add_noise(df_orig,config_orig,percentage_noise=0.5)

    # Randomly select tasks to be included in experiments
    unique_tasks = df_orig[task_col].unique()
    np.random.shuffle(unique_tasks)
    # Number of tasks that share data, one experiment per element in n_tasks_
    n_tasks_ = [1,2,5,10,20,50,100, 250, 500] #, 200, 500
    selected_tasks = unique_tasks[:n_tasks_[-1]]
    df = df_noise[df_noise[task_col].isin(selected_tasks)]

    #Scale data
    df_scaled = standard_scaler(df, scale_cols)

    #Slit data into train and test
    train_scaled, test_scaled = split_data_by_time_or_random(df_scaled, n1=10, n2=30, task_col='ID', time_col='WEEKS')

    # Initialize result lists
    mse = []
    r2 = []
    for i in range(len(n_tasks_)):
        n = n_tasks_[i]
        if n==1:
            # Single-task learning
            SVR_kwargs = {'C': 100.0, 'gamma': 0.4, 'epsilon': 0.1, 'tol': 1e-4, 'cache_size': 1000}
            model = SingleTaskSVR(SVR_kwargs,input_cols,output_cols, task_col)
            model.fit(train_scaled)
            result = evaluate(model, test_scaled)
            print(f'{n} tasks: mse: {result["mean_squared_error"]}, r2: {result["r2_score"]}')
            mse.append(result['mean_squared_error'])
            r2.append(result['r2_score'])
        else:
            # Multi-task learning
            n_models = int(len(selected_tasks)/n)
            mse_mtl = []
            r2_mtl = []
            for j in range(n_models):
                model_tasks = selected_tasks[j * n:(j + 1)*n]
                task_dummies = pd.get_dummies(df[df[task_col].isin(model_tasks)][task_col], prefix='TASK')
                dummy_cols = task_dummies.columns.tolist()
                SVR_kwargs = {'C': 100.0, 'gamma': 0.4, 'epsilon': 0.1, 'tol': 1e-4, 'cache_size': 1000}
                model = MultiTaskSVR(SVR_kwargs, input_cols, output_cols, task_col)
                train_task = train_scaled[train_scaled[task_col].isin(model_tasks)]
                test_task = test_scaled[test_scaled[task_col].isin(model_tasks)]
                model.fit(train_task)
                result = evaluate(model, test_task)
                mse_mtl.append(result['mean_squared_error'])
                r2_mtl.append(result['r2_score'])
            mean_mse = statistics.fmean(mse_mtl)
            mse.append(mean_mse)
            mean_r2 = statistics.fmean(r2_mtl)
            print(f'{n} tasks: mse: {mean_mse}, r2: {mean_r2}')
            r2.append(mean_r2)

    fig_mse = plot_mse(n_tasks_, mse)
    store_path = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    fig_mse.savefig(os.path.join(store_path, 'sol_mse_regression.pdf'), bbox_inches='tight')
    plt.show()
