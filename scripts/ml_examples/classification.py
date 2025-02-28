"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 26 January 2025
Erlend Lundby, erlend@solutionseeker.no
"""
import os
import statistics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from scripts.ml_examples.noise import add_noise
from scripts.ml_examples.base_model import LearningAlgorithm
from scripts.ml_examples.data_loader import load_data, load_config
from scripts.ml_examples.utils import standard_scaler, split_data_by_time_or_random
import scripts.plt_config as plt_config


# Define the columns for input and output
input_cols = ['QGAS', 'QGL', 'QLIQ', 'PWH']
output_cols = ['FRWH']

# Define the column for the task identifier
task_col = 'ID'

# Define columns to scale
scale_cols = ['QGAS', 'QGL', 'QLIQ', 'PWH']

class SingleTaskClassificationModel(LearningAlgorithm):
    def __init__(self, base_model, base_model_kwargs, input_cols, output_cols, task_col, df):
        """
        Initialize the ClassificationModel with a scikit-learn model.
        """
        self.base_model = base_model
        self.base_model_kwargs = base_model_kwargs
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.task_col = task_col
        self.models = {}
        self.label_encoder = LabelEncoder()
        assert df is not None, "Full DataFrame must be provided."
        self.label_encoder.fit(df[self.output_cols].values.ravel())

    def fit(self, df):
        for task in df[self.task_col].unique():
            task_mask = df[self.task_col] == task
            df_task = df[task_mask]
            input_data = df_task[self.input_cols]
            output_data = df_task[self.output_cols].values.ravel()
            # Check the number of unique classes
            # unique_classes = np.unique(output_data)
            # if len(unique_classes) <= 1:
            #     print(f"Skipping task {task} due to insufficient class diversity.")
            #     continue
            output_data = self.label_encoder.transform(output_data)
            output_data = output_data.ravel()
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
            # if task in self.models:
            #     pred = self.models[task].predict(input_data)
            #     predictions[task_mask] = pred
            # else:
            #     print(f"No model found for task {task}. Skipping predictions for this task.")

        return predictions

    def load(self):
        pass


class SingleTaskSVC(SingleTaskClassificationModel):
    def __init__(self,base_model_kwargs={'C':1000.0, 'gamma':0.1, 'cache_size':1000, 'class_weight': 'balanced'},
                 input_cols=input_cols, output_cols=output_cols, task_col=task_col, df=None):

        super().__init__(SVC,
                         base_model_kwargs,
                         input_cols,
                         output_cols,
                         task_col,
                         df)


class MultiTaskClassificationModel(LearningAlgorithm):
    def __init__(self, base_model, base_model_kwargs, input_cols, output_cols, task_col, df):
        """
        Initialize the ClassificationModel with a scikit-learn model.
        """
        self.base_model = base_model
        self.base_model_kwargs = base_model_kwargs if base_model_kwargs is not None else {}
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.task_col = task_col
        self.model = self.base_model(**self.base_model_kwargs)
        self.label_encoder = LabelEncoder()
        self.ohe_tasks = OneHotEncoder()
        self.ohe_tasks.fit(df[task_col].to_numpy().reshape(-1,1))
        self.ohe_cols = self.ohe_tasks.get_feature_names_out().tolist()
        #self.dummy_cols = None

        if df is not None:
            self.ohe_tasks.fit(df[task_col].to_numpy().reshape(-1, 1))
            self.ohe_cols = self.ohe_tasks.get_feature_names_out().tolist()
            self.label_encoder.fit(df[self.output_cols].values.ravel())

    def fit(self, df: pd.DataFrame):

        #task_dummies = pd.get_dummies(df[self.task_col], prefix='TASK')
        #self.dummy_cols = task_dummies.columns.tolist()

        x_tasks = self.ohe_tasks.transform(df[self.task_col].to_numpy().reshape(-1,1)).toarray()
        df_tasks_ohe = pd.DataFrame(x_tasks)
        df_tasks_ohe.columns = self.ohe_tasks.get_feature_names_out()
        df_train = pd.concat([df[self.input_cols + self.output_cols], df_tasks_ohe.set_index(df.index)], axis=1)
        #df_train = pd.concat([df[self.input_cols + self.output_cols], task_dummies], axis=1)
        a = self.input_cols + self.ohe_cols
        input_data = df_train[self.input_cols + self.ohe_cols]
        output_data = df_train[self.output_cols].values.ravel()

        output_data = self.label_encoder.transform(output_data)
        output_data = output_data.ravel()

        self.model.fit(input_data, output_data)

    def predict(self, df: pd.DataFrame):
        # Generate dummy variables for the task column
        #task_dummies = pd.get_dummies(df[self.task_col], prefix='TASK')
        x_tasks = self.ohe_tasks.transform(df[self.task_col].to_numpy().reshape(-1, 1)).toarray()
        df_tasks_ohe = pd.DataFrame(x_tasks)
        df_tasks_ohe.columns = self.ohe_tasks.get_feature_names_out()
        # Ensure all dummy columns are present
        # for col in self.dummy_cols:
        #     if col not in task_dummies:
        #         task_dummies[col] = 0

        # Reorder columns to match the training dummy columns
        # task_dummies = task_dummies[self.dummy_cols]
        input_data = pd.concat([df[self.input_cols], df_tasks_ohe.set_index(df.index)], axis=1)
        predictions = self.model.predict(input_data)
        return predictions

    def load(self):
        pass


class MultiTaskSVC(MultiTaskClassificationModel):
    def __init__(self,base_model_kwargs ={'C':1000.0, 'gamma':0.1, 'cache_size':1000, 'class_weight': 'balanced'},
                 input_cols=input_cols, output_cols=output_cols, task_col=task_col, df=None):
        #={'C':50.0, 'gamma':0.6, 'tol':1e-4, 'cache_size':1000}
        kwargs = base_model_kwargs if base_model_kwargs is not None else {}
        super().__init__(SVC,
                         kwargs,
                         input_cols,
                         output_cols,
                         task_col, df)

def accuracy(model, df):
    """
    Calculate the accuracy of the model on the provided DataFrame.
    """
    true_labels = df[model.output_cols].values.ravel()
    try:
        true_labels = model.label_encoder.transform(true_labels)
    except ValueError as e:
        print(f"Warning: {e}")
        # Handle unseen labels by mapping them to a special class or ignoring them
        # For simplicity, we'll ignore them in this example
        return None

    predictions = model.predict(df)
    correct_predictions = (predictions == true_labels).sum()
    accuracy = correct_predictions / len(true_labels)
    return accuracy


def select_and_replace_tasks(df, task_col, output_col, n_tasks):
    """
    Select and replace tasks to ensure all have more than one unique class for the output variable.

    Parameters:
    - df: DataFrame containing the data.
    - task_col: The column name representing the task identifier.
    - output_col: The column name representing the output variable.
    - n_tasks: The number of tasks to select.

    Returns:
    - A list of task identifiers that have more than one unique class.
    """
    # Find all tasks with more than one unique class
    valid_tasks = []
    for task in df[task_col].unique():
        task_data = df[df[task_col] == task]
        unique_classes = task_data[output_col].nunique()
        if unique_classes > 1:
            valid_tasks.append(task)

    # Shuffle the valid tasks to ensure randomness
    np.random.shuffle(valid_tasks)

    # Select the desired number of tasks
    selected_tasks = valid_tasks[:n_tasks]

    # If there are not enough valid tasks, raise an error
    if len(selected_tasks) < n_tasks:
        raise ValueError("Not enough tasks with multiple classes available to meet the requested number.")

    return selected_tasks


def plot_accuracy(accuracy_scores, n_tasks):
    plt_config.set_mpl_config()
    fig, ax = plt.subplots()
    ax.plot(n_tasks, accuracy_scores, marker='o', linestyle='-')
    ax.set_xlabel('Number of tasks')
    ax.set_ylabel('Class accuracy')
    ax.set_title('Performance')
    ax.grid(True)
    return fig


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")


if __name__ == '__main__':

    # Load dataset and dataset config
    version = 'v9_openloop_nonstationary'
    dataset_name = 'synth_v9_openloop_nonstationary'
    df_orig = load_data(version=version, dataset_name=dataset_name)
    config_orig = load_config(version=version, dataset_name=dataset_name)

    #Add gaussian noise to data
    df_noise, _ = add_noise(df_orig, config_orig, percentage_noise=0.5)

    #df = create_and_add_mass_fraction_cols(df_noise)
    df_scaled = standard_scaler(df_noise, scale_cols)

    #Define number of tasks to share data.
    # len(n_tasks_) number of experiments
    n_tasks_ = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Identify all classes in the dataset
    all_classes = df_scaled[output_cols[0]].unique()

    # Select tasks where all classes are present
    valid_ids = df_scaled.groupby(task_col)[output_cols[0]].nunique()
    valid_ids = valid_ids[valid_ids == len(all_classes)].index

    # Filter the DataFrame to only include valid tasks
    df = df_scaled[df_scaled[task_col].isin(valid_ids)]

    # Number of repeated experiments
    n_runs = 10
    # Initialize results matrices
    acc_mat = np.zeros((n_runs, len(n_tasks_)))

    # Loop over repeated experiments
    for k in range(n_runs):
        # Train-test split including tasks where all classes are present in both train and test
        train, test = split_data_by_time_or_random(df, n1=10, n2=300)
        train_classes = train[output_cols[0]].unique()
        valid_ids = train.groupby(task_col)[output_cols[0]].nunique()
        valid_ids = valid_ids[valid_ids == len(train_classes)].index
        train = train[train[task_col].isin(valid_ids)]
        test = test[test[task_col].isin(valid_ids)]
        selected_tasks = select_and_replace_tasks(train, task_col, output_cols[0], n_tasks_[-1])
        train = train[train[task_col].isin(selected_tasks)]
        test = test[test[task_col].isin(selected_tasks)]

        acc = []
        for i in range(len(n_tasks_)):
            n = n_tasks_[i]
            if n == 1:
                # Single task learning
                model = SingleTaskSVC(df=train)
                model.fit(train)
                result = accuracy(model, test)
                print(f'{n} tasks: accuracy: {result}')
                acc.append(result)
            else:
                # Multi-task learning
                n_models = int(len(selected_tasks) / n)
                acc_mtl = []

                for j in range(n_models):
                    model_tasks = selected_tasks[j * n:(j + 1) * n]
                    train_task = train[train[task_col].isin(model_tasks)]
                    test_task = test[test[task_col].isin(model_tasks)]
                    model = MultiTaskSVC(df=train_task)
                    model.fit(train_task)
                    result = accuracy(model, test_task)

                    acc_mtl.append(result)
                mean_acc = statistics.fmean(acc_mtl)
                print(f'{n} tasks: accuracy: {mean_acc}')
                acc.append(mean_acc)
        acc_mat[k, :] = acc
    acc_mean = acc_mat.mean(axis=0)
    fig = plot_accuracy(acc_mean, n_tasks_)
    store_path = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(store_path):
        os.makedirs(store_path)
    fig.savefig(os.path.join(store_path,'nscl_accuracy_classification.pdf'), bbox_inches='tight')
    plt.show()

