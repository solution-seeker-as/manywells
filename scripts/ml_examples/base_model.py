"""
Copyright (C) 2024 Solution Seeker AS - All Rights Reserved
You may use, distribute and modify this code under the
terms of the CC BY-NC 4.0 International Public License.

Created 26 November 2024
Kristian Løvland, kristian@solutionseeker.no
"""

from abc import ABC, abstractmethod
import pandas as pd


class LearningAlgorithm(ABC):
    """Interface that all models must implement for benchmarking"""

    @abstractmethod
    def __init__(self, u_cols: list, x_cols: list, y_col: str, time_col: str, task_col: str, problem_id: str = None):
        """Initialize model with required column specifications"""
        pass

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """Train model on provided DataFrame"""
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Generate predictions for provided DataFrame"""
        pass

    @abstractmethod
    def load(self):
        """Load a pre-trained model. Returns True if successful, False otherwise"""
        pass