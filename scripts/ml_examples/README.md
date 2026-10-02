# Machine learning examples

The machine learning examples of the ManyWells paper. Each compares single-task learning, one model per well, with
multi-task learning, one model shared by a group of wells with the well's ID as a one-hot input, as the number of
wells that share a model grows. Both use support vector machines from scikit-learn on the published
`manywells-sol-1` dataset, which `data_loader.py` downloads from
[Hugging Face](https://huggingface.co/datasets/solution-seeker-as/manywells) on first use. They read the dataset
only and do not run the simulator.

| Script | Model | Inputs | Output |
|---|---|---|---|
| `regression.py` | SVR | `CHK`, `QGL`, `PWH`, `PDC`, `TWH`, `FOIL`, `FGAS` | `WTOT`, the total mass rate |
| `classification.py` | SVC | `QGAS`, `QGL`, `QLIQ`, `PWH` | `FRWH`, the flow regime at the wellhead |

The features are defined in [`docs/datasets.md`](../../docs/datasets.md).

## Running

From the project root:

```console
uv run python -m scripts.ml_examples.regression
uv run python -m scripts.ml_examples.classification
```

Each script prints the score for each number of wells per model, plots it against that number and saves the plot in
`results/` in the working directory (`sol_mse_regression.pdf` and `nscl_accuracy_classification.pdf`). Each takes
about 3 minutes on one core after the download, most of it adding the noise. No seed is set, so the scores vary
from run to run.

## Procedure

1. Load the dataset and its config, and add Gaussian noise to the measured features with a standard deviation of
   0.5% of each feature's mean (`noise.py`). The noise of the derived rates is derived in turn, so that `WLIQ` is
   still `WOIL + WWAT`, and the volumetric rates follow from the mass rates and the densities in the config.
2. Scale the inputs that are not fractions or the choke position to zero mean and unit variance (`utils.py`).
3. Draw a training and a test set from each well: 10 and 30 samples from each of 500 random wells for the
   regression, and 10 and 300 samples from 100 random wells in which every flow regime occurs for the classification.
4. For each number of wells per model n (1 to 500 for the regression, 1 to 100 for the classification), split the
   wells into groups of n and fit one model per group; n = 1 is single-task learning. Score each model on its
   wells' test sets: mean squared error and R² for the regression, accuracy for the classification, averaged over
   the groups. The classification repeats steps 3 and 4 ten times and averages the accuracies.

## Files

- `regression.py`, `classification.py`: the examples and their models
- `data_loader.py`: downloads a published dataset and its config
- `noise.py`: the noise model; run on its own, it plots the noise added to `manywells-sol-1`
- `utils.py`: scaling and the train-test splits
- `base_model.py`: the interface the models implement (`LearningAlgorithm`)
