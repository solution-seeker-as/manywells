# Machine learning examples
The ML examples are found in the following folders:
```
Project folder/|
├── scripts/ # Script folder
│ ├── ml_examples/ # Folder containing Scripts relevant to ML
│ │ └── classification.py # Classification problem examples
│ │ └── regression.py # Regression problem examples
│ │ └── data_loader.py # Loading datasets
│ │ └── noise.py # Noise script
│ │ └── base_model.py # Abstract model definition
│ │ └── utils.py # Functions for data splitting and scaling
```

# Walk-through of ML examples
The examples exist in the classification.py and regression.py files.

### Classification.py
```
1. Load dataset
2. Add gaussian noise
3. Scale data
4. Define list of examples n_tasks_. Each element in the list correspond to an
   experiment, and list value determines number of wells sharing data in that
   example
5. Identify and select tasks where all classes are present
6. Loop over number of runs
   - 6.1 Train and test split
   - 6.2 Loop over list of experiments
        - 6.2.1 Single task learning
            -Init models (one per task)
            - Fit models 
            - Calculate accuracy
        - 6.2.2 Multi-task learning
            - Loop over n_models MTL 
            - Calculate n_models number og MTL models (n_tasks/n_tasks_sharing_data)
            - Loop over n_models
                - Initialize model
                - Fit model
                - Calculate accuracy
7. Plot and store result
```                

### Regression.py
```
1. Load dataset
2. Add gaussian noise
3. Randomly select tasks
4. Scale data
5. Split data in train and test
6. Loop over experiments
    6.1 Single task learning
        - Initialize models (one per task)
        - Fit model
        - Calculate results
    6.2 Multi-task learning
        - Calculate n_models number of MTL models
        - Loop over MTL models
            - Link tasks to model
            - Initialize model
            - Fit model
            - Calculate results
    7. Plot and store results
```
