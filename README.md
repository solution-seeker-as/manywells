[![license](https://img.shields.io/badge/license-CC--BY--NC%204.0-success)]()

# ManyWells: simulation of multi-phase flow in vertical oil and gas wells

This code implements a steady-state drift-flux model for simulating two-phase (liquid and gas) flow in wells.
Three-phase flow (gas, oil, water) is supported by treating the oil and water as one mixed liquid phase.
Wells are modelled from the bottomhole pressure to the downstream choke pressure. 
Boundary conditions are introduced via an inflow model (bottomhole) and a choke model (topside).

The simulator has several use cases:
- Generate semi-realistic well production data
- Flow prediction by calibrating the model parameters to historical data
- Investigate sensitivities of variables of interest

### Project overview
The project is structured as follows
```
Project folder
|-- data
|   |-- manywells_sol       # Files related to the manywells-sol dataset  
|   |-- manywells_nsol      # Files related to the manywells-nsol dataset
|   |-- manywells_nscl      # Files related to the manywells-nscl dataset
|-- docs                    # Documentation
|-- manywells               # Implementation of simulator
|   |-- calibration         # Code for calibration to data
|   |-- closed_loop         # Simulation with closed-loop control
|-- scripts                 # Various scripts and examples
```

### Python environment
The Python environment required to run the scripts in the project is specified in ``environment.yml``. 
The ``manywells`` environment can be installed using the Conda package manager by running the command: 
```console
conda env create -f environment.yml
``` 

### Reference
If you use ManyWells in a scientific work we kindly ask you to cite it. 
You can cite it as shown in the bibtex entry below. 
```
TODO: Add reference to paper
```

### License
Manywells © 2024 by [Solution Seeker AS](https://solutionseeker.no) is licensed under 
[Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/?ref=chooser-v1). 
The license applies to all the resources contained in this project, including the code and datasets. 
The license can be found in the `LICENSE` file.
