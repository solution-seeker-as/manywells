# Dataset description

The `data` folder contains synthetic well data generated using the code in `scripts/data_generation/`. 
The generated data is stored in data dumps (pickle files), which are later compiled to a dataset 
by running the script `read_dump.py` (for stationary datasets) or `read_dump_nonstationary.py` 
(for non-stationary datasets). 

### Dataset features

| Feature | Description                                                       |  Unit |
|:--------|:------------------------------------------------------------------|------:|
| ID      | Unique well ID                                                    |     - |
| CHK     | Choke position in \[0, 1\]                                        |     - |
| PBH     | Pressure bottomhole                                               |   bar |
| PWH     | Pressure wellhead                                                 |   bar |
| PDC     | Pressure downstream choke                                         |   bar |
| TBH     | Temperature bottomhole                                            |     K |
| TWH     | Temperature wellhead                                              |     K |
| WGL     | Mass flow rate, lift gas                                          |  kg/s |
| WGAS    | Mass flow rate, gas excl. lift gas                                |  kg/s |
| WLIQ    | Mass flow rate, liquid                                            |  kg/s |
| WOIL    | Mass flow rate, oil                                               |  kg/s |
| WWAT    | Mass flow rate, water                                             |  kg/s |
| WTOT    | Mass flow rate, total incl. lift gas                              |  kg/s |
| QGL     | Volumetric flow rate at standard conditions, lift gas             | Sm³/h |
| QGAS    | Volumetric flow rate at standard conditions, gas excl. lift gas   | Sm³/h |
| QLIQ    | Volumetric flow rate at standard conditions, liquid               | Sm³/h |
| QOIL    | Volumetric flow rate at standard conditions, oil                  | Sm³/h |
| QWAT    | Volumetric flow rate at standard conditions, water                | Sm³/h |
| QTOT    | Volumetric flow rate at standard conditions, total incl. lift gas | Sm³/h |
| FGAS    | Inflow gas mass fraction                                          |     - |
| FOIL    | Inflow oil mass fraction                                          |     - |
| FWAT    | Inflow water mass fraction                                        |     - |
| WEEKS   | Weeks since first data point                                      |    7d |
| CHOKED  | Boolean indicating if flow is choked                              |     - |
| FRBH    | Flow regime at bottomhole (represented by string)                 |     - |
| FRWH    | Flow regime at wellhead (represented by string)                   |     - |

The following relations apply:
- WLIQ = WOIL + WWAT
- WTOT = WLIQ + WGAS + WGL
- QLIQ = QOIL + QWAT
- QTOT = QLIQ + QGAS + QGL

The flow regime features, FRBH and FRWH, may take on one of the following string values: 
'bubbly', 'slug-churn', 'annular'. 

The WEEKS feature is only present in non-stationary datasets.
