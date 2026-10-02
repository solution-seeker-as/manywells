# Dataset description

The published ManyWells datasets are on [Hugging Face](https://huggingface.co/datasets/solution-seeker-as/manywells):
`manywells-sol-1` (stationary, open loop), `manywells-nsol-1` (non-stationary, open loop) and `manywells-nscl-1`
(non-stationary, closed loop). Each has one million samples, 500 from each of 2000 wells, generated with ManyWells
v1.0.0, and a config file, `<dataset>_config.zip`, with the parameters each well was simulated with. The
[README](../README.md#datasets) shows how to load them, and [`corrigendum.md`](corrigendum.md) lists their errata.
The published datasets are never modified.

The generators in `scripts/data_generation/` write new datasets with the same features
([`simulate.md`](simulate.md), Generating datasets). Their config file holds each well's draws
(`manywells.sampling.wells.WellDraw`), whose columns differ from those of the published config files.
v1.0.0's generators wrote pickle files, which its scripts `read_dump.py` and `read_dump_nonstationary.py`
compiled into the published datasets; they are at the
[`v1.0.0` tag](https://github.com/solution-seeker-as/manywells/tree/v1.0.0/scripts/data_generation).

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
