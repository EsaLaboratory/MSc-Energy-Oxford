# Code Repository for MSc Energy Systems Dissertation
This Repository contains the code developed for the dissertation "Optimising Battery Energy Storage System Performance: Aging-Aware Strategies for UK Arbitrage and Frequency Response Markets" submitted as part of the 2023/24 MSc in Energy Systems. For any questions, feel free to contact the author [Frederik Schiele](mailto:frederik.schiele@eng.ox.ac.uk).

## Project Description
The aim of the project was to develop an MPC framework capable of analyzing aging-aware operating strategies for BESS participating in the UK arbitrage and DFR markets. Five strategies with varying levels of aging-awareness were developed and implemented. The MPC uses the open source battery model [SimSES]( https://doi.org/10.1016/j.est.2021.103743) for simulating the resulting power profiles.

## Table of Contents

1. [Code Structure](#code-structure)
2. [Using the Model](#using-the-model)
   - [Prerequisites and Setup](#prerequisites-and-setup)
   - [Running a Simulation](#running-a-simulation)
3. [Data](#Data)

## Code Structure

## Using the Model
Before using the model for the first time, please follow the instructions outlined in the setup section below. If you have done this befor, skip to the 
### Prerequisites and Setup
If you are attempting to run this model for the first time, please follow the setup instructions below. Make sure you have the following installed on your system:

- **Python** (version 3.7 or higher)
- **Git**

#### 1. Clone the repository
First, you need to clone the repository from GitHub. Open your terminal and run the following command:

```bash
git clone https://github.com/EsaLaboratory/MSc-Energy-Oxford.git
```
#### 2. Navigate to the Project Directory and set up a Virtual Environment
```bash
cd your-repository-name
python3 -m venv your_environment
```
This will create a directory named `your_environment` containing the virtual environment.

#### 3. Activate the Virtual Environment
Before installing the necessary dependencies, activate the virtual environment.

- on **macOS/Linux** run:
- ```bash
  source venv/bin/activate
  ```
- on **Windows** run:
- ```bash
  .\venv\Scripts\activate
  ```
You should see the name of the virtual environment `your_environment` in your terminal prompt, indicating that it's active.

#### 4. Install the Dependencies
With the virtual environment activated, install the necessary dependencies using the `dependencies.txt` file:

```bash
pip install -r dependencies.txt
```
#### 5. Add Gurobi License
This model uses the commercial solver Gurobi. Instructions on how to add a free license for academic users can be found [ here](https://www.gurobi.com/features/academic-named-user-license/).

### Running a Simulation
Simulations are run using the following command:

```python
python sim.py
```
Before running a simulation, a few configurations can be made.

- Adjust the underlying **operational strategy** by changing the optimizer used in the model. Currently the following operational strategies are implemented: No Legal Limits `optimizer_no_limits`, Baseline `optimizer_no_aging`, Cycle Limit `optimizer_daily_lim`, Cycle Aging `optimizer_cyc_aging` and PL Aging `optimizer_pl_aging`. To change the operational strategy used, modify [this]() line to contain the desired optimizer function.
  
- Change the **configuration** by modifying the corresponding files located in the `configs/` folder:
  - `analysis_optsim.ini`: Required for SimSES tool but no need to change
  - `optimization_optsim.ini`: Change inputs and parameters used by the optimization model
  - `simulation_optsim.ini`: Change inputs and parameters used for the battery model SimSES
    
- Change the desired **output path** for the results by modifying the file path [here]()


## Data
Some of the data used in this work is not available publically and can unfortunately not be shared in this repository. The input can be updated with newer data by replacing the respective `.csv` files in the `Data/` folder. Make sure the new `.csv` files follow the same structure to avoid errors and the filepath in the `optimization_optsim.ini` is adjusted accordingly.

- **Frequency Data:** Second-by-second frequency data is publically accessible from [NGESO](https://www.nationalgrideso.com/data-portal/system-frequency-data)

- **EAC Price Data:** Pricing for the DFR products including information on procured amounts is publically accessible from [NGESO](https://www.nationalgrideso.com/data-portal/eac-auction-results))

- **Half-hourly Energy Prices:** Unfortunately energy prices for the UK are not publically accessible. For the dissertation, an
free academic student acces was negotiated with [NordPool](https://www.nordpoolgroup.com/en/market-data12/Intraday/intraday-auction-uk/uk/evening-auction-17.30-bst/prices-and-volumes/half-hour/?view=table), however the prices cannot be published here. The price data contained in the `GB_energy_prices.csv` file is exemplary and shows the required format

- **Linearized Model**: The linearization of the degradation model was taken over from [this publication](https://doi.org/10.1016/j.apenergy.2023.121531)

