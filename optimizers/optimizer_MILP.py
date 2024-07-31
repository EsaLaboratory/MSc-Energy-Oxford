import gurobipy as gp
from gurobipy import GRB

def optimizer_no_aging(hours, timestep_opt_s, epex_price_GBP_MWh, 
                       clearing_prices, efficiency, max_power_MW, capacity_MWh, starting_soc, 
                       DC_high_response, DC_low_response, DM_high_response,
                       DM_low_response, DR_high_response, DR_low_response):
    
    
    #initialize the gurobi model
    model = gp.Model('FFR_Maximize_Profit')
    model.setParam('OutputFlag',1)

    #set mip gap
    model.setParam('MIPGap', 0.1)

    #add the decision variables
    capacity_allocation = model.addVars(hours//4*6, lb=0, ub=max_power_MW, name="capacity_allocation")
    baseline_charge = model.addVars(hours*2, lb=0, ub=max_power_MW, name="baseline_charge")
    baseline_discharge = model.addVars(hours*2, lb=0, ub=max_power_MW, name="baseline_discharge")

    #add auxiliary variables
    soc = model.addVars(int(3600/timestep_opt_s*hours+1),lb = 0, ub = 1, name="soc")
    power = model.addVars(int(3600/timestep_opt_s*hours),lb = -max_power_MW, ub = max_power_MW, name = 'power')
    charge_status = model.addVars(int(3600/timestep_opt_s*hours), vtype = GRB.BINARY)

    #add the objective function
    #calculate the value of the energy bought or sold through baselines --> division by two is to convert half hourly
    #settlement periods int o MWh
    value_recharge_GBP = (-1*gp.quicksum(baseline_charge[i]*epex_price_GBP_MWh[i] for i in range(len(baseline_charge)))
                        + gp.quicksum(baseline_discharge[i]*epex_price_GBP_MWh[i] for i in range(len(baseline_discharge))))/2

    #calculate the value of the FFR services provided --> multiplication by 4 is to convert hourly clearing prices to 4-hour
    #EAC window
    value_FFR_GBP = gp.quicksum(clearing_prices[i]*capacity_allocation[i] for i in range(len(capacity_allocation)))*4

    model.setObjective(value_recharge_GBP + value_FFR_GBP, GRB.MAXIMIZE)

    #--------------------CONSTRAINTS--------------------
    #manage SOC evolution
    # Precompute values to avoid redundant calculations
    eff_div_capacity_MWh = efficiency / (3600/timestep_opt_s * capacity_MWh)
    inv_eff_div_capacity_MWh = 1 / (efficiency * 3600/timestep_opt_s * capacity_MWh)

    # Add power evolution constraint
    model.addConstrs(power[i] == (baseline_charge[(i)//(1800/timestep_opt_s)] +
        DC_high_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6] +
        DM_high_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6+2] +
        DR_high_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6+4]) -
        (baseline_discharge[(i)//(1800/timestep_opt_s)] +
        DC_low_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6+1] +
        DM_low_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6+3] +
        DR_low_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6+5])
        for i in range(0, int(3600/timestep_opt_s*hours)))
    
    #add charging status constraint
    # Add constraints in a loop
    M = 1e6
    epsilon = 1e-6
    for i in range(int(3600 / timestep_opt_s * hours)):
        model.addConstr(power[i] >= epsilon - M * (1 - charge_status[i]), name=f"charge_status_1_{i}")
        model.addConstr(power[i] <= M * charge_status[i], name=f"charge_status_2_{i}")

    # Add SOC constraints in bulk
    model.addConstr(soc[0] == starting_soc)

    model.addConstrs(
        (soc[i] == soc[i-1] + power[i-1]/capacity_MWh/3600*timestep_opt_s * (1.05-0.1*charge_status[i-1]))
        for i in range(1,len(soc))
    )

    # Add maximum power constraints for charge in bulk
    model.addConstrs(
        (baseline_charge[(i-1)//(1800/timestep_opt_s)] +
        DC_high_response[i-1] * capacity_allocation[((i-1)//(14400/timestep_opt_s))*6] +
        DM_high_response[i-1] * capacity_allocation[((i-1)//(14400/timestep_opt_s))*6+2] +
        DR_high_response[i-1] * capacity_allocation[((i-1)//(14400/timestep_opt_s))*6+4] <= max_power_MW)
        for i in range(1, int(3600/timestep_opt_s*hours+1))
    )

    # Add maximum power constraints for discharge in bulk
    model.addConstrs(
        (baseline_discharge[(i-1)//(1800/timestep_opt_s)] +
        DC_low_response[i-1] * capacity_allocation[((i-1)//(14400/timestep_opt_s))*6+1] +
        DM_low_response[i-1] * capacity_allocation[((i-1)//(14400/timestep_opt_s))*6+3] +
        DR_low_response[i-1] * capacity_allocation[((i-1)//(14400/timestep_opt_s))*6+5] <= max_power_MW)
        for i in range(1, int(3600/timestep_opt_s*hours+1))
    )

    #add soc constraint to be within 0 and 1
    model.addConstrs((soc[i] <= 1 for i in range(len(soc))))
    model.addConstrs((soc[i] >= 0 for i in range(len(soc))))

    model.optimize()

    #retreive the optimal solution
    daily_profit_GBP = model.objVal

    #retreive the optimal capacity allocation
    optimal_capacity_allocation_MW = []
    for i in range(len(capacity_allocation)):
        optimal_capacity_allocation_MW.append(capacity_allocation[i].x)

    #retreive the optimal baseline charge
    optimal_baseline_charge_MW = []
    for i in range(len(baseline_charge)):
        optimal_baseline_charge_MW.append(baseline_charge[i].x)
    
    #retreive the optimal baseline discharge
    optimal_baseline_discharge_MW = []
    for i in range(len(baseline_discharge)):
        optimal_baseline_discharge_MW.append(baseline_discharge[i].x)
    
    power_output = []
    for i in range(len(charge_status)):
        power_output.append(power[i].x)
    
    #retreive the optimal SOC as well as cycle count
    cycles = 0
    soc_results = []
    for i in range(len(soc)):
        soc_results.append(soc[i].x)
        if i > 0:
            cycles += abs(soc_results[-1] - soc_results[-2])
    
    return daily_profit_GBP, optimal_capacity_allocation_MW, optimal_baseline_charge_MW, optimal_baseline_discharge_MW, soc_results, cycles/2

def optimizer_no_aging_legal_limits(hours, timestep_opt_s, epex_price_GBP_MWh, 
                       clearing_prices, efficiency, max_power_MW, capacity_MWh, starting_soc, 
                       DC_high_response, DC_low_response, DM_high_response,
                       DM_low_response, DR_high_response, DR_low_response):
    
    
    #initialize the gurobi model
    model = gp.Model('FFR_Maximize_Profit')
    model.setParam('OutputFlag',1)

    #set mip gap
    model.setParam('MIPGap', 0.1)

    #add the decision variables
    capacity_allocation = model.addVars(hours//4*6, lb=0, ub=max_power_MW, name="capacity_allocation")
    baseline_charge = model.addVars(hours*2, lb=0, ub=max_power_MW, name="baseline_charge")
    baseline_discharge = model.addVars(hours*2, lb=0, ub=max_power_MW, name="baseline_discharge")

    #add auxiliary variables
    soc = model.addVars(int(3600/timestep_opt_s*hours+1),lb = 0, ub = 1, name="soc")
    power = model.addVars(int(3600/timestep_opt_s*hours),lb = -max_power_MW, ub = max_power_MW, name = 'power')
    charge_status = model.addVars(int(3600/timestep_opt_s*hours), vtype = GRB.BINARY)

    #add the objective function
    #calculate the value of the energy bought or sold through baselines --> division by two is to convert half hourly
    #settlement periods int o MWh
    value_recharge_GBP = (-1*gp.quicksum(baseline_charge[i]*epex_price_GBP_MWh[i] for i in range(len(baseline_charge)))
                        + gp.quicksum(baseline_discharge[i]*epex_price_GBP_MWh[i] for i in range(len(baseline_discharge))))/2

    #calculate the value of the FFR services provided --> multiplication by 4 is to convert hourly clearing prices to 4-hour
    #EAC window
    value_FFR_GBP = gp.quicksum(clearing_prices[i]*capacity_allocation[i] for i in range(len(capacity_allocation)))*4

    model.setObjective(value_recharge_GBP + value_FFR_GBP, GRB.MAXIMIZE)

    #--------------------CONSTRAINTS--------------------
    #manage SOC evolution
    # Precompute values to avoid redundant calculations
    eff_div_capacity_MWh = efficiency / (3600/timestep_opt_s * capacity_MWh)
    inv_eff_div_capacity_MWh = 1 / (efficiency * 3600/timestep_opt_s * capacity_MWh)

    # Add power evolution constraint
    model.addConstrs(power[i] == (baseline_charge[(i)//(1800/timestep_opt_s)] +
        DC_high_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6] +
        DM_high_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6+2] +
        DR_high_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6+4]) -
        (baseline_discharge[(i)//(1800/timestep_opt_s)] +
        DC_low_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6+1] +
        DM_low_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6+3] +
        DR_low_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6+5])
        for i in range(0, int(3600/timestep_opt_s*hours)))
    
    #add charging status constraint
    # Add constraints in a loop
    M = 1e6
    epsilon = 1e-6
    for i in range(int(3600 / timestep_opt_s * hours)):
        model.addConstr(power[i] >= epsilon - M * (1 - charge_status[i]), name=f"charge_status_1_{i}")
        model.addConstr(power[i] <= M * charge_status[i], name=f"charge_status_2_{i}")

    # Add SOC constraints in bulk
    model.addConstr(soc[0] == starting_soc)

    model.addConstrs(
        (soc[i] == soc[i-1] + power[i-1]/capacity_MWh/3600*timestep_opt_s * (1.05-0.1*charge_status[i-1]))
        for i in range(1,len(soc))
    )

    # Add maximum power constraints for charge in bulk
    model.addConstrs(
        (baseline_charge[(i-1)//(1800/timestep_opt_s)] +
        DC_high_response[i-1] * capacity_allocation[((i-1)//(14400/timestep_opt_s))*6] +
        DM_high_response[i-1] * capacity_allocation[((i-1)//(14400/timestep_opt_s))*6+2] +
        DR_high_response[i-1] * capacity_allocation[((i-1)//(14400/timestep_opt_s))*6+4] <= max_power_MW)
        for i in range(1, int(3600/timestep_opt_s*hours+1))
    )

    # Add maximum power constraints for discharge in bulk
    model.addConstrs(
        (baseline_discharge[(i-1)//(1800/timestep_opt_s)] +
        DC_low_response[i-1] * capacity_allocation[((i-1)//(14400/timestep_opt_s))*6+1] +
        DM_low_response[i-1] * capacity_allocation[((i-1)//(14400/timestep_opt_s))*6+3] +
        DR_low_response[i-1] * capacity_allocation[((i-1)//(14400/timestep_opt_s))*6+5] <= max_power_MW)
        for i in range(1, int(3600/timestep_opt_s*hours+1))
    )

    #add soc constraint to be within 0 and 1
    model.addConstrs((soc[i] <= 1 for i in range(len(soc))))
    model.addConstrs((soc[i] >= 0 for i in range(len(soc))))

    #add soc constraint to comply with legal limits
    model.addConstrs((soc[i*1800/timestep_opt_s] <= 1-(0.25*capacity_allocation[i//8]+0.5*capacity_allocation[i//8+2]+
                    capacity_allocation[i//8+4])/capacity_MWh for i in range(hours*2)))
    model.addConstrs((soc[i*1800/timestep_opt_s] >= (0.25*capacity_allocation[i//8+1]+0.5*capacity_allocation[i//8+3]+
                    capacity_allocation[i//8+5])/capacity_MWh for i in range(hours*2)))

    model.optimize()

    #retreive the optimal solution
    daily_profit_GBP = model.objVal

    #retreive the optimal capacity allocation
    optimal_capacity_allocation_MW = []
    for i in range(len(capacity_allocation)):
        optimal_capacity_allocation_MW.append(capacity_allocation[i].x)

    #retreive the optimal baseline charge
    optimal_baseline_charge_MW = []
    for i in range(len(baseline_charge)):
        optimal_baseline_charge_MW.append(baseline_charge[i].x)
    
    #retreive the optimal baseline discharge
    optimal_baseline_discharge_MW = []
    for i in range(len(baseline_discharge)):
        optimal_baseline_discharge_MW.append(baseline_discharge[i].x)
    
    power_output = []
    for i in range(len(charge_status)):
        power_output.append(power[i].x)
    
    #retreive the optimal SOC as well as cycle count
    cycles = 0
    soc_results = []
    for i in range(len(soc)):
        soc_results.append(soc[i].x)
        if i > 0:
            cycles += abs(soc_results[-1] - soc_results[-2])
    
    return daily_profit_GBP, optimal_capacity_allocation_MW, optimal_baseline_charge_MW, optimal_baseline_discharge_MW, soc_results, cycles/2



import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


def optimizer_no_aging_linprog(hours, timestep_opt_s, epex_price_GBP_MWh, 
                       clearing_prices, efficiency, max_power_MW, capacity_MWh, starting_soc, 
                       DC_high_response, DC_low_response, DM_high_response,
                       DM_low_response, DR_high_response, DR_low_response):

    num_timesteps = int(3600 / timestep_opt_s * hours)
    num_4hour_periods = (hours // 4) * 6
    num_baseline_charge_discharge = hours * 2


    # Decision variables: capacity_allocation, baseline_charge, baseline_discharge, SOC
    n_vars = num_4hour_periods + num_baseline_charge_discharge * 2 + num_timesteps + 1

    # Objective function
    c = np.zeros(n_vars)
    for i in range(num_baseline_charge_discharge):
        c[num_4hour_periods + i] = epex_price_GBP_MWh[i] / 2  # Charging
        c[num_4hour_periods + num_baseline_charge_discharge + i] = -epex_price_GBP_MWh[i] / 2  # Discharging
    for i in range(num_4hour_periods):
        c[i] = -clearing_prices[i] * 4

    # Inequality constraints (A_ub * x <= b_ub)
    A_ub = lil_matrix((2 * num_timesteps, n_vars))
    b_ub = np.zeros(2 * num_timesteps)

    # Max power constraints for charge and discharge
    for i in range(num_timesteps):
        hour_index = i // (1800 // timestep_opt_s)
        four_hour_index = (i // (14400 // timestep_opt_s)) * 6

        # Charge constraint
        A_ub[2 * i, num_4hour_periods + hour_index] = 1
        A_ub[2 * i, four_hour_index] = DC_high_response[i]
        A_ub[2 * i, four_hour_index + 2] = DM_high_response[i]
        A_ub[2 * i, four_hour_index + 4] = DR_high_response[i]

        # Discharge constraint
        A_ub[2 * i + 1, num_4hour_periods + num_baseline_charge_discharge + hour_index] = 1
        A_ub[2 * i + 1, four_hour_index + 1] = DC_low_response[i]
        A_ub[2 * i + 1, four_hour_index + 3] = DM_low_response[i]
        A_ub[2 * i + 1, four_hour_index + 5] = DR_low_response[i]

        b_ub[2 * i] = max_power_MW
        b_ub[2 * i + 1] = max_power_MW

    # SOC constraints
    A_eq = lil_matrix((num_timesteps + 1, n_vars))
    b_eq = np.zeros(num_timesteps + 1)

    # Initial SOC
    A_eq[0, num_4hour_periods + num_baseline_charge_discharge*2] = 1
    b_eq[0] = 0.5

    eff_div_capacity_MWh = efficiency / (3600 / timestep_opt_s * capacity_MWh)
    inv_eff_div_capacity_MWh = 1 / (efficiency * 3600 / timestep_opt_s * capacity_MWh)

    # SOC evolution constraints
    for i in range(1, num_timesteps + 1):
        hour_index = (i - 1) // (1800 // timestep_opt_s)
        four_hour_index = ((i - 1) // (14400 // timestep_opt_s)) * 6

        # SOC change due to charging and discharging
        A_eq[i, num_4hour_periods + 2*num_baseline_charge_discharge + i] = 1  # SOC(i)
        A_eq[i, num_4hour_periods + 2*num_baseline_charge_discharge + i - 1] = -1  # SOC(i-1)

        # SOC change due to charging (efficiency adjusted)
        A_eq[i, num_4hour_periods + hour_index] = -eff_div_capacity_MWh
        A_eq[i, four_hour_index] = -DC_high_response[i - 1] * eff_div_capacity_MWh
        A_eq[i, four_hour_index + 2] = -DM_high_response[i - 1] * eff_div_capacity_MWh
        A_eq[i, four_hour_index + 4] = -DR_high_response[i - 1] * eff_div_capacity_MWh

        # SOC change due to discharging (efficiency adjusted)
        A_eq[i, num_4hour_periods + num_baseline_charge_discharge + hour_index] = inv_eff_div_capacity_MWh
        A_eq[i, four_hour_index + 1] = DC_low_response[i - 1] * inv_eff_div_capacity_MWh
        A_eq[i, four_hour_index + 3] = DM_low_response[i - 1] * inv_eff_div_capacity_MWh
        A_eq[i, four_hour_index + 5] = DR_low_response[i - 1] * inv_eff_div_capacity_MWh


    bounds = [(0, max_power_MW)] * num_4hour_periods + [(0, max_power_MW)] * (num_baseline_charge_discharge * 2) + [(0, 1)] * (num_timesteps + 1)
    result = linprog(c, A_ub=A_ub.tocsr(), b_ub=b_ub, A_eq=A_eq.tocsr(), b_eq=b_eq, bounds=bounds, method='highs')

    if result.success:
        daily_profit_GBP = -result.fun

        optimal_capacity_allocation_MW = result.x[:num_4hour_periods].tolist()
        optimal_baseline_charge_MW = result.x[num_4hour_periods:num_4hour_periods + num_baseline_charge_discharge].tolist()
        optimal_baseline_discharge_MW = result.x[num_4hour_periods + num_baseline_charge_discharge:num_4hour_periods + num_baseline_charge_discharge * 2].tolist()
        soc_results = result.x[num_4hour_periods + num_baseline_charge_discharge * 2:].tolist()

        cycles = 0
        for i in range(1, len(soc_results)):
            cycles += abs(soc_results[i] - soc_results[i-1])

        return daily_profit_GBP, optimal_capacity_allocation_MW, optimal_baseline_charge_MW, optimal_baseline_discharge_MW, soc_results, cycles/2

    else:
        raise ValueError("Optimization failed")


def optimizer_no_aging_linprog_legal_limits(hours, timestep_opt_s, epex_price_GBP_MWh, 
                       clearing_prices, efficiency, max_power_MW, capacity_MWh, starting_soc, 
                       DC_high_response, DC_low_response, DM_high_response,
                       DM_low_response, DR_high_response, DR_low_response):

    num_timesteps = int(3600 / timestep_opt_s * hours)
    num_4hour_periods = (hours // 4) * 6
    num_baseline_charge_discharge = hours * 2


    # Decision variables: capacity_allocation, baseline_charge, baseline_discharge, SOC
    n_vars = num_4hour_periods + num_baseline_charge_discharge * 2 + num_timesteps + 1

    # Objective function
    c = np.zeros(n_vars)
    for i in range(num_baseline_charge_discharge):
        c[num_4hour_periods + i] = epex_price_GBP_MWh[i] / 2  # Charging
        c[num_4hour_periods + num_baseline_charge_discharge + i] = -epex_price_GBP_MWh[i] / 2  # Discharging
    for i in range(num_4hour_periods):
        c[i] = -clearing_prices[i] * 4

    # Inequality constraints (A_ub * x <= b_ub)
    A_ub = lil_matrix((4*hours + 2*hours*2, n_vars))
    b_ub = np.zeros(4*hours + 2*hours*2)

    # Max power constraints for charge and discharge
    for i in range(2*hours):

        # Charge constraint
        A_ub[2 * i, num_4hour_periods + i] = 1
        A_ub[2 * i, (i//8)*6] = 1
        A_ub[2 * i, (i//8)*6 + 2] = 1
        A_ub[2 * i, (i//8)*6 + 4] = 1

        # Discharge constraint
        A_ub[2 * i + 1, num_4hour_periods + num_baseline_charge_discharge + i] = 1
        A_ub[2 * i + 1, (i//8)*6 + 1] = 1
        A_ub[2 * i + 1, (i//8)*6 + 3] = 1
        A_ub[2 * i + 1, (i//8)*6 + 5] = 1

        b_ub[2 * i] = max_power_MW
        b_ub[2 * i + 1] = max_power_MW
    
    #Legal limit constraints on SOC at the beginning of settlement period
    for i in range(hours*2):

        #SOC_t <= 1-(0.25*DC_high+0.5*DM_high+DR_high)/capacity
        b_ub[4*hours + i] = 1
        A_ub[4*hours + i, (i//8)*6] = 0.25/capacity_MWh*efficiency
        A_ub[4*hours + i, (i//8)*6+2] = 0.5/capacity_MWh*efficiency
        A_ub[4*hours + i, (i//8)*6+4] = 1/capacity_MWh*efficiency
        A_ub[4*hours + i, num_4hour_periods + num_baseline_charge_discharge * 2+i*1800//timestep_opt_s] = 1

        #SOC_t >= (0.25*DC_low+0.5*DM_low+DR_low)/capacity
        b_ub[4*hours + hours*2 + i] = 0
        A_ub[4*hours + hours*2 + i, (i//8)*6+1] = 0.25/capacity_MWh/efficiency
        A_ub[4*hours + hours*2 + i, (i//8)*6+3] = 0.5/capacity_MWh/efficiency
        A_ub[4*hours + hours*2 + i, (i//8)*6+5] = 1/capacity_MWh/efficiency
        A_ub[4*hours + hours*2 + i, num_4hour_periods + num_baseline_charge_discharge * 2+i*1800//timestep_opt_s] = -1

    # SOC constraints
    A_eq = lil_matrix((num_timesteps + 1, n_vars))
    b_eq = np.zeros(num_timesteps + 1)

    # Initial SOC
    A_eq[0, num_4hour_periods + num_baseline_charge_discharge*2] = 1
    b_eq[0] = starting_soc

    eff_div_capacity_MWh = efficiency / (3600 / timestep_opt_s * capacity_MWh)
    inv_eff_div_capacity_MWh = 1 / (efficiency * 3600 / timestep_opt_s * capacity_MWh)

    # SOC evolution constraints
    for i in range(1, num_timesteps + 1):
        hour_index = (i - 1) // (1800 // timestep_opt_s)
        four_hour_index = ((i - 1) // (14400 // timestep_opt_s)) * 6

        # SOC change due to charging and discharging
        A_eq[i, num_4hour_periods + 2*num_baseline_charge_discharge + i] = 1  # SOC(i)
        A_eq[i, num_4hour_periods + 2*num_baseline_charge_discharge + i - 1] = -1  # SOC(i-1)

        # SOC change due to charging (efficiency adjusted)
        A_eq[i, num_4hour_periods + hour_index] = -eff_div_capacity_MWh
        A_eq[i, four_hour_index] = -DC_high_response[i - 1] * eff_div_capacity_MWh
        A_eq[i, four_hour_index + 2] = -DM_high_response[i - 1] * eff_div_capacity_MWh
        A_eq[i, four_hour_index + 4] = -DR_high_response[i - 1] * eff_div_capacity_MWh

        # SOC change due to discharging (efficiency adjusted)
        A_eq[i, num_4hour_periods + num_baseline_charge_discharge + hour_index] = inv_eff_div_capacity_MWh
        A_eq[i, four_hour_index + 1] = DC_low_response[i - 1] * inv_eff_div_capacity_MWh
        A_eq[i, four_hour_index + 3] = DM_low_response[i - 1] * inv_eff_div_capacity_MWh
        A_eq[i, four_hour_index + 5] = DR_low_response[i - 1] * inv_eff_div_capacity_MWh


    bounds = [(0, max_power_MW)] * num_4hour_periods + [(0, max_power_MW)] * (num_baseline_charge_discharge * 2) + [(0, 1)] * (num_timesteps + 1)
    result = linprog(c, A_ub=A_ub.tocsr(), b_ub=b_ub, A_eq=A_eq.tocsr(), b_eq=b_eq, bounds=bounds, method='highs')

    if result.success:
        daily_profit_GBP = -result.fun

        optimal_capacity_allocation_MW = result.x[:num_4hour_periods].tolist()
        optimal_baseline_charge_MW = result.x[num_4hour_periods:num_4hour_periods + num_baseline_charge_discharge].tolist()
        optimal_baseline_discharge_MW = result.x[num_4hour_periods + num_baseline_charge_discharge:num_4hour_periods + num_baseline_charge_discharge * 2].tolist()
        soc_results = result.x[num_4hour_periods + num_baseline_charge_discharge * 2:].tolist()

        cycles = 0
        for i in range(1, len(soc_results)):
            cycles += abs(soc_results[i] - soc_results[i-1])

        return daily_profit_GBP, optimal_capacity_allocation_MW, optimal_baseline_charge_MW, optimal_baseline_discharge_MW, soc_results, cycles/2

    else:
        raise ValueError("Optimization failed")
    

def optimizer_no_aging_max_min(hours, timestep_opt_s, epex_price_GBP_MWh, 
                       clearing_prices, efficiency, max_power_MW, capacity_MWh, starting_soc, 
                       DC_high_response, DC_low_response, DM_high_response,
                       DM_low_response, DR_high_response, DR_low_response,
                       DC_high_response_max, DC_low_response_max, DM_high_response_max,
                       DM_low_response_max, DR_high_response_max, DR_low_response_max):
    
    
    #initialize the gurobi model
    model = gp.Model('FFR_Maximize_Profit')
    model.setParam('OutputFlag',1)

    #set mip gap
    model.setParam('MIPGap', 0.1)

    #add the decision variables
    capacity_allocation = model.addVars(hours//4*6, lb=0, ub=max_power_MW, name="capacity_allocation")
    baseline_charge = model.addVars(hours*2, lb=0, ub=max_power_MW, name="baseline_charge")
    baseline_discharge = model.addVars(hours*2, lb=0, ub=max_power_MW, name="baseline_discharge")

    #add auxiliary variables
    soc = model.addVars(int(3600/timestep_opt_s*hours+1),lb = 0, ub = 1, name="soc")
    power = model.addVars(int(3600/timestep_opt_s*hours),lb = -max_power_MW, ub = max_power_MW, name = 'power')
    charge_status = model.addVars(int(3600/timestep_opt_s*hours), vtype = GRB.BINARY)

    #add the objective function
    #calculate the value of the energy bought or sold through baselines --> division by two is to convert half hourly
    #settlement periods int o MWh
    value_recharge_GBP = (-1*gp.quicksum(baseline_charge[i]*epex_price_GBP_MWh[i] for i in range(len(baseline_charge)))
                        + gp.quicksum(baseline_discharge[i]*epex_price_GBP_MWh[i] for i in range(len(baseline_discharge))))/2

    #calculate the value of the FFR services provided --> multiplication by 4 is to convert hourly clearing prices to 4-hour
    #EAC window
    value_FFR_GBP = gp.quicksum(clearing_prices[i]*capacity_allocation[i] for i in range(len(capacity_allocation)))*4

    #calculate the value of the energy won/lost throughout the day
    value_energy_diff_GBP = (soc[len(soc)-1] - soc[0])*capacity_MWh * np.mean(epex_price_GBP_MWh)

    model.setObjective(value_recharge_GBP + value_FFR_GBP + value_energy_diff_GBP, GRB.MAXIMIZE)

    #--------------------CONSTRAINTS--------------------
    #manage SOC evolution

    # Add power evolution constraint
    model.addConstrs(power[i] == (baseline_charge[(i)//(1800/timestep_opt_s)] +
        DC_high_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6] +
        DM_high_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6+2] +
        DR_high_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6+4]) -
        (baseline_discharge[(i)//(1800/timestep_opt_s)] +
        DC_low_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6+1] +
        DM_low_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6+3] +
        DR_low_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6+5])
        for i in range(0, int(3600/timestep_opt_s*hours)))
    
    #add charging status constraint
    # Add constraints in a loop
    M = 1e6
    epsilon = 1e-6
    for i in range(int(3600 / timestep_opt_s * hours)):
        model.addConstr(power[i] >= epsilon - M * (1 - charge_status[i]), name=f"charge_status_1_{i}")
        model.addConstr(power[i] <= M * charge_status[i], name=f"charge_status_2_{i}")

    # Add SOC constraints in bulk
    model.addConstr(soc[0] == starting_soc)

    model.addConstrs(
        (soc[i] == soc[i-1] + power[i-1]/capacity_MWh/3600*timestep_opt_s * (1.05-0.1*charge_status[i-1]))
        for i in range(1,len(soc))
    )

    # Add maximum power constraints for charge in bulk
    model.addConstrs(
        (baseline_charge[(i-1)//(1800/timestep_opt_s)] +
        capacity_allocation[((i-1)//(14400/timestep_opt_s))*6] +
        capacity_allocation[((i-1)//(14400/timestep_opt_s))*6+2] +
        capacity_allocation[((i-1)//(14400/timestep_opt_s))*6+4] <= max_power_MW)
        for i in range(1, int(3600/timestep_opt_s*hours+1))
    )

    # Add maximum power constraints for discharge in bulk
    model.addConstrs(
        (baseline_discharge[(i-1)//(1800/timestep_opt_s)] +
        capacity_allocation[((i-1)//(14400/timestep_opt_s))*6+1] +
        capacity_allocation[((i-1)//(14400/timestep_opt_s))*6+3] +
        capacity_allocation[((i-1)//(14400/timestep_opt_s))*6+5] <= max_power_MW)
        for i in range(1, int(3600/timestep_opt_s*hours+1))
    )

    #add soc constraint to be within 0 and 1
    model.addConstrs((soc[i] <= 1 for i in range(len(soc))))
    model.addConstrs((soc[i] >= 0 for i in range(len(soc))))

    #add soc constraint to comply with legal limits
    model.addConstrs((soc[i*1800/timestep_opt_s] <= 1-(0.25*capacity_allocation[i//8]+0.5*capacity_allocation[i//8+2]+
                    capacity_allocation[i//8+4])/capacity_MWh for i in range(hours*2)))
    model.addConstrs((soc[i*1800/timestep_opt_s] >= (0.25*capacity_allocation[i//8+1]+0.5*capacity_allocation[i//8+3]+
                    capacity_allocation[i//8+5])/capacity_MWh for i in range(hours*2)))

    model.optimize()

    model.optimize()

    #retreive the optimal solution
    daily_profit_GBP = model.objVal

    #retreive the optimal capacity allocation
    optimal_capacity_allocation_MW = []
    for i in range(36):
        optimal_capacity_allocation_MW.append(capacity_allocation[i].x)

    #retreive the optimal baseline charge
    optimal_baseline_charge_MW = []
    for i in range(48):
        optimal_baseline_charge_MW.append(baseline_charge[i].x)
    
    #retreive the optimal baseline discharge
    optimal_baseline_discharge_MW = []
    for i in range(48):
        optimal_baseline_discharge_MW.append(baseline_discharge[i].x)
    
    power_output = []
    for i in range(len(charge_status)):
        power_output.append(power[i].x)
    
    #retreive the optimal SOC as well as cycle count
    cycles = 0
    soc_results = []
    for i in range(len(soc)//2):
        soc_results.append(soc[i].x)
        if i > 0:
            cycles += abs(soc_results[-1] - soc_results[-2])
    
    return daily_profit_GBP, optimal_capacity_allocation_MW, optimal_baseline_charge_MW, optimal_baseline_discharge_MW, soc_results, cycles/2


def optimizer_cyc_aging(hours, timestep_opt_s, epex_price_GBP_MWh, 
                       clearing_prices, efficiency, max_power_MW, capacity_MWh, starting_soc, 
                       DC_high_response, DC_low_response, DM_high_response,
                       DM_low_response, DR_high_response, DR_low_response,
                       aging_cost_GBP_per_cycle):
    
    
    print('Correct function used.')

    #initialize the gurobi model
    model = gp.Model('FFR_Maximize_Profit')
    model.setParam('OutputFlag',1)

    #set mip gap
    model.setParam('MIPGap', 0.01)
    model.setParam('TimeLimit', 300)

    #add the decision variables
    capacity_allocation = model.addVars(hours//4*6, lb=0, ub=max_power_MW, name="capacity_allocation")
    baseline_charge = model.addVars(hours*2, lb=0, ub=max_power_MW, name="baseline_charge")
    baseline_discharge = model.addVars(hours*2, lb=0, ub=max_power_MW, name="baseline_discharge")

    #add auxiliary variables
    soc = model.addVars(int(3600/timestep_opt_s*hours+1),lb = 0, ub = 1, name="soc")
    power_tot = model.addVars(int(3600/timestep_opt_s*hours),lb = -max_power_MW, ub = max_power_MW, name = 'power')
    power_charge = model.addVars(int(3600/timestep_opt_s*hours),lb = 0, ub = max_power_MW, name = 'power')
    power_discharge = model.addVars(int(3600/timestep_opt_s*hours),lb = 0, ub = max_power_MW, name = 'power')
    charge_status = model.addVars(int(3600/timestep_opt_s*hours), vtype = GRB.BINARY)
    cycles_opt = model.addVar(lb = 0, ub = 10,name = 'cycles')
    charge_status = model.addVars(int(3600/timestep_opt_s*hours), vtype = GRB.BINARY)

    #add the objective function
    #calculate the value of the energy bought or sold through baselines --> division by two is to convert half hourly
    #settlement periods int o MWh
    value_recharge_GBP = (-1*gp.quicksum(baseline_charge[i]*epex_price_GBP_MWh[i] for i in range(len(baseline_charge)))
                        + gp.quicksum(baseline_discharge[i]*epex_price_GBP_MWh[i] for i in range(len(baseline_discharge))))/2

    #calculate the value of the FFR services provided --> multiplication by 4 is to convert hourly clearing prices to 4-hour
    #EAC window
    value_FFR_GBP = gp.quicksum(clearing_prices[i]*capacity_allocation[i] for i in range(len(capacity_allocation)))*4

    #calculate the value of the energy won/lost throughout the day
    #value_energy_diff_GBP = (soc[len(soc)-1] - soc[0])*capacity_MWh * np.mean(epex_price_GBP_MWh)

    #calculate the aging cost
    value_aging_GBP = cycles_opt*aging_cost_GBP_per_cycle

    model.setObjective(value_recharge_GBP + value_FFR_GBP -value_aging_GBP, GRB.MAXIMIZE)

    #--------------------CONSTRAINTS--------------------
    #manage SOC evolution

    # Add power evolution constraint
    model.addConstrs(power_tot[i] == (baseline_charge[(i)//(1800/timestep_opt_s)] +
        DC_high_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6] +
        DM_high_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6+2] +
        DR_high_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6+4]) -
        (baseline_discharge[(i)//(1800/timestep_opt_s)] +
        DC_low_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6+1] +
        DM_low_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6+3] +
        DR_low_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6+5])
        for i in range(0, int(3600/timestep_opt_s*hours)))
    
    #add charging status constraint
    # Add constraints in a loop

    M = 1e6
    epsilon = 1e-6
    for i in range(int(3600 / timestep_opt_s * hours)):
        model.addConstr(power_tot[i] >= epsilon - M * (1 - charge_status[i]), name=f"charge_status_1_{i}")
        model.addConstr(power_tot[i] <= M * charge_status[i], name=f"charge_status_2_{i}")
    
    #add constraints for power charge and discharge
    #total power
    model.addConstrs(power_tot[i] == power_charge[i] - power_discharge[i] for i in range(len(power_tot)))

    #add constraint to force one non-zero value in charge and discharge
    model.addConstrs(power_charge[i] <= max_power_MW*charge_status[i] for i in range(len(power_charge)))
    model.addConstrs(power_discharge[i] <= max_power_MW*(1-charge_status[i]) for i in range(len(power_discharge)))

    # Add SOC constraints in bulk
    model.addConstr(soc[0] == starting_soc)

    model.addConstrs(
        (soc[i] == soc[i-1] + efficiency*power_charge[i-1]/capacity_MWh/3600*timestep_opt_s - power_discharge[i-1]/capacity_MWh/3600*timestep_opt_s/efficiency)
        for i in range(1,len(soc))
    )

    # Add cycle variable constraint
    model.addConstr(cycles_opt == gp.quicksum((-1+2*charge_status[i])*power_tot[i]/2/(capacity_MWh/0.9)/3600*timestep_opt_s for i in range(len(power_tot))))

    # Add maximum power constraints for charge in bulk
    model.addConstrs(
        (baseline_charge[(i-1)//(1800/timestep_opt_s)] +
        capacity_allocation[((i-1)//(14400/timestep_opt_s))*6] +
        capacity_allocation[((i-1)//(14400/timestep_opt_s))*6+2] +
        capacity_allocation[((i-1)//(14400/timestep_opt_s))*6+4] <= max_power_MW)
        for i in range(1, int(3600/timestep_opt_s*hours+1))
    )

    # Add maximum power constraints for discharge in bulk
    model.addConstrs(
        (baseline_discharge[(i-1)//(1800/timestep_opt_s)] +
        capacity_allocation[((i-1)//(14400/timestep_opt_s))*6+1] +
        capacity_allocation[((i-1)//(14400/timestep_opt_s))*6+3] +
        capacity_allocation[((i-1)//(14400/timestep_opt_s))*6+5] <= max_power_MW)
        for i in range(1, int(3600/timestep_opt_s*hours+1))
    )

    #add soc constraint to be within 0 and 1
    model.addConstrs((soc[i] <= 1 for i in range(len(soc))))
    model.addConstrs((soc[i] >= 0 for i in range(len(soc))))

    #add soc constraint to comply with legal limits
    model.addConstrs((soc[i*1800/timestep_opt_s] <= 1-(0.25*capacity_allocation[i//8]+0.5*capacity_allocation[i//8+2]+
                    capacity_allocation[i//8+4])/capacity_MWh for i in range(hours*2)))
    model.addConstrs((soc[i*1800/timestep_opt_s] >= (0.25*capacity_allocation[i//8+1]+0.5*capacity_allocation[i//8+3]+
                    capacity_allocation[i//8+5])/capacity_MWh for i in range(hours*2)))

    model.optimize()

    #retreive the optimal solution
    daily_profit_GBP = model.objVal

    #retreive the optimal capacity allocation
    optimal_capacity_allocation_MW = []
    for i in range(36):
        optimal_capacity_allocation_MW.append(capacity_allocation[i].x)

    #retreive the optimal baseline charge
    optimal_baseline_charge_MW = []
    for i in range(48):
        optimal_baseline_charge_MW.append(baseline_charge[i].x)
    
    #retreive the optimal baseline discharge
    optimal_baseline_discharge_MW = []
    for i in range(48):
        optimal_baseline_discharge_MW.append(baseline_discharge[i].x)
    
    power_output_charge = []
    for i in range(len(charge_status)):
        power_output_charge.append(power_charge[i].x)
    
    power_output_discharge = []
    for i in range(len(charge_status)):
        power_output_discharge.append(power_discharge[i].x)
    
    power_output_total = []
    for i in range(len(charge_status)):
        power_output_total.append(power_tot[i].x)

    power_status = []
    for i in range(len(charge_status)):
        power_status.append(charge_status[i].x)
    
    #retreive the optimal SOC as well as cycle count
    cycles = 0
    soc_results = []
    for i in range(len(soc)):
        soc_results.append(soc[i].x)
        if i > 0:
            cycles += abs(soc_results[-1] - soc_results[-2])
    
    return daily_profit_GBP, optimal_capacity_allocation_MW, optimal_baseline_charge_MW, optimal_baseline_discharge_MW, soc_results, cycles/2, cycles_opt.x

import sys

def optimizer_pl_aging(hours, timestep_opt_s, epex_price_GBP_MWh, 
                       clearing_prices, efficiency, max_power_MW, capacity_MWh, starting_soc, 
                       DC_high_response, DC_low_response, DM_high_response,
                       DM_low_response, DR_high_response, DR_low_response,
                       aging_cost_GBP_per_cycle, SOH):
    
    #read in the piecewise linear degradation coefficients
    sys.path.insert(0, '/Users/freddi/Documents/Frederik/MSc Energy Systems/Thesis/Bidding Strategy/Functions')
    from lfp_sony_coefficients_lin_degradation import SonyLFPCoefficientsLinearDegradation
    linear_degradation: SonyLFPCoefficientsLinearDegradation = SonyLFPCoefficientsLinearDegradation()

    coefficients_lin_deg_cal = linear_degradation.get_cal_coefficients(soh=0.95)
    c_lin_x_cal = coefficients_lin_deg_cal[:, 0]
    c_lin_y_cal = coefficients_lin_deg_cal[:, 1]*1e6
    set_I_cal = range(0, int(c_lin_y_cal.shape[0]))

    coefficients_lin_deg_cyc = linear_degradation.get_cyc_coefficients(soh=0.95)
    c_lin_x_cyc = coefficients_lin_deg_cyc[:, 0] * capacity_MWh
    c_lin_y_cyc = coefficients_lin_deg_cyc[:, 1] * 1e6
    set_J_cyc = range(0, int(c_lin_y_cyc.shape[0]))



    #initialize the gurobi model
    model = gp.Model('FFR_Maximize_Profit')
    model.setParam('OutputFlag',1)

    #set mip gap
    model.setParam('MIPGap', 0.1)

    #add the decision variables
    capacity_allocation = model.addVars(hours//4*6, lb=0, ub=max_power_MW, name="capacity_allocation")
    baseline_charge = model.addVars(hours*2, lb=0, ub=max_power_MW, name="baseline_charge")
    baseline_discharge = model.addVars(hours*2, lb=0, ub=max_power_MW, name="baseline_discharge")

    #add auxiliary variables
    soc = model.addVars(int(3600/timestep_opt_s*hours+1),lb = 0, ub = 1, name="soc")
    power = model.addVars(int(3600/timestep_opt_s*hours),lb = -max_power_MW, ub = max_power_MW, name = 'power')
    charge_status = model.addVars(int(3600/timestep_opt_s*hours), vtype = GRB.BINARY)
    cycles_opt = model.addVar(lb = 0, ub = 10,name = 'cycles')

    #add the variables for handling of the aging
    q_loss_cal = model.addVars(hours*4, lb = 0)
    q_loss_cyc = model.addVars(hours//4,lb = 0)
    

    lambda_cal = {}
    lambda_cyc = {}

    for i in range(len(q_loss_cal)):
        lambda_cal[i] = model.addVars(set_I_cal,lb = 0, ub = 1, name = f'lambda_cal_{i}')
    
    for i in range(len(q_loss_cyc)):
        lambda_cyc[i] = model.addVars(set_J_cyc,lb = 0, ub = 1, name = f'lambda_cyc_{i}')
    



    #add the objective function
    #calculate the value of the energy bought or sold through baselines --> division by two is to convert half hourly
    #settlement periods int o MWh
    value_recharge_GBP = (-1*gp.quicksum(baseline_charge[i]*epex_price_GBP_MWh[i] for i in range(len(baseline_charge)))
                        + gp.quicksum(baseline_discharge[i]*epex_price_GBP_MWh[i] for i in range(len(baseline_discharge))))/2

    #calculate the value of the FFR services provided --> multiplication by 4 is to convert hourly clearing prices to 4-hour
    #EAC window
    value_FFR_GBP = gp.quicksum(clearing_prices[i]*capacity_allocation[i] for i in range(len(capacity_allocation)))*4

    #calculate the value of the energy won/lost throughout the day
    #value_energy_diff_GBP = (soc[len(soc)-1] - soc[0])*capacity_MWh * np.mean(epex_price_GBP_MWh)

    #calculate the aging cost
    value_aging_GBP = 400000/0.2*(gp.quicksum(q_loss_cal)+gp.quicksum(q_loss_cyc))/1e6

    model.setObjective(value_recharge_GBP + value_FFR_GBP-value_aging_GBP, GRB.MAXIMIZE)

    #--------------------CONSTRAINTS--------------------
    #manage SOC evolution

    # Add power evolution constraint
    model.addConstrs(power[i] == (baseline_charge[(i)//(1800/timestep_opt_s)] +
        DC_high_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6] +
        DM_high_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6+2] +
        DR_high_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6+4]) -
        (baseline_discharge[(i)//(1800/timestep_opt_s)] +
        DC_low_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6+1] +
        DM_low_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6+3] +
        DR_low_response[i] * capacity_allocation[((i)//(14400/timestep_opt_s))*6+5])
        for i in range(0, int(3600/timestep_opt_s*hours)))
    
    #add charging status constraint
    # Add constraints in a loop
    M = 1e6
    epsilon = 1e-6
    for i in range(int(3600 / timestep_opt_s * hours)):
        model.addConstr(power[i] >= epsilon - M * (1 - charge_status[i]), name=f"charge_status_1_{i}")
        model.addConstr(power[i] <= M * charge_status[i], name=f"charge_status_2_{i}")

    # Add SOC constraints in bulk
    model.addConstr(soc[0] == starting_soc)

    model.addConstrs(
        (soc[i] == soc[i-1] + power[i-1]/capacity_MWh/3600*timestep_opt_s * (1.05-0.1*charge_status[i-1]))
        for i in range(1,len(soc))
    )

    # Add cycle variable constraint
    model.addConstr(cycles_opt == gp.quicksum((-1+2*charge_status[i])*power[i]/2/(capacity_MWh/0.9)/3600*timestep_opt_s for i in range(len(power))))

    # Add maximum power constraints for charge in bulk
    model.addConstrs(
        (baseline_charge[(i-1)//(1800/timestep_opt_s)] +
        capacity_allocation[((i-1)//(14400/timestep_opt_s))*6] +
        capacity_allocation[((i-1)//(14400/timestep_opt_s))*6+2] +
        capacity_allocation[((i-1)//(14400/timestep_opt_s))*6+4] <= max_power_MW)
        for i in range(1, int(3600/timestep_opt_s*hours+1))
    )

    # Add maximum power constraints for discharge in bulk
    model.addConstrs(
        (baseline_discharge[(i-1)//(1800/timestep_opt_s)] +
        capacity_allocation[((i-1)//(14400/timestep_opt_s))*6+1] +
        capacity_allocation[((i-1)//(14400/timestep_opt_s))*6+3] +
        capacity_allocation[((i-1)//(14400/timestep_opt_s))*6+5] <= max_power_MW)
        for i in range(1, int(3600/timestep_opt_s*hours+1))
    )

    #add soc constraint to be within 0 and 1
    model.addConstrs((soc[i] <= 1 for i in range(len(soc))))
    model.addConstrs((soc[i] >= 0 for i in range(len(soc))))

    #add soc constraint to comply with legal limits
    model.addConstrs((soc[i*1800/timestep_opt_s] <= 1-(0.25*capacity_allocation[i//8]+0.5*capacity_allocation[i//8+2]+
                    capacity_allocation[i//8+4])/capacity_MWh for i in range(hours*2)))
    model.addConstrs((soc[i*1800/timestep_opt_s] >= (0.25*capacity_allocation[i//8+1]+0.5*capacity_allocation[i//8+3]+
                    capacity_allocation[i//8+5])/capacity_MWh for i in range(hours*2)))

    #add the constraints for the calendar aging
    for i in range(len(q_loss_cal)):
        #creates a sos type 2 array for linear interpolation between within the piecewise linearized function
        model.addSOS(GRB.SOS_TYPE2, vars = lambda_cal[i])
        #makes sure that the interpolation sums to 1
        model.addConstr(gp.quicksum(lambda_cal[i][j] for j in set_I_cal) == 1)
        #sets the lambda factors to correct weights for the two points
        model.addConstr(gp.quicksum(lambda_cal[i][j]*c_lin_x_cal[j] for j in set_I_cal) == soc[i*900//timestep_opt_s])
        #finds the corresponding q_loss point
        model.addConstr(gp.quicksum(lambda_cal[i][j]*c_lin_y_cal[j] for j in set_I_cal) == q_loss_cal[i])

    #add the constraints for the cyclic aging
    for i in range(len(q_loss_cyc)):
        #creates a sos type 2 array for linear interpolation between within the piecewise linearized function
        model.addSOS(GRB.SOS_TYPE2, vars = lambda_cyc[i])
        #makes sure that the interpolation sums to 1
        model.addConstr(gp.quicksum(lambda_cyc[i][j] for j in set_I_cal) == 1)
        #sets the lambda factors to correct weights for the two points
        model.addConstr(gp.quicksum(lambda_cyc[i][j]*c_lin_x_cyc[j] for j in set_I_cal) == 
                        gp.quicksum((-1+2*charge_status[t])*power[i]/0.9/3600*timestep_opt_s for t in range(int(i*14400/timestep_opt_s),int(i*14400//timestep_opt_s+14400//timestep_opt_s))))
        #finds the corresponding q_loss point
        model.addConstr(gp.quicksum(lambda_cal[i][j]*c_lin_y_cyc[j] for j in set_I_cal) == q_loss_cyc[i])

    model.optimize()

    #retreive the optimal solution
    daily_profit_GBP = model.objVal

    #retreive the optimal capacity allocation
    optimal_capacity_allocation_MW = []
    for i in range(36):
        optimal_capacity_allocation_MW.append(capacity_allocation[i].x)

    #retreive the optimal baseline charge
    optimal_baseline_charge_MW = []
    for i in range(48):
        optimal_baseline_charge_MW.append(baseline_charge[i].x)
    
    #retreive the optimal baseline discharge
    optimal_baseline_discharge_MW = []
    for i in range(48):
        optimal_baseline_discharge_MW.append(baseline_discharge[i].x)
    
    power_output = []
    for i in range(len(charge_status)):
        power_output.append(power[i].x)
    
    #retreive the optimal SOC as well as cycle count
    cycles = 0
    soc_results = []
    for i in range(len(soc)):
        soc_results.append(soc[i].x)
        if i > 0:
            cycles += abs(soc_results[-1] - soc_results[-2])
    
    #retreive the aging
    aging = 0

    for i in range(len(q_loss_cal)):
        aging += q_loss_cal[i].x/1e6
    
    for i in range(len(q_loss_cyc)):
        for j in range(len(c_lin_x_cyc)):
           print(lambda_cyc[i][j].x) 
        aging += q_loss_cyc[i].x/1e6
    
    return daily_profit_GBP, optimal_capacity_allocation_MW, optimal_baseline_charge_MW, optimal_baseline_discharge_MW, soc_results, cycles/2, aging
