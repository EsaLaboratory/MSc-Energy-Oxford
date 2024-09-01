import gurobipy as gp
from gurobipy import GRB

def optimizer_no_limits(hours, timestep_opt_s, epex_price_GBP_MWh, 
                       clearing_prices, efficiency, max_power_MW, capacity_MWh, starting_soc, 
                       DC_high_response, DC_low_response, DM_high_response,
                       DM_low_response, DR_high_response, DR_low_response):
    
    #initialize the gurobi model
    model = gp.Model('FFR_Maximize_Profit')
    model.setParam('OutputFlag',1)

    #set mip gap
    model.setParam('MIPGap', 0.01)
    model.setParam('TimeLimit', 300)

    ## add the decision variables
    #capacity allocation is a set of 6 variables for each 4-hour period to allocate a capacity in MW to
    #each of the 6 FFR products
    capacity_allocation = model.addVars(hours//4*6, lb=0, ub=max_power_MW, name="capacity_allocation")

    #baseline charge denotes the amount of power that is bought from the grid to charge the battery
    baseline_charge = model.addVars(hours*2, lb=0, ub=max_power_MW, name="baseline_charge")

    #baseline discharge denotes the amount of power that is sold to the grid to discharge the battery
    baseline_discharge = model.addVars(hours*2, lb=0, ub=max_power_MW, name="baseline_discharge")

    ## add auxiliary variables
    #the soc stores the state of charge of the battery at each time step
    soc = model.addVars(int(3600/timestep_opt_s*hours+1),lb = 0, ub = 1, name="soc")
    
    #total power is a helper variable that makes the constraints more readable and is the sum of all power
    #draws
    power_tot = model.addVars(int(3600/timestep_opt_s*hours),lb = -max_power_MW, ub = max_power_MW, name = 'power')
    
    #power charge and power discharge are the total powers in either direction that are used for soc calculation
    power_charge = model.addVars(int(3600/timestep_opt_s*hours),lb = 0, ub = max_power_MW, name = 'power')
    power_discharge = model.addVars(int(3600/timestep_opt_s*hours),lb = 0, ub = max_power_MW, name = 'power')
    
    #cycles_opt denotes the variable that stores the total number of cycles that the battery has gone through
    cycles_opt = model.addVar(lb = 0, ub = 10,name = 'cycles')

    #charge status is a binary variable that denotes whether the battery is charging (1) or discharging (0)
    charge_status = model.addVars(int(3600/timestep_opt_s*hours), vtype = GRB.BINARY)

    #the binary variables baseline_charge_status and baseline_discharge_status denote whether the baseline
    #charge or discharge is non-zero - only one of the two can be non-zero
    baseline_charge_status = model.addVars(hours*2, vtype = GRB.BINARY)
    baseline_discharge_status = model.addVars(hours*2, vtype = GRB.BINARY)

    #add the objective function
    #calculate the value of the energy bought or sold through baselines --> division by two is to convert half hourly
    #settlement periods
    value_recharge_GBP = (-1*gp.quicksum(baseline_charge[i]*epex_price_GBP_MWh[i] for i in range(len(baseline_charge)))
                        + gp.quicksum(baseline_discharge[i]*epex_price_GBP_MWh[i] for i in range(len(baseline_discharge))))/2

    #calculate the value of the FFR services provided --> multiplication by 4 is to convert hourly clearing prices to 4-hour
    #EAC window
    value_FFR_GBP = gp.quicksum(clearing_prices[i]*capacity_allocation[i] for i in range(len(capacity_allocation)))*4

    model.setObjective(value_recharge_GBP + value_FFR_GBP, GRB.MAXIMIZE)

    #--------------------CONSTRAINTS--------------------
    # Constrain either baseline charge or baseline discharge to be non-zero
    model.addConstrs(baseline_charge_status[i] + baseline_discharge_status[i] <= 1 for i in range(hours*2))

    model.addConstrs(baseline_charge[i]<= baseline_charge_status[i]* max_power_MW for i in range(hours*2))
    model.addConstrs(baseline_discharge[i]<= baseline_discharge_status[i]* max_power_MW for i in range(hours*2))

    # Add power evolution
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
    model.addConstr(cycles_opt == gp.quicksum((power_charge[i] + power_discharge[i])/2/(capacity_MWh)/3600*timestep_opt_s for i in range(len(power_tot))))

    #add soc constraint to be within 0 and 1
    model.addConstrs((soc[i] <= 1 for i in range(len(soc))))
    model.addConstrs((soc[i] >= 0 for i in range(len(soc))))

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
    
    return daily_profit_GBP, optimal_capacity_allocation_MW, optimal_baseline_charge_MW, optimal_baseline_discharge_MW, soc_results, cycles_opt.x

def optimizer_no_aging(hours, timestep_opt_s, epex_price_GBP_MWh, 
                       clearing_prices, efficiency, max_power_MW, capacity_MWh, starting_soc, 
                       DC_high_response, DC_low_response, DM_high_response,
                       DM_low_response, DR_high_response, DR_low_response):
    
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
    cycles_opt = model.addVar(lb = 0, ub = 10,name = 'cycles')
    charge_status = model.addVars(int(3600/timestep_opt_s*hours), vtype = GRB.BINARY)
    baseline_charge_status = model.addVars(hours*2, vtype = GRB.BINARY)
    baseline_discharge_status = model.addVars(hours*2, vtype = GRB.BINARY)

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
    # Constrain either baseline charge or baseline discharge to be non-zero
    model.addConstrs(baseline_charge_status[i] + baseline_discharge_status[i] <= 1 for i in range(hours*2))

    model.addConstrs(baseline_charge[i]<= baseline_charge_status[i]* max_power_MW for i in range(hours*2))
    model.addConstrs(baseline_discharge[i]<= baseline_discharge_status[i]* max_power_MW for i in range(hours*2))

    # Add power evolution
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

    #initialize the soc to the set starting soc
    model.addConstr(soc[0] == starting_soc)

    #add the soc evolution constraint which takes into account a constant charging/discharging efficiency
    model.addConstrs(
        (soc[i] == soc[i-1] + efficiency*power_charge[i-1]/capacity_MWh/3600*timestep_opt_s - power_discharge[i-1]/capacity_MWh/3600*timestep_opt_s/efficiency)
        for i in range(1,len(soc))
    )

    # Add cycle variable constraint which calculates the total number of cycles that the battery has gone through

    model.addConstr(cycles_opt == gp.quicksum((power_charge[i] + power_discharge[i])/2/(capacity_MWh)/3600*timestep_opt_s for i in range(len(power_tot))))

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
    model.addConstrs((soc[i*1800/timestep_opt_s] <= 1-(0.25*capacity_allocation[(i//8)*6]+0.5*capacity_allocation[(i//8)*6+2]+
                    capacity_allocation[(i//8)*6+4])/capacity_MWh for i in range(hours*2)))
    model.addConstrs((soc[i*1800/timestep_opt_s] >= (0.25*capacity_allocation[(i//8)*6+1]+0.5*capacity_allocation[(i//8)*6+3]+
                    capacity_allocation[(i//8)*6+5])/capacity_MWh for i in range(hours*2)))

    model.optimize()

    ## data handling of the optimization outputs
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
    
    #retreive the optimal SOC as well as cycle count
    cycles = 0
    soc_results = []
    for i in range(len(soc)//2):
        soc_results.append(soc[i].x)
        if i > 0:
            cycles += abs(soc_results[-1] - soc_results[-2])
    
    return daily_profit_GBP, optimal_capacity_allocation_MW, optimal_baseline_charge_MW, optimal_baseline_discharge_MW, soc_results, cycles/2

def optimizer_daily_lim(hours, timestep_opt_s, epex_price_GBP_MWh, 
                       clearing_prices, efficiency, max_power_MW, capacity_MWh, starting_soc, 
                       DC_high_response, DC_low_response, DM_high_response,
                       DM_low_response, DR_high_response, DR_low_response, daily_lim):
    
    #initialize the gurobi model
    model = gp.Model('FFR_Maximize_Profit')
    model.setParam('OutputFlag',1)

    #set mip gap
    model.setParam('MIPGap', 0.01)
    model.setParam('TimeLimit', 300)

    ## add the decision variables
    #capacity allocation is a set of 6 variables for each 4-hour period to allocate a capacity in MW to
    #each of the 6 FFR products
    capacity_allocation = model.addVars(hours//4*6, lb=0, ub=max_power_MW, name="capacity_allocation")

    #baseline charge denotes the amount of power that is bought from the grid to charge the battery
    baseline_charge = model.addVars(hours*2, lb=0, ub=max_power_MW, name="baseline_charge")

    #baseline discharge denotes the amount of power that is sold to the grid to discharge the battery
    baseline_discharge = model.addVars(hours*2, lb=0, ub=max_power_MW, name="baseline_discharge")

    ## add auxiliary variables
    #the soc stores the state of charge of the battery at each time step
    soc = model.addVars(int(3600/timestep_opt_s*hours+1),lb = 0, ub = 1, name="soc")
    
    #total power is a helper variable that makes the constraints more readable and is the sum of all power
    #draws
    power_tot = model.addVars(int(3600/timestep_opt_s*hours),lb = -max_power_MW, ub = max_power_MW, name = 'power')
    
    #power charge and power discharge are the total powers in either direction that are used for soc calculation
    power_charge = model.addVars(int(3600/timestep_opt_s*hours),lb = 0, ub = max_power_MW, name = 'power')
    power_discharge = model.addVars(int(3600/timestep_opt_s*hours),lb = 0, ub = max_power_MW, name = 'power')
    
    #cycles_opt denotes the variable that stores the total number of cycles that the battery has gone through
    #in a day
    cycles_opt = model.addVars(2*hours//24, lb = 0, ub = daily_lim ,name = 'cycles')

    #charge status is a binary variable that denotes whether the battery is charging (1) or discharging (0)
    charge_status = model.addVars(int(3600/timestep_opt_s*hours), vtype = GRB.BINARY)

    #the binary variables baseline_charge_status and baseline_discharge_status denote whether the baseline
    #charge or discharge is non-zero - only one of the two can be non-zero
    baseline_charge_status = model.addVars(hours*2, vtype = GRB.BINARY)
    baseline_discharge_status = model.addVars(hours*2, vtype = GRB.BINARY)

    #add the objective function
    #calculate the value of the energy bought or sold through baselines --> division by two is to convert half hourly
    #settlement periods
    value_recharge_GBP = (-1*gp.quicksum(baseline_charge[i]*epex_price_GBP_MWh[i] for i in range(len(baseline_charge)))
                        + gp.quicksum(baseline_discharge[i]*epex_price_GBP_MWh[i] for i in range(len(baseline_discharge))))/2

    #calculate the value of the FFR services provided --> multiplication by 4 is to convert hourly clearing prices to 4-hour
    #EAC window
    value_FFR_GBP = gp.quicksum(clearing_prices[i]*capacity_allocation[i] for i in range(len(capacity_allocation)))*4

    model.setObjective(value_recharge_GBP + value_FFR_GBP, GRB.MAXIMIZE)

    #--------------------CONSTRAINTS--------------------
    # Constrain either baseline charge or baseline discharge to be non-zero
    model.addConstrs(baseline_charge_status[i] + baseline_discharge_status[i] <= 1 for i in range(hours*2))

    model.addConstrs(baseline_charge[i]<= baseline_charge_status[i]* max_power_MW for i in range(hours*2))
    model.addConstrs(baseline_discharge[i]<= baseline_discharge_status[i]* max_power_MW for i in range(hours*2))

    # Add power evolution
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

    #initialize the soc to the set starting soc
    model.addConstr(soc[0] == starting_soc)

    #add the soc evolution constraint which takes into account a constant charging/discharging efficiency
    model.addConstrs(
        (soc[i] == soc[i-1] + efficiency*power_charge[i-1]/capacity_MWh/3600*timestep_opt_s - power_discharge[i-1]/capacity_MWh/3600*timestep_opt_s/efficiency)
        for i in range(1,len(soc))
    )

    # Add cycle variable constraint which calculates the total number of cycles that the battery has gone through
    #fot each day
    model.addConstr(cycles_opt[i//(86400/timestep_opt_s)] == gp.quicksum((power_charge[i] + power_discharge[i])/2/(capacity_MWh)/3600*timestep_opt_s for i in range(len(power_tot))))

    #explicitly add the cycle limit constraint
    model.addConstrs(cycles_opt[i] <= daily_lim for i in range(2*hours//24))

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
    model.addConstrs((soc[i*1800/timestep_opt_s] <= 1-(0.25*capacity_allocation[(i//8)*6]+0.5*capacity_allocation[(i//8)*6+2]+
                    capacity_allocation[(i//8)*6+4])/capacity_MWh for i in range(hours*2)))
    model.addConstrs((soc[i*1800/timestep_opt_s] >= (0.25*capacity_allocation[(i//8)*6+1]+0.5*capacity_allocation[(i//8)*6+3]+
                    capacity_allocation[(i//8)*6+5])/capacity_MWh for i in range(hours*2)))

    model.optimize()

    ## data handling of the optimization outputs
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
    cycles_opt = model.addVar(lb = 0, ub = 10,name = 'cycles')
    charge_status = model.addVars(int(3600/timestep_opt_s*hours), vtype = GRB.BINARY)
    baseline_charge_status = model.addVars(hours*2, vtype = GRB.BINARY)
    baseline_discharge_status = model.addVars(hours*2, vtype = GRB.BINARY)

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
    # Constrain either baseline charge or baseline discharge to be non-zero
    model.addConstrs(baseline_charge_status[i] + baseline_discharge_status[i] <= 1 for i in range(hours*2))

    model.addConstrs(baseline_charge[i]<= baseline_charge_status[i]* max_power_MW for i in range(hours*2))
    model.addConstrs(baseline_discharge[i]<= baseline_discharge_status[i]* max_power_MW for i in range(hours*2))

    # Add power evolution
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
    model.addConstr(cycles_opt == gp.quicksum((power_charge[i] + power_discharge[i])/2/(capacity_MWh)/3600*timestep_opt_s for i in range(len(power_tot))))

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
    model.addConstrs((soc[i*1800/timestep_opt_s] <= 1-(0.25*capacity_allocation[(i//8)*6]+0.5*capacity_allocation[(i//8)*6+2]+
                    capacity_allocation[(i//8)*6+4])/capacity_MWh for i in range(hours*2)))
    model.addConstrs((soc[i*1800/timestep_opt_s] >= (0.25*capacity_allocation[(i//8)*6+1]+0.5*capacity_allocation[(i//8)*6+3]+
                    capacity_allocation[(i//8)*6+5])/capacity_MWh for i in range(hours*2)))

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
    
    return daily_profit_GBP, optimal_capacity_allocation_MW, optimal_baseline_charge_MW, optimal_baseline_discharge_MW, soc_results, cycles_opt.x

import sys

def optimizer_pl_aging(hours, timestep_opt_s, epex_price_GBP_MWh, 
                       clearing_prices, efficiency, max_power_MW, capacity_MWh, starting_soc, 
                       DC_high_response, DC_low_response, DM_high_response,
                       DM_low_response, DR_high_response, DR_low_response,
                       aging_cost_GBP_per_MWh, SOH):
    


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
    model.setParam('MIPGap', 0.01)
    model.setParam('TimeLimit', 150)

    ## add the decision variables
    #capacity allocation is a set of 6 variables for each 4-hour period to allocate a capacity in MW to
    #each of the 6 FFR products
    capacity_allocation = model.addVars(hours//4*6, lb=0, ub=max_power_MW, name="capacity_allocation")

    #baseline charge denotes the amount of power that is bought from the grid to charge the battery
    baseline_charge = model.addVars(hours*2, lb=0, ub=max_power_MW, name="baseline_charge")

    #baseline discharge denotes the amount of power that is sold to the grid to discharge the battery
    baseline_discharge = model.addVars(hours*2, lb=0, ub=max_power_MW, name="baseline_discharge")

    ## add auxiliary variables
    #the soc stores the state of charge of the battery at each time step
    soc = model.addVars(int(3600/timestep_opt_s*hours+1),lb = 0, ub = 1, name="soc")
    
    #total power is a helper variable that makes the constraints more readable and is the sum of all power
    #draws
    power_tot = model.addVars(int(3600/timestep_opt_s*hours),lb = -max_power_MW, ub = max_power_MW, name = 'power')
    
    #power charge and power discharge are the total powers in either direction that are used for soc calculation
    power_charge = model.addVars(int(3600/timestep_opt_s*hours),lb = 0, ub = max_power_MW, name = 'power')
    power_discharge = model.addVars(int(3600/timestep_opt_s*hours),lb = 0, ub = max_power_MW, name = 'power')
    
    #cycles_opt denotes the variable that stores the total number of cycles that the battery has gone through
    #in each day
    cycles_opt = model.addVars(2*hours//24, lb = 0, ub = 6 ,name = 'cycles')

    #charge status is a binary variable that denotes whether the battery is charging (1) or discharging (0)
    charge_status = model.addVars(int(3600/timestep_opt_s*hours), vtype = GRB.BINARY)

    #the binary variables baseline_charge_status and baseline_discharge_status denote whether the baseline
    #charge or discharge is non-zero - only one of the two can be non-zero
    baseline_charge_status = model.addVars(hours*2, vtype = GRB.BINARY)
    baseline_discharge_status = model.addVars(hours*2, vtype = GRB.BINARY)

    ##add the variables for handling of the aging
    #q_loss_cal denotes the calendar aging
    q_loss_cal = model.addVars(hours*4, lb = 0)

    #q_loss_cyc_ch and q_loss_cyc_dch denote the cyclic aging during charge and discharge respectively
    q_loss_cyc_ch = model.addVars(hours//4,lb = 0)
    q_loss_cyc_dis = model.addVars(hours//4,lb = 0)
    
    ##add the sos type 2 lambda variables as weights for the piecewise linearised degradation functions
    lambda_cal = {}
    lambda_cyc_ch = {}
    lambda_cyc_dis = {}

    for i in range(len(q_loss_cal)):
        lambda_cal[i] = model.addVars(set_I_cal,lb = 0, ub = 1, name = f'lambda_cal_{i}')
    
    for i in range(len(q_loss_cyc_ch)):
        lambda_cyc_ch[i] = model.addVars(set_J_cyc,lb = 0, ub = 1, name = f'lambda_cyc_ch_{i}')

    for i in range(len(q_loss_cyc_dis)):
        lambda_cyc_dis[i] = model.addVars(set_J_cyc,lb = 0, ub = 1, name = f'lambda_cyc_dis_{i}')
    


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

    #calculate the aging cost, the division by 1e6 just divides out the factor that was added for numerical accuracy
    #of the solver
    value_aging_GBP = capacity_MWh*aging_cost_GBP_per_MWh/0.2*(gp.quicksum(q_loss_cal)+gp.quicksum(q_loss_cyc_ch)+gp.quicksum(q_loss_cyc_dis))/1e6

    model.setObjective(value_recharge_GBP + value_FFR_GBP-value_aging_GBP, GRB.MAXIMIZE)

    #--------------------CONSTRAINTS--------------------
    # Constrain either baseline charge or baseline discharge to be non-zero
    model.addConstrs(baseline_charge_status[i] + baseline_discharge_status[i] <= 1 for i in range(hours*2))

    model.addConstrs(baseline_charge[i]<= baseline_charge_status[i]* max_power_MW for i in range(hours*2))
    model.addConstrs(baseline_discharge[i]<= baseline_discharge_status[i]* max_power_MW for i in range(hours*2))

    # Add power evolution
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
    model.addConstr(cycles_opt[i//(86400/timestep_opt_s)] == gp.quicksum((power_charge[i] + power_discharge[i])/2/(capacity_MWh)/3600*timestep_opt_s for i in range(len(power_tot))))

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
    model.addConstrs((soc[i*1800/timestep_opt_s] <= 1-(0.25*capacity_allocation[(i//8)*6]+0.5*capacity_allocation[(i//8)*6+2]+
                    capacity_allocation[(i//8)*6+4])/capacity_MWh for i in range(hours*2)))
    model.addConstrs((soc[i*1800/timestep_opt_s] >= (0.25*capacity_allocation[(i//8)*6+1]+0.5*capacity_allocation[(i//8)*6+3]+
                    capacity_allocation[(i//8)*6+5])/capacity_MWh for i in range(hours*2)))

    #add the constraints for the calendar aging
    for i in range(len(q_loss_cal)):
        #creates a sos type 2 array for linear interpolation between within the piecewise linearized function
        model.addSOS(GRB.SOS_TYPE2, vars = lambda_cal[i])
        #makes sure that the interpolation sums to 1
        model.addConstr(gp.quicksum(lambda_cal[i][j] for j in set_I_cal) == 1)
        #sets the lambda factors to correct weights for the two points to accommodate the average soc over the 15 min period
        model.addConstr(gp.quicksum(lambda_cal[i][j]*c_lin_x_cal[j] for j in set_I_cal) == gp.quicksum(soc[i*900//timestep_opt_s+k] for k in range(900//timestep_opt_s))/(900/timestep_opt_s))
        #finds the corresponding q_loss point
        model.addConstr(gp.quicksum(lambda_cal[i][j]*c_lin_y_cal[j] for j in set_I_cal) == q_loss_cal[i])

    #add the constraints for the cyclic aging during charge
    for i in range(len(q_loss_cyc_ch)):
        #creates a sos type 2 array for linear interpolation between within the piecewise linearized function
        model.addSOS(GRB.SOS_TYPE2, vars = lambda_cyc_ch[i])

        #makes sure that the interpolation sums to 1
        model.addConstr(gp.quicksum(lambda_cyc_ch[i][j] for j in set_I_cal) == 1)

        #sets the lambda factors to correct weights for the two points that represent the amount of energy
        #in the 4h period
        model.addConstr(gp.quicksum(lambda_cyc_ch[i][j]*c_lin_x_cyc[j] for j in set_I_cal) == 
                        gp.quicksum(power_charge[t]/3600*timestep_opt_s for t in range(int(i*14400/timestep_opt_s),int(i*14400//timestep_opt_s+14400//timestep_opt_s))))
        
        #finds the corresponding q_loss point
        model.addConstr(gp.quicksum(lambda_cyc_ch[i][j]*c_lin_y_cyc[j] for j in set_I_cal) == q_loss_cyc_ch[i])

    #add the constraints for the cyclic aging during discharge
    for i in range(len(q_loss_cyc_dis)):
        #creates a sos type 2 array for linear interpolation between within the piecewise linearized function
        model.addSOS(GRB.SOS_TYPE2, vars = lambda_cyc_dis[i])

        #makes sure that the interpolation sums to 1
        model.addConstr(gp.quicksum(lambda_cyc_dis[i][j] for j in set_I_cal) == 1)

        #sets the lambda factors to correct weights for the two points that represent the amount of energy
        #in the 4h period
        model.addConstr(gp.quicksum(lambda_cyc_dis[i][j]*c_lin_x_cyc[j] for j in set_I_cal) == 
                        gp.quicksum(power_discharge[t]/3600*timestep_opt_s for t in range(int(i*14400/timestep_opt_s),int(i*14400//timestep_opt_s+14400//timestep_opt_s))))
        
        #finds the corresponding q_loss point
        model.addConstr(gp.quicksum(lambda_cyc_dis[i][j]*c_lin_y_cyc[j] for j in set_I_cal) == q_loss_cyc_dis[i])

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
    
    #retreive the total amount of ageing incurred
    cal_aging = 0
    for i in range(len(q_loss_cal)):
        cal_aging += q_loss_cal[i].x/1e6

    power_output = []
    for i in range(len(power_tot)):
        power_output.append(power_tot[i].x)

    cyc_aging_ch = 0
    cyc_aging_dis = 0
    for i in range(len(q_loss_cyc_ch)):
        print(q_loss_cyc_ch[i].x/1e6)
        print(q_loss_cyc_dis[i].x/1e6)
        cyc_aging_ch += q_loss_cyc_ch[i].x/1e6
        cyc_aging_dis += q_loss_cyc_dis[i].x/1e6
    
    print('Calendar Aging for 2 days: ', cal_aging)
    print('Cyclic Aging Charge: ', cyc_aging_ch)
    print('Cyclic Aging Discharge: ', cyc_aging_dis)
    print('Number of cycles: ',cycles_opt[0].x+cycles_opt[1].x)
    
    #retreive the optimal SOC as well as cycle count
    soc_results = []
    for i in range(len(soc)//2):
        soc_results.append(soc[i].x)
    
    
    return daily_profit_GBP, optimal_capacity_allocation_MW, optimal_baseline_charge_MW, optimal_baseline_discharge_MW, soc_results, cycles_opt[0].x
