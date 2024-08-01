import pandas as pd 
import numpy as np
import time 
from optimizers.optimizer_MILP import optimizer_cyc_aging


from configparser import ConfigParser
from utils.utils import Frequency, FrequencyProfile, AuctionData
from datetime import timedelta
from simses.main import SimSES
from simses.commons.state.technology.lithium_ion import LithiumIonState
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

import sys


class BiddingSim():
    def __init__(self, config_sim: ConfigParser, config_opt: ConfigParser, config_ana: ConfigParser):

        self.__config_sim = config_sim
        self.__config_opt = config_opt
        self.__config_ana = config_ana
        self.__start_date = pd.to_datetime(self.__config_opt['TIMEFRAME']['START_DATE'])-pd.Timedelta(hours=1)
        self.__no_of_days = int(self.__config_opt['TIMEFRAME']['NO_OF_DAYS'])
        self.__frequency_path = str(self.__config_opt['PROFILE']['FREQUENCY_PROFILE']) 
        self.__auction_data_path = str(self.__config_opt['PROFILE']['AUCTION_DATA'])
        self.__price_data_path = str(self.__config_opt['PROFILE']['PRICE_DATA'])
        self.__timestep_opt_s = int(self.__config_opt['GENERAL']['OPT_TIMESTEP'])
        self.__timestep_sim_s = int(self.__config_sim['GENERAL']['SIM_TIMESTEP'])
        self.__max_power_MW = float(self.__config_opt['GENERAL']['MAX_POWER'])
        self.__efficiency = float(self.__config_opt['GENERAL']['EFFICIENCY'])
        self.__initial_soc = float(self.__config_sim['BATTERY']['START_SOC'])
        self.__horizon_length = int(self.__config_opt['GENERAL']['HORIZON'])
        self.__aging_cost_GB_per_cycle = float(self.__config_opt['GENERAL']['AGING_COST'])
        
        #separate by commas and convert to float
        self.__capacity = float(self.__config_sim['STORAGE_SYSTEM']['STORAGE_TECHNOLOGY'].split(',')[1]) 
        self.simses = SimSES(path='results', name='test', simulation_config=self.__config_sim, analysis_config=self.__config_ana)              
    
    #def __generate_aging_cost(self)
    
    #accepted_bids = self.__run_market(bids)
        
    def run(self):
        # Read the frequency profile
        frequency_profile = pd.read_csv(self.__frequency_path, index_col=0, parse_dates=True)
        
        # Generate the frequency profile
        frequency_profile = FrequencyProfile(frequency_profile)
        frequency_profile.add_response()

        # Resample to appropriate intervals and aggregate
        frequency_profile_resampled_opt_mean = frequency_profile.resample(str(self.__timestep_opt_s)+'s').mean()
        frequency_profile_resampled_sim = frequency_profile.resample(str(self.__timestep_sim_s)+'s').mean()

        #read the nordpool price data
        nordpool_price_GBP_MWh = pd.read_csv(self.__price_data_path, index_col=0, parse_dates=True)

        #read the auction data
        auction_data = pd.read_csv(self.__auction_data_path)
        auction_data = AuctionData(auction_data)
        auction_data.rearrange()

        current_capacity_MWh = self.__capacity/1000000
        initial_soc = self.__initial_soc

        #initialize trackers for daily optimization data
        daily_date = []
        daily_cycles_tracker = []
        daily_profit_GBP_tracker = []
        daily_capacity_allocation_MW_tracker = []
        daily_baseline_charge_MW_tracker = []
        daily_baseline_discharge_MW_tracker = []
        daily_degradation_tracker = []
        daily_fullfillment_tracker = []
        daily_revenue_from_FFR_tracker = []
        daily_aging_cost_tracker = []
        daily_arbitrage_tracker = []
        optimization_time_tracker = []
        simulation_time_tracker = []

        for day in range(self.__no_of_days):

            #get the current day in epoch
            current_day_start = self.__start_date + timedelta(days=day)
            current_day_end = current_day_start + timedelta(days=1)
            current_horizon_end = current_day_start + timedelta(days=self.__horizon_length)

            #get the frequency profile for the current day
            frequency_profile_opt_day_mean = frequency_profile_resampled_opt_mean.loc[(frequency_profile_resampled_opt_mean.index >= current_day_start) & 
                                                        (frequency_profile_resampled_opt_mean.index < current_horizon_end)]

            frequency_profile_sim_day = frequency_profile_resampled_sim.loc[(frequency_profile_resampled_sim.index >= current_day_start) &
                                                        (frequency_profile_resampled_sim.index < current_day_end)]
            
            clearing_prices = self.__extract_clearing_prices(auction_data, current_day_start)

            #get the nordpool prices for the current day
            nordpool_price_GBP_MWh_horizon = np.array(nordpool_price_GBP_MWh['UK Price (GBP)'].loc[(nordpool_price_GBP_MWh.index >= current_day_start) &
                                                        (nordpool_price_GBP_MWh.index < current_horizon_end)])

            #optimize the capacity allocation for the day
            DC_high_response_mean = np.array(frequency_profile_opt_day_mean['DC_high'])
            DC_low_response_mean = np.array(frequency_profile_opt_day_mean['DC_low'])
            DM_high_response_mean = np.array(frequency_profile_opt_day_mean['DM_high'])
            DM_low_response_mean = np.array(frequency_profile_opt_day_mean['DM_low'])
            DR_high_response_mean = np.array(frequency_profile_opt_day_mean['DR_high'])
            DR_low_response_mean = np.array(frequency_profile_opt_day_mean['DR_low'])

            print('Optimizer for day',day,)
            start_time_opt = time.time()
            daily_profit_GBP, optimal_capacity_allocation_MW, optimal_baseline_charge_MW,\
            optimal_baseline_discharge_MW, soc_results, cycles = optimizer_cyc_aging(self.__horizon_length*24, 
                    self.__timestep_opt_s, nordpool_price_GBP_MWh_horizon, 
                    clearing_prices, self.__efficiency, self.__max_power_MW, current_capacity_MWh, initial_soc, 
                    DC_high_response_mean, DC_low_response_mean, DM_high_response_mean,
                    DM_low_response_mean, DR_high_response_mean, DR_low_response_mean,
                    aging_cost_GBP_per_cycle=self.__aging_cost_GB_per_cycle)
            
            end_time_opt = time.time()

            #convert optimal capacity allocation into a dictionary
            daily_capacity_allocation_MW = {
            'DR_high': [],
            'DR_low': [],
            'DC_high': [],
            'DC_low': [],
            'DM_high': [],
            'DM_low': []
                }
            for i in range(0,6):
                daily_capacity_allocation_MW['DC_high'].append(optimal_capacity_allocation_MW[6*i])
                daily_capacity_allocation_MW['DC_low'].append(optimal_capacity_allocation_MW[6*i+1])
                daily_capacity_allocation_MW['DM_high'].append(optimal_capacity_allocation_MW[6*i+2])
                daily_capacity_allocation_MW['DM_low'].append(optimal_capacity_allocation_MW[6*i+3])
                daily_capacity_allocation_MW['DR_high'].append(optimal_capacity_allocation_MW[6*i+4])
                daily_capacity_allocation_MW['DR_low'].append(optimal_capacity_allocation_MW[6*i+5])
            
            print('Optimal Capacity Allocation for day: ', daily_capacity_allocation_MW)
            print('Daily Profit: ', daily_profit_GBP)
            print('Final SOC: ', soc_results[len(soc_results)//2-1])
            print('Baseline Charge: ', optimal_baseline_charge_MW)
            print('Baseline Discharge: ', optimal_baseline_discharge_MW)

            #plot the soc_results
            #plt.plot(soc_results)
            #plt.show()
            
            #generate power demand profile based on the allocated capacities
            power_demand_MW = self.__generate_power_demand(frequency_profile_sim_day, daily_capacity_allocation_MW, optimal_baseline_charge_MW, optimal_baseline_discharge_MW)

            #run the battery simulation tool with output powerdemand
            print('Running Battery Simulation for day: ', day)
            start_time_sim = time.time()
            fullfill, soc, capacity_Wh = self.__run_daily_battery_simulation(power_demand_MW)
            print('Fulfillment: ', fullfill)
            print('SOC: ', soc[-1])
            print('Capacity: ', capacity_Wh)
            end_time_sim = time.time()

            #update the battery capacity and soc with the results from the simulation
            current_capacity_Wh = capacity_Wh/1e6
            initial_soc = soc[-1]

            print('SOC after simulation: ', initial_soc)
            print('Capacity after simulation: ', current_capacity_Wh)
            print('Fullfillment after simulation: ', fullfill)

            #track the daily results
            daily_date.append(current_day_end.date())
            daily_profit_GBP_tracker.append(daily_profit_GBP)
            daily_capacity_allocation_MW_tracker.append(daily_capacity_allocation_MW)
            daily_baseline_charge_MW_tracker.append(optimal_baseline_charge_MW)
            daily_baseline_discharge_MW_tracker.append(optimal_baseline_discharge_MW)
            daily_cycles_tracker.append(cycles)
            daily_fullfillment_tracker.append(fullfill)
            daily_revenue_from_FFR_tracker.append(np.array(clearing_prices[:36]).dot(np.array(optimal_capacity_allocation_MW))*4)
            daily_aging_cost_tracker.append(0)
            daily_arbitrage_tracker.append((-np.array(optimal_baseline_charge_MW)+np.array(optimal_baseline_discharge_MW)).dot(np.array(nordpool_price_GBP_MWh_horizon[:48]))/2)
            daily_degradation_tracker.append(capacity_Wh/1e6)
            optimization_time_tracker.append(end_time_opt-start_time_opt)
            simulation_time_tracker.append(end_time_sim-start_time_sim)

        #create a df to export results into a csv
        print('Exporting results...')
        results = pd.DataFrame({'Date': daily_date, 'Total_Profit_GBP': daily_profit_GBP_tracker, 'Capacity_Allocation_MW': daily_capacity_allocation_MW_tracker,
                                'Baseline_Charge_MW': daily_baseline_charge_MW_tracker, 'Baseline_Discharge_MW': daily_baseline_discharge_MW_tracker,
                                'Cycles': daily_cycles_tracker, 'Fullfillment': daily_fullfillment_tracker, 'Revenue_from_FFR': daily_revenue_from_FFR_tracker,
                                'Aging_Cost': daily_aging_cost_tracker, 'Arbitrage': daily_arbitrage_tracker, 'Degradation': daily_degradation_tracker,
                                'Optimization_Time': optimization_time_tracker, 'Simulation_Time': simulation_time_tracker})
        results.to_csv('results/0801/gurobi_cyc_aging_10s_full_run.csv')
        print('Done!')
        self.simses.close()

    #function that converts the daily frequency as well as a dictionary of capacity allocations for all services throughout the day
    #into a power demand profile which can then be used to feed into the battery simulation tool    
    def __generate_power_demand(self, one_day_frequency, capacity_allocation, baselines_charge, baselines_discharge):
        # Initialize the power_demand DataFrame
        power_demand = pd.DataFrame(index=one_day_frequency.index, columns=['frequency', 'power_demand'])
        
        # Set the frequency column in one go
        power_demand['frequency'] = one_day_frequency['f']
        power_demand['power_demand'] = 0  # Initialize power_demand to 0
        
        for key in capacity_allocation.keys():
            # Determine the adjustment based on whether 'low' is in the key
            adjustment = -1 if 'low' in key else 1
            capacity_windows = capacity_allocation[key] # Convert to numpy array if not already
            
            # Calculate the capacity for every timestep
            capacity_seconds = np.concatenate((np.ones(int(3600/self.__timestep_sim_s*4))*capacity_windows[0],np.ones(int(3600/self.__timestep_sim_s*4))*capacity_windows[1],
                                               np.ones(int(3600/self.__timestep_sim_s*4))*capacity_windows[2],np.ones(int(3600/self.__timestep_sim_s*4))*capacity_windows[3],
                                               np.ones(int(3600/self.__timestep_sim_s*4))*capacity_windows[4], np.ones(int(3600/self.__timestep_sim_s*4))*capacity_windows[5]),axis = 0)
            
            # Calculate the contribution for this key in a vectorized manner
            contribution = one_day_frequency[key].to_numpy() * capacity_seconds * adjustment
            
            # Add the contribution to the power_demand column
            power_demand['power_demand'] += contribution
        
        baseline_array = np.zeros(len(one_day_frequency))
        
        for i in range(48):
            baseline_array[i*int(1800/self.__timestep_sim_s):(i+1)*int(1800/self.__timestep_sim_s)] += baselines_charge[i]
            baseline_array[i*int(1800/self.__timestep_sim_s):(i+1)*int(1800/self.__timestep_sim_s)] -= baselines_discharge[i]

        power_demand['power_demand'] += baseline_array 

        return power_demand
    
    def __run_daily_battery_simulation(self, power_demand):
        '''
        This function runs the battery simulation for a single day and returns the resulting aging'''

        #initialize tracker
        power_sim_ac_delivered = np.zeros(int(86400/self.__timestep_sim_s))
        power_sim_ac_requested = np.zeros(int(86400/self.__timestep_sim_s))
        soc_sim = np.zeros(int(86400/self.__timestep_sim_s))
        fullfill = np.zeros(int(86400/self.__timestep_sim_s))
        sim_losses_charge = np.zeros(int(86400/self.__timestep_sim_s))
        sim_losses_discharge = np.zeros(int(86400/self.__timestep_sim_s))
        timestamps = power_demand.index.astype(int) / 10**9

        for i in range(0,int(86400/self.__timestep_sim_s)):
            self.simses.run_one_simulation_step(time = timestamps[i], power = power_demand['power_demand'].iloc[i]*1e6)
            # track values from simses
            power_sim_ac_delivered[i] = self.simses.state.ac_power_delivered
            power_sim_ac_requested[i] = self.simses.state.ac_power
            soc_sim[i] = self.simses.state.soc
            fullfill[i] = self.simses.state.ac_fulfillment

            if self.simses.state.ac_power_delivered > 0:
                sim_losses_charge[i] = self.simses.state.pe_losses + self.simses.state.dc_power_loss
                sim_losses_discharge[i] = 0
            elif self.simses.state.ac_power_delivered < 0:
                sim_losses_charge[i] = 0
                sim_losses_discharge[i] = self.simses.state.pe_losses + self.simses.state.dc_power_loss
            else:
                sim_losses_charge[i] = self.simses.state.pe_losses + self.simses.state.dc_power_loss
                sim_losses_discharge[i] = self.simses.state.pe_losses + self.simses.state.dc_power_loss
            
        return np.mean(fullfill), soc_sim, self.simses.state.capacity
    
    
    
    def __extract_clearing_prices(self,auction_data, day: pd.Timestamp):
        '''
        This function extracts the clearing prices for a given day and returns them in a list ordered by day and 
        DC/DM/DR high/low
        '''
        clearing_prices =[]
        day_start = day
        day_end = day + pd.Timedelta(days=self.__horizon_length)
        auction_data = auction_data.loc[(auction_data['dtm'] >= day_start) & (auction_data['dtm'] < day_end)]

        for index,row in auction_data.iterrows():
            for key in auction_data.columns[1:]:
                clearing_prices.append(row[key])
        
        return clearing_prices

if __name__ == '__main__':
    config_sim = ConfigParser()
    config_opt = ConfigParser()
    config_ana = ConfigParser()
    config_ana.read('configs/analysis.optsim.ini')
    config_opt.read('configs/optimization.optsim.ini')
    config_sim.read('configs/simulation.optsim.ini')
    x = BiddingSim(config_sim, config_opt, config_ana)
    x.run()