import pandas as pd 
import numpy as np
import time 
from configparser import ConfigParser
from utils.utils import Frequency, FrequencyProfile, AuctionData
from datetime import timedelta
from simses.main import SimSES
from simses.commons.state.technology.lithium_ion import LithiumIonState
import gurobipy as gp
from gurobipy import GRB
class BiddingSim():
    def __init__(self, config_sim: ConfigParser, config_opt: ConfigParser, config_ana: ConfigParser):

        self.__config_sim = config_sim
        self.__config_opt = config_opt
        self.__config_ana = config_ana
        self.__start_date = pd.to_datetime(self.__config_opt['TIMEFRAME']['START_DATE'])-pd.Timedelta(hours=1)
        self.__no_of_weeks = int(self.__config_opt['TIMEFRAME']['NO_OF_WEEKS'])
        self.__auction_data = pd.read_csv(str(self.__config_opt['PROFILE']['AUCTION_DATA']))
        self.__intraday_prices = pd.read_csv(str(self.__config_opt['PROFILE']['INTRADAY_PRICES']))
        self.__frequency_path = str(self.__config_opt['PROFILE']['FREQUENCY_PROFILE']) 
        self.__auction_data_path = str(self.__config_opt['PROFILE']['AUCTION_DATA'])
        
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

        #read the auction data
        auction_data = pd.read_csv(self.__auction_data_path)
        auction_data = AuctionData(auction_data)
        auction_data.rearrange()
        
        for day in range(self.__no_of_weeks*7):

            #get the current day in epoch
            current_day_start = self.__start_date + timedelta(days=day)
            current_day_end = current_day_start + timedelta(days=1)

            #generate bids for the day
            bids = {'DR_high' : [-2,-2,-2,-2,-2,-2], 
                                   'DR_low' : [5,5,5,5,5,5], 
                                   'DC_high' : [0,0,0,0,0,0], 
                                   'DC_low' : [2.5,2.5,2.5,2.5,2.5,2.5], 
                                   'DM_high' : [0,0,0,0,0,0], 
                                   'DM_low' : [2,2,2,2,2,2]}
            
            clearing_prices
            
            #check which bids are accepted and return resulting capacity allocation
            capacity_allocation = self.__run_market(bids, offers)
            
            #check whether capacity allocation is valid
            print(self.__capacity)

            # Retrieve all lists from the dictionary values
            allocations = capacity_allocation.values()
            
            # Flatten all lists into a single list
            allocations_concat = [element for sublist in allocations for element in sublist]
            
            # Sum all elements in the flattened list
            total_allocation = sum(allocations_concat)

            if total_allocation > self.__capacity:
                raise ValueError('Capacity allocation exceeds storage capacity')
            
            
            #generate power demand profile based on the allocated capacities
            power_demand = self.__generate_power_demand(frequency_profile.loc[(frequency_profile.index >= current_day_start) & 
                                                        (frequency_profile.index < current_day_end)], capacity_allocation)
            
            #run the battery simulation tool with output powerdemand
            print('Running Battery Simulation for day: ', day)
            fullfill, soc, capacity = self.__run_daily_battery_simulation(power_demand)
            print('Fulfillment: ', fullfill)
            print('SOC: ', soc)
            print('Capacity: ', capacity)
            

            

    #function that converts the daily frequency as well as a dictionary of capacity allocations for all services throughout the day
    #into a power demand profile which can then be used to feed into the battery simulation tool    
    def __generate_power_demand(self, one_day_frequency, capacity_allocation):
        # Initialize the power_demand DataFrame
        power_demand = pd.DataFrame(index=one_day_frequency.index, columns=['frequency', 'power_demand'])
        
        # Set the frequency column in one go
        power_demand['frequency'] = one_day_frequency['f']
        power_demand['power_demand'] = 0  # Initialize power_demand to 0
        
        for key in capacity_allocation.keys():
            # Determine the adjustment based on whether 'low' is in the key
            adjustment = -1 if 'low' in key else 1
            hourly_capacity = np.array(capacity_allocation[key])  # Convert to numpy array if not already
            
            # Calculate the hour block indices
            hour_blocks = (one_day_frequency.index.hour // 4).to_numpy()  # Convert to numpy array
            
            # Calculate the contribution for this key in a vectorized manner
            contribution = one_day_frequency[key].to_numpy() * hourly_capacity[hour_blocks] * adjustment
            
            # Add the contribution to the power_demand column
            power_demand['power_demand'] += contribution

        return power_demand
    
    def __run_daily_battery_simulation(self, power_demand):
        '''
        This function runs the battery simulation for a single day and returns the resulting aging'''

        #initialize tracker
        power_sim_ac_delivered = np.zeros(86400)
        power_sim_ac_requested = np.zeros(86400)
        soc_sim = np.zeros(86400)
        fullfill = np.zeros(86400)
        sim_losses_charge = np.zeros(86400)
        sim_losses_discharge = np.zeros(86400)
        timestamps = power_demand.index.astype(int) / 10**9

        for i in range(0,86400):
            self.simses.run_one_simulation_step(time = timestamps[i], power = power_demand['power_demand'].iloc[i])
            
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
            
        return np.mean(fullfill), soc_sim[-1], self.simses.state.capacity
    
    def __run_market(self, clearing_prices, bids):
        # Create a new model
        m = gp.Model('market_optimizer')

        # Set Gurobi parameter to suppress output
        m.setParam('OutputFlag', 0)

        # Add binary decision variables
        allocation_DR = m.addVars(6, vtype=GRB.BINARY, name='allocation_DR')
        allocation_DC = m.addVars(6, vtype=GRB.BINARY, name='allocation_DC')
        allocation_DM = m.addVars(6, vtype=GRB.BINARY, name='allocation_DM')

        # Define the objective function using gp.quicksum for efficiency
        m.setObjective(gp.quicksum(
            allocation_DR[i] * (clearing_prices['DR_high'][i] - bids['DR_high'][i] + clearing_prices['DR_low'][i] - bids['DR_low'][i]) +
            allocation_DC[i] * (clearing_prices['DC_high'][i] - bids['DC_high'][i] + clearing_prices['DC_low'][i] - bids['DC_low'][i]) +
            allocation_DM[i] * (clearing_prices['DM_high'][i] - bids['DM_high'][i] + clearing_prices['DM_low'][i] - bids['DM_low'][i])
            for i in range(6)
        ), GRB.MAXIMIZE)

        # Add constraints to ensure that the sum of allocations for each time period does not exceed 1
        m.addConstrs((allocation_DR[i] + allocation_DC[i] + allocation_DM[i] <= 1 for i in range(6)), name="allocation_limit")

        # Optimize the model
        m.optimize()

        # Prepare dictionary to store the results
        allocations = {
            'DR_high': [],
            'DR_low': [],
            'DC_high': [],
            'DC_low': [],
            'DM_high': [],
            'DM_low': []
        }

        # Check if optimization was successful and store results
        if m.status == GRB.OPTIMAL:
            for i in range(6):
                allocations['DR_high'].append(allocation_DR[i].x)
                allocations['DR_low'].append(allocation_DC[i].x)
                allocations['DC_high'].append(allocation_DM[i].x)
                allocations['DC_low'].append(allocation_DM[i].x)
                allocations['DM_high'].append(allocation_DM[i].x)
                allocations['DM_low'].append(allocation_DM[i].x)

        return allocations


if __name__ == '__main__':
    config_sim = ConfigParser()
    config_opt = ConfigParser()
    config_ana = ConfigParser()
    config_ana.read('configs/analysis.optsim.ini')
    config_opt.read('configs/optimization.optsim.ini')
    config_sim.read('configs/simulation.optsim.ini')
    x = BiddingSim(config_sim, config_opt, config_ana)
    x.run()