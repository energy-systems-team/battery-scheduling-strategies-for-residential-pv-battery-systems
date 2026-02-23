"""
Source code for running the HEMS simulations
"""

from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import pandas as pd
import numpy as np
import os
from Functions import read_data
from HEM_strategies import no_battery, SCM, MPC, MILP_whole_year, MILP_BC_whole_year
from Battery.Inverter import Inverter
from Battery.Battery import Battery

class Simulation:
    def __init__(self, filename, inputfiletype, forecast, spot_multiplier=1):
        """
        Inputs:
            filename: (str) filepath for input data
            inputfiletype: (str) indicator what type the input data is ('input' or 'result')
            forecast: (str) either "perfect" or "naive"
            spot_multiplier: (float) spot price multiplier (e.g., if 2, spot prices are multiplied by 2) 
        """
        # Load data if filetype is 'input'
        if inputfiletype == 'input':
            data_x, spot, future_spots, pv, pv_forecast, \
            load, load_forecast, bat, datetime = read_data(filename, 
                                                            first_month=1, 
                                                            last_month=12,
                                                            return_datetime=True, 
                                                            multiply_data=1,
                                                            forecast=forecast,
                                                            pv_inverter=None, 
                                                            spot_multiplier=spot_multiplier)
            self.data_x, self.spot, self.future_spots, self.pv, self.pv_forecast, \
            self.load, self.load_forecast, self.bat, self.datetime = data_x, spot, future_spots, pv, pv_forecast, \
                                                                    load, load_forecast, bat, datetime
            # Initialize result variable
            self.results = None
        
            self.MARGIN = 0.4  # c/kWh
            # Transmission + electricity tax (inc. VAT) [2018, 2019, 2020, 2021, 2022, 2023]
            self.FIXED_COSTS = [5.69, 5.90, 5.90, 5.44, 5.01, 5.01] # c/kWh
            self.FIXED_COST = self.FIXED_COSTS[pd.Series(self.datetime)[0].year-2018]
        
        # If filetype is 'result' then read only simulation results
        elif inputfiletype == 'result':
            self.results = pd.read_csv(filename, parse_dates=['datetime'])


    def run_HEMS(self, battery, strategy):

        if strategy == 'NO-BATTERY':
            results, cost = no_battery(self.pv, self.load, self.spot, self.MARGIN, self.FIXED_COST, battery)
        
        elif strategy == 'SCM':
            results, cost = SCM(self.pv, self.load, self.spot, self.MARGIN, self.FIXED_COST, battery)
        
        elif strategy == 'MPC':
            results, cost = MPC(self.pv, self.pv_forecast, self.load, self.load_forecast, self.spot, 
                                self.future_spots, self.datetime, self.MARGIN, self.FIXED_COST, 
                                battery, bat_cost=False)

        elif strategy == 'MPC-BC':
            results, cost = MPC(self.pv, self.pv_forecast, self.load, self.load_forecast, self.spot, 
                                self.future_spots, self.datetime, self.MARGIN, self.FIXED_COST, 
                                battery, bat_cost=True)
            
        elif strategy == 'MILP-YEAR':
            results, cost = MILP_whole_year(self.pv, self.pv_forecast, self.load, self.load_forecast, self.spot, 
                     self.datetime, self.MARGIN, self.FIXED_COST, battery)
        
        elif strategy == 'MILP-BC-YEAR':
            results, cost = MILP_BC_whole_year(self.pv, self.pv_forecast, self.load, self.load_forecast, self.spot, 
                     self.datetime, self.MARGIN, self.FIXED_COST, battery)


        print('Yearly cost:', cost)

        # Convert numpy array to pandas df and add additional columns
        if strategy == 'MILP-ENERGY-ARBITRAGE-YEAR':
            results_df = pd.DataFrame(data=results, columns=['pv_to_house', 'pv_to_bat', 'bat_to_house', 'grid_to_house', 'pv_to_grid', 
                                                             'grid_to_bat', 'bat_to_grid', 'pv_to_bat_grid_wasted', 
                                                             'to_house_wasted', 'soc', 'cost', 'profit', 'delta_SOH_opt'])
        else:
            results_df = pd.DataFrame(data=results, columns=['pv_to_house', 'pv_to_bat', 'bat_to_house',
                                                        'grid_to_house', 'pv_to_grid','pv_to_bat_grid_wasted', 
                                                        'to_house_wasted', 'soc', 'cost', 'profit', 'delta_SOH_opt'])
        results_df['spot'] = self.spot
        results_df['datetime'] = self.datetime
        results_df['power'] = self.pv
        results_df['load'] = self.load
        results_df['battery_capacity'] = battery.capacity_history
        results_df['deltaSOH_cyclic'] = battery.deltaSOH_cyclic_history
        results_df['deltaSOH_float'] = battery.deltaSOH_float_history
        results_df['battery_deltaSOH'] = battery.deltaSOH_history
        results_df['battery_usage_cost'] = battery.deltaSOH_history * battery.CAPEX
        results_df['aging_factor_cyclic'] = battery.aging_factor_cyclic_history
        results_df['aging_factor_float'] = battery.aging_factor_float_history
        results_df['battery_aging_factor'] = battery.aging_factor_history
        self.results = results_df

    def print_results(self):
        self_consumed = self.results.pv_to_house + self.results.pv_to_bat
        load = self.results.pv_to_house + self.results.bat_to_house + self.results.grid_to_house
        prod = self.results.pv_to_house + self.results.pv_to_bat + self.results.pv_to_grid
        scr = self_consumed.sum()/prod.sum()
        ssr = self_consumed.sum()/load.sum()
        cost = (self.results.cost.sum() - self.results.profit.sum())

        print('=======RESULTS===================')
        Csoh = np.sum(self.results.battery_usage_cost)
        print('Load:', load.sum(), self.results.load.sum())
        print('Max SOC:', self.results.soc.max())
        print('Capacity in the end:', self.results.battery_capacity.min())
        print('SCR:', scr)
        print('SSR:', ssr)
        print('Cost:', cost)
        print('Delta SOH:', np.sum(self.results.battery_deltaSOH))
        print('Csoh:', Csoh)
        print('Total costs:', cost+Csoh)


def read_file():
    """
    Function for reading input data. Asks the input data with GUI.
    The input file can be either
    (1) already processed input data, including both consumption and production data
    (2) result data, in which case only the results will be printed
    Returns:
        tuple: (filename (str), filetype (input or result))
    """

    Tk().withdraw()
    filename = askopenfilename()
    if 'data' in filename:
        filetype = 'input'
    elif 'Results' in filename:
        filetype = 'result'

    return filename, filetype


def run_simulation_matrix(input_grid):
    for params in input_grid:
        input_file_base = params['input_file']
        strategy = params['strategy']
        forecast = params['forecast']
        include_bat_degradation = params['battery_degradation']
        save_folder = params['save_folder']
        pv_capacity = params['pv_capacity']
        years = params['years']
        spot_multiplier = params['spot_multiplier']

        include_bat_degradation = include_bat_degradation == 'yes'  # Make this a boolean variable
        battery = Battery(include_bat_degradation)

        # Iterate through the years
        for year in years:
            input_file = input_file_base + str(year) + '_UTC03.csv'
            print(f'Running:\nFile: {input_file}\nStrategy: {strategy}\nForecast: {forecast}\nBattery degradation: {include_bat_degradation}')
            
            # Save name
            degr = 'degr_' if include_bat_degradation else 'nodegr_'
            drive, path_and_file = os.path.splitdrive(input_file)
            path, save_file = os.path.split(path_and_file)
            save_file = strategy + '_' + forecast + '_' + degr + save_file
            # Check if simulation results exist, and skip if so
            existing_simulations = os.listdir(save_folder)
            if save_file in existing_simulations:
                print('Simulation already exist, skipping\n')
                continue
            
            save_file = os.path.join(save_folder, save_file)

            # Run simulation
            simulation = Simulation(input_file, 'input', forecast, spot_multiplier)
            simulation.run_HEMS(battery, strategy)

            # Save results
            simulation.results.to_csv(save_file, index=False)
            print('Results saved!\n')
            battery.reset_history()


def main(input_grid=None):
    """
    Inputs:
        input_grid: (dict) A dictionary containing the following keys: 
            {'input_file':str, 
            'years':list, 
            'strategy':str, 
            'forecast':str, 
            'battery_degradation':str, 
            'pv_capacity':int, 
            'save_folder':str,
            'spot_multiplier':float}

            where each input are lists containing one or more values, described below. Simulations will be run with all possible combinations.
            input_file: (str) path to input file + the name of the input file until the year (e.g. "path\\L9__south_4kWp_")
            years: (list of ints) list containing the years which will be simulated
            strategy: (str) "NO-BATTERY", "SCM", "MPC", "MPC-BC", "MILP-YEAR", or "MILP-BC-YEAR"
            forecast: (str) "perfect" or "naive"
            battery_degradation: (str) "yes" or "no"
            pv_capacity: (int) size of the PV system in kWp, used for sizing the inverter (IN CURRENT CODE, THIS VALUE IS INDIFFERENT) 
            save_folder: (str) path to existing folder where all results are saved
            spot_multiplier: (float) spot price multiplier (e.g., if 2, spot prices are multiplied by 2) 
    """

    if input_grid:
        run_simulation_matrix(input_grid)
        return

    # Read file
    filename, filetype = read_file()

    #****************************************************************
    # If result file is given as input, only read and print result
    #****************************************************************
    if filetype == 'result':

        # Load simulation results
        print(f'Loading simulation results ...\n')
        simulation = Simulation(filename, filetype, forecast)

        # Print results
        simulation.print_results()
        return

    #****************************************************************
    # If tiletype is 'input', run the simulation
    #****************************************************************
    # Ask HEMS strategy
    while True:
        strategy = input("Select HEMS-strategy (NO-BATTERY, SCM, RULE-BC, MPC, MPC-BC): ")
        if strategy in ['NO-BATTERY', 'SCM', 'RULE-BC', 'MPC', 'MPC-BC']:
            break
        else:
            print(f'{strategy} not usable!')
    
    # Ask forecast method
    while True:
        forecast = input("Select forecast method (perfect, naive): ")
        if forecast in ['perfect', 'naive']:
            break
        else:
            print(f'{strategy} not usable!')
    
    # Initialize battery
    include_bat_degradation = input("Include battery degradation (yes/no): ")
    include_bat_degradation = include_bat_degradation == 'yes'  # Make this a boolean variable

    battery = Battery(include_bat_degradation)

    # Run simulation
    print('Preparing simulation ...\n')
    simulation = Simulation(filename, filetype, forecast)

    print('Running simulation ...\n')
    simulation.run_HEMS(battery, strategy)

    # Print results
    simulation.print_results()

    # Save results
    Tk().withdraw()
    save_filename = asksaveasfilename(filetypes=[("CSV files", "*.csv")])
    simulation.results.to_csv(save_filename, index=False)
    print('Results saved!')



if __name__ == '__main__':

    loads = ['L9', 'L11', 'L12', 'L13']

    for load in loads:
        input_folder = f'{load}'
        input_files = [os.path.join('data',input_folder,f'{load}_south_4kWp_')]
        years = [[2020, 2021, 2022]]
        strategies = ['NO-BATTERY', 'SCM', 'MPC', 'MPC-BC']
        battery_degradations = ['yes']
        #battery_degradations = ['no']
        forecasts = ['naive']
        pv_capacity = 4

        spot_multipliers = [1]
        result_folder = [os.path.join('Results', 'Final_simulations')]

        grid = [{'input_file':i_file, 'years':year, 'strategy':strat, 'forecast':forecast, 'battery_degradation':b_degr,
                'pv_capacity':pv_capacity, 'save_folder':s_folder, 'spot_multiplier':spot_multipl} \
                for i_file in input_files
                for strat in strategies
                for forecast in forecasts
                for b_degr in battery_degradations
                for s_folder in result_folder
                for spot_multipl in spot_multipliers
                for year in years]
        main(grid)
