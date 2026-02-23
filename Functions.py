import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
from math import sqrt
import os
from tkinter import Tk
from tkinter.filedialog import askdirectory, asksaveasfilename
import glob

# Battery sizing somewhat based on Tesla Powerwall
#*************************************************
BATTERY_SIZE_NAMEPLATE = 14 # [kWh] 
BATTERY_SIZE = 13.5 # Usable battery size
DISCHARGE_POWER = 5 # [kW]
CHARGE_POWER = 5 # [kW]
BATTERY_MIN_CHARGE = 0
#*************************************************

Nhours = 7*24

C = BATTERY_SIZE
L = 10000
h = 1.3
r = 0.3
MARGIN = 0.4 # [c/kWh]
# Transmission + electricity tax (inc. VAT) [2018, 2019, 2020, 2021, 2022, 2023]
FIXED_COSTS = [5.69, 5.90, 5.90, 5.44, 5.01, 5.01] # c/kWh

plt.rc('font', size=11)



def naive_forecast(df_original, var, random=0, interval=7*24):
    """
    Performs naive forecasting using a given forecast interval I, 
    where predicted variable L' on hour h is obtained by:
    L'(h) = L(h-I).
    This is used for load forecasts in e.g.
    Azualatam et al. (2019) https://doi.org/10.1016/j.rser.2019.06.007

    Inputs:
        df_original: (Pandas df) input Pandas dataframe, including datetime variable and the target variable
        var: (str) the name of the target variable (e.g. 'power') in the df
        random: (float [0,1]) the maximum value for random noise in the forecasted value
        interval: (int) the naive forecast interval I in hours
    
    Output:
        Pandas series of forecasted variables
    """

    df = df_original.copy()
    df['forecast'] = np.nan
    
    # Iterate trough df, starting from the first valid point (leave one interval out if the beginning)
    for i in range(interval, len(df)):
        ts = df.loc[i, 'datetime']
        current_var = df.loc[i, var]
        previous_var = df[df.datetime == (ts - datetime.timedelta(hours=interval))][var]
        if len(previous_var) == 0:
            previous_var = np.nan
        else:
            previous_var = float(previous_var.iloc[0])
        random_noise = np.random.uniform(low=0.0, high=random*current_var, size=1)
        
        df.loc[i, 'forecast'] = previous_var
        df.loc[i, 'pred_noise'] = previous_var + random_noise
    
    return df['forecast'].values


def read_data(path='data/Data_2019_in_UTC03.csv', first_month=1, last_month=12, return_datetime=False, multiply_data=10, 
              forecast='naive', pv_inverter=None, spot_multiplier=1):
    """
    Function for reading input data for energy management simulations.
    Inputs:
        path: (str) path to input data
        first_month: (int) the first month the data is selected
        last_month: (int) the last month the data is selected
        return_datetime: (boolean) if True, the function return includes datetime series
        multiply_data: (int) an integer describing how many times the data is expanded 
        forecast: (str) either 'naive' or 'perfect'
        pv_inverter: (Inverter class instance) not usable at the moment
        spot_multiplier: (float) spot price multiplier (e.g., if 2, spot prices are multiplied by 2)  

    Output:
        A tuple containing data input matrice for NN model, forecasted and real spot prices, production and load, and
        initialized battery charges for NN model

    """

    #****************************************************************
    # Load data
    #****************************************************************
    #path = os.path.join('data', file)
    data_x = pd.read_csv(path, parse_dates=['datetime'])
    data_x = data_x[(data_x.datetime.dt.month >= first_month) & (data_x.datetime.dt.month <= last_month)]
    data_x = data_x.reset_index()
    data_x = data_x.drop(columns=['index'])
    # Add inverter efficiency to PV power
    if pv_inverter is not None:
        data_x['power'] = pv_inverter.convert(data_x.power)
    #****************************************************************
    # Load and production forecasts
    #****************************************************************
    if forecast == 'naive':
        forecast_interval_pv = 24
        forecast_interval_load = 7*24 
        pv_forecast = naive_forecast(df_original=data_x, var='power', random=0., interval=forecast_interval_pv)
        load_forecast = naive_forecast(df_original=data_x, var='load', random=0., interval=forecast_interval_load)

    elif forecast == 'perfect':
        forecast_interval_pv = 24
        forecast_interval_load = 7*24
        pv_forecast = data_x.power.values
        load_forecast = data_x.load.values

    #****************************************************************
    # Multiply data to increase the size of input data (to have multiple various initial battery charges)
    # for machine learning training
    #****************************************************************
    forecast_hours = 24  # How many hours forward is the prediction made
    points_from_beginning = max(forecast_interval_pv, forecast_interval_load)  # Get rid of the one week without load forecasts 
    data_x = data_x[points_from_beginning:-forecast_hours]
    data_x = data_x.reset_index()
    data_x = data_x.drop(columns=['index'])

    # Save datetimes for later possible use
    datetime = data_x.datetime
    data_x = data_x.drop(columns=['datetime'])
    data_x = data_x.reset_index()
    data_x = data_x.drop(columns=['index'])

    # Drop columns
    if 'windSpeed' in data_x.columns:
        data_x = data_x.drop(columns=['windSpeed'])
    if 'RH' in data_x.columns:
        data_x = data_x.drop(columns=['RH'])
    if 'TEMP' in data_x.columns:
        data_x = data_x.drop(columns=['TEMP'])
    if 'GHI' in data_x.columns:
        data_x = data_x.drop(columns=['GHI'])
    if 'dayofyear' in data_x.columns:
        data_x = data_x.drop(columns=['dayofyear'])
    if 'dayofweek' in data_x.columns:
        data_x = data_x.drop(columns=['dayofweek'])
    if 'hour' in data_x.columns:
        data_x = data_x.drop(columns=['hour'])

    data_x['spot'] = data_x.spot / 10 *spot_multiplier # Change EUR/MWh to c/kWh and multiply by the scaler
    for i in range(1, 25):  # Columns 1-25 correspond to day ahead prices
        data_x['{}'.format(i)] /= 10 * spot_multiplier

    #****************************************************************
    # Known near future spot pirces
    #****************************************************************
    future_spots = data_x[[str(i) for i in range(1, 25)]].to_numpy()
    
    # Create separate input features for purhcase and sell price
    FIXED_COST = FIXED_COSTS[datetime.dt.year.iloc[0]-2018]
    data_x['spot_purch'] = data_x.spot*1.24 + MARGIN + FIXED_COST
    data_x['spot_sell'] = data_x.spot - MARGIN

    #****************************************************************
    # Data for performance evaluation
    #****************************************************************
    spot   = np.array([data_x.loc[ind,'spot'] for ind in range(data_x.shape[0])])
    power   = np.array([data_x.loc[ind, 'power'] for ind in range(data_x.shape[0])])
    load = np.array([data_x.loc[ind, 'load'] for ind in range(data_x.shape[0])])
    datetime = datetime.values

    pv_forecast = pv_forecast[points_from_beginning:]
    load_forecast = load_forecast[points_from_beginning:]

    #****************************************************************
    # Add next 24h power and load forecast to input data matrice
    #****************************************************************
    power_next24h = np.array([pv_forecast[ind+1:ind+(1+forecast_hours)] for ind 
                              in range(len(pv_forecast)-forecast_hours)])
    load_next24h = np.array([load_forecast[ind+1:ind+(1+forecast_hours)] for ind 
                              in range(len(load_forecast)-forecast_hours)])
    power_next24h = np.tile(power_next24h, (multiply_data,1))  # Multiply data
    load_next24h = np.tile(load_next24h, (multiply_data, 1))

    #****************************************************************
    # Change dataframe to numpy array and remove first and last datapoints
    #****************************************************************
    data_x = pd.concat([data_x for _ in range(multiply_data)])
    data_x = data_x.to_numpy()

    # Multiply other data
    spot = np.tile(spot, multiply_data)
    power = np.tile(power, multiply_data)
    load = np.tile(load, multiply_data)
    datetime = np.tile(datetime, multiply_data)

    pv_forecast = pv_forecast[:-forecast_hours]
    load_forecast = load_forecast[:-forecast_hours]
    pv_forecast = np.tile(pv_forecast, multiply_data)
    load_forecast = np.tile(load_forecast, multiply_data)

    #****************************************************************
    # Initilaize battery charge 
    #****************************************************************
    np.random.seed(42) # Use a fixed seed to get reproducable results
    bat_ch = np.random.rand(data_x.shape[0])*BATTERY_SIZE
    data_x  = np.hstack((data_x, 
                         power_next24h.reshape(data_x.shape[0], forecast_hours),
                         load_next24h.reshape(data_x.shape[0], forecast_hours),
                         bat_ch.reshape(data_x.shape[0],1)))

    if return_datetime:
        return data_x, spot, future_spots, power, pv_forecast, load, load_forecast, bat_ch, datetime
    return data_x, future_spots, spot, power, pv_forecast, load, load_forecast, bat_ch


#*******************************************************************************************************************************************************************************************************
# FUNCTIONS FOR PROCESSING SIMULATION RESULTS
#*******************************************************************************************************************************************************************************************************

def compile_results(folder):
    """
    Function for compiling multiple simulation results into one csv file.

    Inputs
        folder: (str) a folder containing the simulation results (csv) to be compiled to one csv file.
                Each simulation result csv's name should be formatted as 
                '[method]_[degr]_[load]_[pv_capacity]_[year]_UTC03.csv' 
    """

    # Read filenames from given folder
    if folder is None:
        Tk().withdraw()
        folder = askdirectory()
    if folder == '':
        return
    all_files = glob.glob(os.path.join(folder, "*.csv"))
    print(f'{len(all_files)} in {folder}')
    # Initialize dataframe
    results_df = pd.DataFrame()
    
    # Read files
    for filepath in all_files:
        drive, path_and_file = os.path.splitdrive(filepath)
        path, filename = os.path.split(path_and_file)
        results = pd.read_csv(filepath, parse_dates=['datetime'])

        # Chop the filename to obtain simulation information
        parts = filename.split('_')
        strategy = parts[0]
        forecast = parts[1]
        degradation = parts[2]
        load_profile = parts[3]
        pv_profile = parts[4]
        pv_capacity = parts[5]
        year = int(parts[6])

        # Calculate the annual results
        self_consumed = (results.pv_to_house + results.pv_to_bat).sum()
        #load = (results.pv_to_house + results.bat_to_house + results.grid_to_house).sum()
        load = results.load.sum()
        #prod = (results.pv_to_house + results.pv_to_bat + results.pv_to_grid).sum()
        prod = results.power.sum()
        scr = self_consumed/prod
        ssr = self_consumed/load
        el_bill = (results.cost.sum() - results.profit.sum())
        bat_usage_cost = results.battery_usage_cost.sum()
        total_cost = el_bill + bat_usage_cost
        wasted_energy = (results.pv_to_bat_grid_wasted + results.to_house_wasted).sum()
        battery_capacity = results.battery_capacity.max()
        capacity_fade = battery_capacity - results.battery_capacity.min()
        delta_SOH = results.battery_deltaSOH.sum()
        delta_SOH_float = results.deltaSOH_float.sum()
        delta_SOH_cyc = results.deltaSOH_cyclic.sum()
        aging_factor = results.battery_aging_factor.sum()
        aging_factor_float = results.aging_factor_float.sum() 
        aging_factor_cyc = results.aging_factor_cyclic.sum()
        
        # Create a new row to be added to the dataframe
        new_row = {'strategy':strategy, 'forecast':forecast, 'year':year, 'load_profile':load_profile, 'load_tot':load, 'pv_profile':pv_profile, 'pv_capacity':pv_capacity,
                   'pv_tot':prod, 'bat_capacity':battery_capacity, 'degradation':degradation, 'bat_deg_kWh':capacity_fade, 
                   'delta_SOH':delta_SOH, 'delta_SOH_float':delta_SOH_float, 'delta_SOH_cyc':delta_SOH_cyc, 
                   'aging_factor':aging_factor, 'aging_factor_float':aging_factor_float, 'aging_factor_cyc':aging_factor_cyc,
                   'scr':scr, 'ssr':ssr, 'wasted_energy':wasted_energy, 'el_bill':el_bill, 'bat_usage_cost':bat_usage_cost,
                   'total_cost':total_cost}
        
        results_df = results_df._append(new_row, ignore_index=True)

    # Save results
    Tk().withdraw()
    save_filename = asksaveasfilename(filetypes=[("CSV files", "*.csv")])
    results_df.to_csv(save_filename, index=False)
    print('Results saved!')


