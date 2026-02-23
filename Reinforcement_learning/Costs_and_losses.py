import torch
import numpy as np
import pandas as pd
from math import sqrt

#******************************************************************
# Constants related to battery usage cost
BAT_CAPEX = 9000
A = 1.2698*10**6
B = -1.3133
D = 2
E = -1.2
F = -0.0275
BATTERY_SIZE = 13.5 # [kWh]
DISCHARGE_POWER = 5
CHARGE_POWER = 5
RT_EFF = 0.9
CH_EFF = sqrt(RT_EFF)
DCH_EFF = sqrt(RT_EFF)
#******************************************************************
# Initialize simulation parameters
#************************************************
Nhours = 7*24


class Inverter:
    def __init__(self, rated_power):
        # ********************************************************
        # Inverter parameters
        # ********************************************************
        self.P_RATED = rated_power  # Rated power [kW]    

    def efficiency(self, input_power=None):
        """
        A function that gives the invertes efficiency with a given input power.
        """        
        return 0.97
    
    def convert(self, input_power):
        output_power =  input_power * self.efficiency(input_power)
        return output_power

# Initialize the inverter
inverter = Inverter(CHARGE_POWER)


def Interpret(recommendation, pv, load, battery_sh):
    """
    This function takes the output of the NN model and simulates the energy system based on that recommendation.

    Inputs:
        recommendation: (torch tensor) output of the NN model
        pv: (torch tensor) PV production
        load: (torch tensor) electricity load
        battery_sh: (torch tensor) battery charge
    
    Output:
        A tuple of arrays containing optimized actions
    """
    # Obtain charge and discharge recommendations from NN output 
    pv_to_bat_rec = CHARGE_POWER * torch.where(recommendation[:,0]>=0, recommendation[:,0], torch.zeros(recommendation[:,0].shape))
    bat_to_house_rec = DISCHARGE_POWER * torch.where(recommendation[:,0]<0, -recommendation[:,0], torch.zeros(recommendation[:,0].shape))
    # Battery charge at the recommendation time
    battery = battery_sh
    # Calculate battery charge for the next hour and check battery constraints
    # From battery to house
    bat_to_house_ex = torch.minimum(battery*DCH_EFF*inverter.efficiency(bat_to_house_rec), bat_to_house_rec)
    bat_to_house_in = bat_to_house_ex/(DCH_EFF*inverter.efficiency(bat_to_house_ex))  # More electricity is discharged than needed due to the efficiency 
    battery = battery - bat_to_house_in

    # From PV to battery
    pv_to_bat_ex = torch.minimum(
        torch.minimum(
            torch.maximum((BATTERY_SIZE-battery)/(CH_EFF*inverter.efficiency(pv_to_bat_rec)),torch.zeros(pv_to_bat_rec.shape, dtype=torch.float32)), 
                    pv), 
                        pv_to_bat_rec)
    pv_to_bat_in = pv_to_bat_ex*CH_EFF*inverter.efficiency(pv_to_bat_ex)
    battery = battery + pv_to_bat_in

    # PV to house, grid to house, and PV to grid, using the previously realized energy flows (PV->bat and bat->house)
    # PV to house is priorized over PV to grid in this model, although theoretically it might not provide 
    # economically optimal resuts (i.e. sell PV to grid and use battery charge during high prices)
    pv_to_house = torch.minimum(pv-pv_to_bat_ex, load)
    grid_to_house = torch.maximum(load - bat_to_house_ex - pv_to_house, torch.zeros(load.shape))
    pv_to_grid = torch.where(grid_to_house>0.0, torch.zeros(pv_to_house.shape), pv-pv_to_bat_ex-pv_to_house)
    
    pv_wasted = pv - pv_to_bat_ex-pv_to_house - pv_to_grid
    return (pv_to_house, pv_to_bat_ex, pv_to_grid,
            bat_to_house_ex, grid_to_house, pv_wasted, battery)



def Cost(recommendation, price_real, pv, load_real, battery_sh, bat_stor_val, MARGIN, FIXED_COST):
    """
    Cost function for RL model.

    Inputs:
        recommendation: (torch tensor) output of the NN model
        price_real: (torch tensor) electricity price
        pv: (torch tensor) PV production
        load_real: (torch tensor) electricity load
        battery_sh: (torch tensor) battery charge
        bat_stor_val: (torch tensor) value of stored energy (not used)
        MARGIN: (float) the margin added to electricity price
        FIXED_COST: (float) the fixed electricity price component (inc. VAT, TAX, and distribution fee)

    Output:
        A tuple consisting of the total cost and battery charge
    """
    p_purch = price_real*1.24 + MARGIN + FIXED_COST  # Electricity purchase price
    p_sell = price_real - MARGIN  # Electricity selling price

    # Obtain realized energies and battery charge for the next hour
    (pv_to_house_rec, pv_to_bat, pv_to_grid, 
            bat_to_house, grid_to_house, pv_wasted, battery) \
                = Interpret(recommendation, pv, load_real, battery_sh)
    
    # Wasted discharged energy to house 
    bat_to_house_wasted = torch.maximum(torch.zeros(load_real.shape), bat_to_house-load_real)
    
    costs = grid_to_house*p_purch
    profits = pv_to_grid*p_sell
    J = costs - profits
    
    return J, battery


def Cost_bat_cost(recommendation, price_real, pv, load_real, battery_sh, bat_stor_val, delta_SOC, operation_mode, MARGIN, FIXED_COST):
    """
    Cost function for RL-BC model.

    Inputs:
        recommendation: (torch tensor) output of the NN model
        price_real: (torch tensor) electricity price
        pv: (torch tensor) PV production
        load_real: (torch tensor) electricity load
        battery_sh: (torch tensor) battery charge
        bat_stor_val: (torch tensor) value of stored energy (not used)
        delta_SOC: (torch tensor) change in state-of-charge of the current half-cycle
        operation_mode: (int) either 0, 1, or -1. 0=idle, 1=charge, -1=discharge.
        MARGIN: (float) the margin added to electricity price
        FIXED_COST: (float) the fixed electricity price component (inc. VAT, TAX, and distribution fee)

    Output:
        A tuple consisting of the total cost, battery charge, change in battery SOC, and battery operation mode
    """

    p_purch = price_real*1.24 + MARGIN + FIXED_COST  # Electricity purchase price
    p_sell = price_real - MARGIN  # Electricity selling price

    # Obtain realized energies and battery charge for the next hour
    (pv_to_house_rec, pv_to_bat, pv_to_grid, 
            bat_to_house, grid_to_house, pv_wasted, battery) \
                = Interpret(recommendation, pv, load_real, battery_sh)
    
    # Float degradation
    float_AF = float_aging_factor(battery, BATTERY_SIZE)

    # Wasted discharged energy to house
    bat_to_house_wasted = torch.maximum(torch.zeros(load_real.shape), bat_to_house-load_real)
    costs = grid_to_house*p_purch
    profits = pv_to_grid*p_sell

    # Check if operation mode (charge, discharge) changed and update half-cycle history
    delta_SOC_new = delta_SOC.clone()
    cyclic_AF = torch.zeros_like(battery)
    
    # Battery is charged
    charging_condition = battery > battery_sh
    discharging_condition = battery_sh > battery

    # Reset delta SOC if operation mode chages
    delta_SOC_new = torch.where(charging_condition & (operation_mode == -1.), torch.zeros_like(delta_SOC), delta_SOC_new)
    delta_SOC_new = torch.where(discharging_condition & (operation_mode == 1.), torch.zeros_like(delta_SOC), delta_SOC_new)

    # Handle charging condition
    delta_SOC_new = torch.where(charging_condition, delta_SOC_new + (battery - battery_sh), delta_SOC_new)
    cyclic_AF = torch.where(charging_condition & (operation_mode == -1.), cyclic_aging_factor(delta_SOC, BATTERY_SIZE), cyclic_AF)
    aging_factor =  torch.maximum(cyclic_AF, float_AF)  # Degradation is the maximum of the two degradation mechanisms
    operation_mode = torch.where(charging_condition, torch.tensor(1., dtype=torch.float32), operation_mode)
        
    # Handle discharging condition
    #delta_SOC_new = torch.where(discharging_condition & (operation_mode == -1.), delta_SOC_new + (battery_prev - battery), delta_SOC_new)
    delta_SOC_new = torch.where(discharging_condition, delta_SOC_new + (battery_sh - battery), delta_SOC_new)
    cyclic_AF = torch.where(discharging_condition & (operation_mode == 1.), cyclic_aging_factor(delta_SOC, BATTERY_SIZE), cyclic_AF)
    aging_factor =  torch.maximum(cyclic_AF, float_AF)  # Degradation is the maximum of the two degradation mechanisms
    operation_mode = torch.where(discharging_condition, torch.tensor(-1., dtype=torch.float32), operation_mode)

    # Calculate battery usage cost
    delta_SOH = BATTERY_SIZE*aging_factor*0.2/(13.5*0.2)
    battery_usage_cost = delta_SOH * BAT_CAPEX * 100  # Battery price in cents
    J = costs + battery_usage_cost - profits

    return J, battery, delta_SOC_new, operation_mode


def __N_life(delta_SOC):
    """
    Function for estimating cycle life from DOD, based on
    https://www.researchgate.net/publication/293145570_CONCEPT_OF_A_BATTERY_AGING_MODEL_FOR_LITHIUM-ION_BATTERIES_CONSIDERING_THE_LIFETIME_DEPENDENCY_ON_THE_OPERATION_STRATEGY
    """
    N_max_soc = A * delta_SOC ** B
    return N_max_soc


def __aging_factor(delta_SOC):
    return 1/__N_life(delta_SOC*100+1E-20) / 2  # Divide by two since this is a half-cycle, not full cycle


def cyclic_aging_factor(delta_SOC, capacity):
    delta_SOC = delta_SOC/capacity
    aging_factor = __aging_factor(delta_SOC)
    return aging_factor


def __soc_degradation(charge, capacity):
    return 1/(D+E*torch.exp(F*(100-charge/capacity*100)))


def float_aging_factor(charge, capacity):
    soc_degr = __soc_degradation(charge, capacity) / (15*8760)  # Divide by the reference lifetime of 15 years
    return soc_degr


#*************************************************************************************************************
#************************* LOSS FUNCTIONS *******************************************************************
#*************************************************************************************************************


def Loss(x, spot, pv, load, battery_sh, bat_stor_val, model, MARGIN, FIXED_COST):
    loss = torch.tensor(np.zeros(x.shape[0]), requires_grad=True)
    battery_sh_i = battery_sh.T[0]
    battery_profit = torch.tensor(np.zeros(x.shape[0]), requires_grad=True)
    # Iterate through the following week
    for x_i, spot_i, pv_i, load_i, bat_stor_val_i in zip(x.swapaxes(0,1), spot.T,pv.T,load.T, bat_stor_val.T):
        #*************************************************************
        # Forward pass: Compute predicted y by passing x to the model
        #*************************************************************
        # Perform out-of-place operation to update the columns
        updated_soc = battery_sh_i.view(-1,1) / BATTERY_SIZE
        x_i_upd = torch.cat([x_i[:,:-1], updated_soc], dim=1)
        recommendation = model(x_i_upd)
        #*************************************************************
        # Compute loss
        #*************************************************************
        last_battery = battery_sh_i
        loss_i, battery_sh_i = Cost(recommendation, spot_i, pv_i, load_i, battery_sh_i, bat_stor_val_i, MARGIN, FIXED_COST)
        loss = loss + loss_i
    if torch.mean(bat_stor_val_i) == 0:
        pass
    else:
       # battery_profit = battery_profit + (battery_sh_i-last_battery)*(bat_stor_val_i/100+MARGIN/100+FIXED_COST/100)
        battery_profit = (battery_sh_i-battery_sh.T[0])*(bat_stor_val_i/100+MARGIN/100+FIXED_COST/100)
    loss =  loss / Nhours*24/100 - battery_profit/(Nhours/24) # To get cost per day 
   
    return torch.mean(loss)


def Loss_bat_cost(x, spot, pv, load, battery_sh, bat_stor_val, model, MARGIN, FIXED_COST):
    loss = torch.tensor(np.zeros(x.shape[0]), requires_grad=True)
    battery_sh_i = battery_sh.T[0]
    battery_profit = torch.tensor(np.zeros(x.shape[0]), requires_grad=True, dtype=torch.float32)
    operation_mode = torch.tensor(np.zeros(x.shape[0]), requires_grad=True, dtype=torch.float32)
    delta_SOC = torch.tensor(np.zeros(x.shape[0]), requires_grad=True, dtype=torch.float32)
    # Iterate through the following week
    for x_i, spot_i, pv_i, load_i, bat_stor_val_i in zip(x.swapaxes(0,1), spot.T, pv.T, load.T, bat_stor_val.T):
        #*************************************************************
        # Forward pass: Compute predicted y by passing x to the model
        #*************************************************************
        # Perform out-of-place operation to update the columns
        updated_soc = battery_sh_i.view(-1,1) / BATTERY_SIZE
        x_i_upd = torch.cat([x_i[:,:-1], updated_soc], dim=1)
        recommendation = model(x_i_upd)
        #*************************************************************
        # Compute loss
        #*************************************************************
        last_battery = battery_sh_i
        loss_i, battery_sh_i, delta_SOC, operation_mode = Cost_bat_cost(recommendation, spot_i, pv_i, load_i, battery_sh_i, 
                                             bat_stor_val_i, delta_SOC, operation_mode,  MARGIN, FIXED_COST)
        loss = loss + loss_i
    if torch.mean(bat_stor_val_i) == 0:
        pass
    else:
        battery_profit = (battery_sh_i-battery_sh.T[0])*(bat_stor_val_i/100+MARGIN/100+FIXED_COST/100)
    loss = loss / Nhours*24/100 - battery_profit/(Nhours/24) # To get cost per day 
   
    return torch.mean(loss)