import numpy as np
import sys
path_to_gurobi = r'C:\gurobi1202\win64\bin\gurobi_cl.exe'
from gurobipy import Model, GRB, quicksum


def no_battery(pv, load, spot, MARGIN, FIXED_COST, battery):
    """
    Energy management strategy without battery, i.e., excess PV production is sold to grid and uncovered load is 
    purchased from grid.

    Inputs:
        pv: (np array) PV production (in kW)
        load: (np array) electricity consumption (in kW)
        spot: (np array) electricity spot price (in EUR/kWh)
        MARGIN: (float) the margin added to electricity price
        FIXED_COST: (float) the fixed electricity price component (inc. VAT, TAX, and distribution fee)
        battery: instance of 'Battery' class

    Output: 
        A tuple of an array containing performed actions and total costs as float
    """
    costs = 0
    results = np.zeros(11)  # Initialize result array
    battery.ch = 0  # Set battery charge to 0
    battery.C = 0  # Set battery capacity to 0
    for h in range(len(pv)):
        p_from_grid = p_to_grid = p_bat_dch = p_bat_ch  = 0
        p_res = pv[h] - load[h]
        if p_res > 0:  # PV power greater than consumption
            p_to_grid = abs(p_res)
        elif p_res < 0: # Less production than consumption
            p_from_grid = abs(p_res)
        else:  # Production = consumption
            p_from_grid = p_to_grid = p_bat_dch = p_bat_ch = 0

        battery.update(charge_power=p_bat_ch, discharge_power=p_bat_dch, include_float_degradation=False)

        profit = p_to_grid*(spot[h] - MARGIN) / 100  # Convert to euros
        cost = p_from_grid*(spot[h]*1.24 + MARGIN + FIXED_COST) / 100  # Convert to euros
        results = np.vstack([results, 
                             np.array([min(pv[h], load[h]), 
                                       p_bat_ch, p_bat_dch, 
                                       p_from_grid, 
                                       p_to_grid,
                                       0.0,  # PV wasted
                                       0.0,  # to house wasted
                                       battery.ch,
                                       cost,
                                       profit,
                                       0.0  # DeltaSOH
                                       ])])
        
        costs += (cost - profit)
    results = results[1:,:]  # Get rid of the first row of zeros
    return results, costs


def SCM(pv, load, spot, MARGIN, FIXED_COST, battery):
    """
    Rule-based energy management strategy based on self-consumption maximization (SCM).

    Inputs:
        pv: (np array) PV production (in kW)
        load: (np array) electricity consumption (in kW)
        spot: (np array) electricity spot price (in EUR/kWh)
        MARGIN: (float) the margin added to electricity price
        FIXED_COST: (float) the fixed electricity price component (inc. VAT, TAX, and distribution fee)
        battery: instance of 'Battery' class

    Output: 
        A tuple of an array containing performed actions and total costs as float
    """
    costs = 0
    results = np.zeros(11)  # Initialize result array
    battery.ch = 0  # Set battery charge to 0
    for h in range(len(pv)):
        p_from_grid = p_to_grid = p_bat_dch = p_bat_ch  = 0
        p_res = pv[h] - load[h]

        if p_res > 0:  # PV power greater than consumption
            e_bat_res = battery.C - battery.ch
            if e_bat_res > 0:
                # Cap charging power using max charge power, excess power, and battery charge
                p_bat_ch = min(battery.MAX_CH_POWER, 
                               min(abs(p_res), e_bat_res/battery.eff(power=battery.MAX_CH_POWER)))
                p_to_grid = abs(p_res) - p_bat_ch  # Export remainder
            else:
                # If battery full, sell all to grid
                p_to_grid = abs(p_res)
        elif p_res < 0: # Less production than consumption
            e_bat_res = battery.ch - battery.MIN_CHARGE
            if battery.ch == battery.MIN_CHARGE:  # Battery is empty
                p_from_grid = abs(p_res)
            else: # Battery not empty
                # Cap discharge power
                p_bat_dch = min(battery.MAX_DCH_POWER, 
                                min(abs(p_res), e_bat_res*battery.eff(power=battery.MAX_DCH_POWER)))
                p_from_grid = abs(p_res) - p_bat_dch  # Import remainder
        else:  # Production = consumption
            p_from_grid = p_to_grid = p_bat_dch = p_bat_ch = 0
        
        # Update battery charge, efficiency considered inside the update function
        start_capacity = battery.C
        battery.update(charge_power=p_bat_ch, discharge_power=p_bat_dch)
        end_capacity = battery.C
        delta_SOC = end_capacity - start_capacity

        profit = p_to_grid*(spot[h] - MARGIN) / 100  # Convert to euros
        cost = p_from_grid*(spot[h]*1.24 + MARGIN + FIXED_COST) / 100  # Convert to euros
        results = np.vstack([results, 
                             np.array([min(pv[h], load[h]), 
                                       p_bat_ch, p_bat_dch, 
                                       p_from_grid, 
                                       p_to_grid,
                                       0.0,  # PV wasted
                                       0.0,  # to house wasted
                                       battery.ch,
                                       cost,
                                       profit,
                                       delta_SOC])])
        
        
        costs += (cost - profit)
    results = results[1:,:]  # Get rid of the first row of zeros
    return results, costs

    

def MILP(pv_power, pv_forecast, consumption, load_forecast, electricity_price, battery, MARGIN, FIXED_COST):
    """
    Multi-integer linear programming for optimizing energy management by minimizing the electricity bill
    (exlcuding battery costs) using Gurboi.

    Inputs:
        pv_power: (np array) production for the given optimization horizon (in kW)
        pv_forecast: (np array) forecasted production for the given optimization horizon (in kW)
        consumption: (np array) consumption for the given optimization horizon (in kW)
        load_forecast: (np array) forecasted consumption for the given optimization horizon (in kW)
        electricity_price: (np array) electricity spot price for the given optimization horizon (in EUR/kWh)
        battery: instance of 'Battery' class
        MARGIN: (float) the margin added to electricity price
        FIXED_COST: (float) the fixed electricity price component (inc. VAT, TAX, and distribution fee)

    Output: 
        A tuple of arrays containing optimized actions for the given optimization horizon
    """
    # Create the optimization problem
    model = Model("BatteryScheduling")
    model.setParam("OutputFlag", 0)  # Suppress solver output
    model.setParam("Threads", 8)
    model.setParam("MIPGap", 0.05)
    model.setParam("MIPFocus", 3)

    # Change the first forecasted power/load, since that corresponds the current hour,
    # which is known in the given moment
    pv_forecast[0] = pv_power[0]
    load_forecast[0] = consumption[0]

    MAX_CHARGE_POWER = battery.MAX_CH_POWER
    MAX_DISCHARGE_POWER = battery.MAX_DCH_POWER
    BAT_CAPACITY = battery.C
    CH_EFF = battery.eff(0)   # THIS SHOULD BE FIXED TO ACCOUNT NON-LINEAR BEHAVIOUR
    DCH_EFF = battery.eff(0)   # THIS SHOULD BE FIXED TO ACCOUNT NON-LINEAR BEHAVIOUR

    # Decision variables
    charge = model.addVars(len(pv_power), lb=0., ub=MAX_CHARGE_POWER, name="Charge")
    discharge = model.addVars(len(pv_power), lb=0., ub=MAX_DISCHARGE_POWER, name="Dicharge")
    grid_buy = model.addVars(len(pv_power), lb=0., ub=load_forecast.max(), name="GridBuy")
    grid_sell = model.addVars(len(pv_power), lb=0., ub=pv_forecast.max(), name="GridSell")
    self_consumption = model.addVars(len(pv_power), lb=0., ub=max(load_forecast.max(), pv_forecast.max()), name="SelfConsumption")
    battery_charge = model.addVars(len(pv_power) + 1, lb=0., ub=BAT_CAPACITY, name="BatteryCharge")

    # Binary variable to prevent simultaneous charging and discharging
    is_charging = model.addVars(len(pv_power), vtype=GRB.BINARY, name="IsCharging")  # If 1, then in charging mode, if 0, then in discharge mode
    is_idle = model.addVars(len(pv_power), vtype=GRB.BINARY, name="IsIdle")
    
    # Binary variable to prevent simultaneous selling and buying
    is_buying = model.addVars(len(pv_power), vtype=GRB.BINARY, name="IsBuying")

    # Objective function    
    objective_expr = (
        quicksum((electricity_price[t]*1.24 + MARGIN + FIXED_COST) * grid_buy[t] for t in range(len(pv_power))) -
        quicksum((electricity_price[t] - MARGIN) * grid_sell[t] for t in range(len(pv_power)))
    )

    model.setObjective(objective_expr, GRB.MINIMIZE)

    # Constraints
    model.addConstr(battery_charge[0] == battery.ch)  # Fix the first (=current) battery charge value

    # Formulate the constraints
    for t in range(len(pv_power)):
        
        model.addConstr(self_consumption[t] <= pv_forecast[t])
        model.addConstr(self_consumption[t] <= load_forecast[t])
        model.addConstr(battery_charge[t+1] == battery_charge[t] + charge[t] * CH_EFF - discharge[t] * (1/DCH_EFF))

        # Charge constraints
        model.addConstr(charge[t] <= (BAT_CAPACITY - battery_charge[t]) * (1/CH_EFF))
        model.addConstr(charge[t] <= pv_forecast[t] - self_consumption[t])
        # Discharge constraints
        model.addConstr(discharge[t] <= battery_charge[t] * DCH_EFF)
        model.addConstr(discharge[t] <= load_forecast[t] - self_consumption[t])

        # Prevent simultaneous charging and discharging
        M = 500
        model.addConstr(charge[t] <= M * is_charging[t])
        model.addConstr(discharge[t] <= M * (1-is_charging[t]))

        model.addConstr(charge[t] <= M * (1 - is_idle[t]))
        model.addConstr(discharge[t] <= M * (1 - is_idle[t]))
        # If charge[t] > 0 or discharge[t] > 0, then is_idle[t] == 0
        model.addConstr(charge[t] + discharge[t] >= 1e-3 * (1 - is_idle[t]))

        # Prevent simultaneous buying and selling
        M = 500  # Big-M constant

        model.addConstr(grid_buy[t] <= M*is_buying[t])
        model.addConstr(grid_sell[t] <= M*(1-is_buying[t]))

        model.addConstr(grid_buy[t] >= load_forecast[t] - self_consumption[t] - discharge[t])
        model.addConstr(grid_sell[t] <= pv_forecast[t] - self_consumption[t] - charge[t])

    #***********Solve optimization****************************************************************************************
    # Solve optimization and convert variables to arrays 
    model.optimize()
    if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL:
        pass
    else:
        print('Infeasible')
    charge = np.array([charge[i].X for i in range(len(charge))])
    discharge = np.array([discharge[i].X for i in range(len(discharge))])
    grid_buy = np.array([grid_buy[i].X for i in range(len(grid_buy))])
    grid_sell = np.array([grid_sell[i].X for i in range(len(grid_sell))])
    self_consumption = np.array([self_consumption[i].X for i in range(len(self_consumption))])
    battery_charge = np.array([battery_charge[i].X for i in range(len(battery_charge))])
    
    return charge, discharge, grid_buy, grid_sell, self_consumption, battery_charge, np.zeros(len(charge))


def MILP_batcost(pv_power, pv_forecast, consumption, load_forecast, electricity_price, battery, MARGIN, FIXED_COST, 
                 previous_start_charge, previous_charge_bool, previous_start_discharge, previous_discharge_bool, previous_values):
    """
    Multi-integer linear programming for optimizing energy management by minimizing the electricity bill
    and battery costs using Gurobi. Battery degradation based on 
    https://www.researchgate.net/publication/293145570_CONCEPT_OF_A_BATTERY_AGING_MODEL_FOR_LITHIUM-ION_BATTERIES_CONSIDERING_THE_LIFETIME_DEPENDENCY_ON_THE_OPERATION_STRATEGY

    Inputs:
        pv_power: (np array) production for the given optimization horizon (in kW)
        pv_forecast: (np array) forecasted production for the given optimization horizon (in kW)
        consumption: (np array) consumption for the given optimization horizon (in kW)
        load_forecast: (np array) forecasted consumption for the given optimization horizon (in kW)
        electricity_price: (np array) electricity spot price for the given optimization horizon (in EUR/kWh)
        battery: instance of 'Battery' class
        MARGIN: (float) the margin added to electricity price
        FIXED_COST: (float) the fixed electricity price component (inc. VAT, TAX, and distribution fee)
        previous_start_charge: (float) battery charge in the last step if charging
        previous_charge_bool: (int) 1 if last step battery was charging, otherwise 0
        previous_start_discharge: (float) battery charge in the last step if discharging
        previous_discharge_bool: (int) 1 if last step battery was discharging, otherwise 0
        previous_values: (dict) variable values obtained during last iteration for warm start

    Output: 
        A tuple of arrays containing optimized actions for the given optimization horizon
    """
    # Create the optimization problem
    model = Model("BatteryScheduling")
    model.setParam("OutputFlag", 0)  # Suppress solver output
    model.setParam("Threads", 8)
    model.setParam("MIPGap", 0.05)
    model.setParam("MIPFocus", 3)

    # Change the first forecasted power/load, since that corresponds the current hour,
    # which is known in the given moment
    pv_forecast[0] = pv_power[0]
    load_forecast[0] = consumption[0]

    MAX_CHARGE_POWER = battery.MAX_CH_POWER
    MAX_DISCHARGE_POWER = battery.MAX_DCH_POWER
    BAT_CAPACITY = battery.C
    CH_EFF = battery.eff(0)
    DCH_EFF = battery.eff(0)
    BATTERY_LIFTIME = battery.LIFETIME
    EOL = battery.END_OF_LIFE

    # Degradation curve parameters
    CAPEX = battery.CAPEX
    a00 = battery.cyclic_degradation_curve_lin_slopes[0]
    a01 = battery.cyclic_degradation_curve_lin_slopes[1]
    a02 = battery.cyclic_degradation_curve_lin_slopes[2]
    a03 = battery.cyclic_degradation_curve_lin_slopes[3]
    b00 = battery.cyclic_degradation_curve_lin_inters[0]
    b01 = battery.cyclic_degradation_curve_lin_inters[1]
    b02 = battery.cyclic_degradation_curve_lin_inters[2]
    b03 = battery.cyclic_degradation_curve_lin_inters[3]

    a10 = battery.float_degradation_curve_lin_slopes[0]
    a11 = battery.float_degradation_curve_lin_slopes[1]
    a12 = battery.float_degradation_curve_lin_slopes[2]
    a13 = battery.float_degradation_curve_lin_slopes[3]
    b10 = battery.float_degradation_curve_lin_inters[0]
    b11 = battery.float_degradation_curve_lin_inters[1]
    b12 = battery.float_degradation_curve_lin_inters[2]
    b13 = battery.float_degradation_curve_lin_inters[3]

    # Initialize breakpoints for SOS2
    cyclic_breakpoints = [i * BAT_CAPACITY / 100 for i in [0, 10, 25, 60, 100]]
    cyclic_values = [
        a00 * 0 + b00,
        a01 * 10 + b01,
        a02 * 25 + b02,
        a03 * 60 + b03,
        a03 * 100 + b03
    ]

    float_breakpoints = [i * BAT_CAPACITY / 100 for i in [0, 25, 60, 85, 100]]
    float_values = [
        a10 * 0 + b10,
        a11 * 25 + b11,
        a12 * 60 + b12,
        a13 * 85 + b13,
        a13 * 100 + b13
    ]

    # Decision variables
    charge = model.addVars(len(pv_power), lb=0., ub=MAX_CHARGE_POWER, name="Charge")
    discharge = model.addVars(len(pv_power), lb=0., ub=MAX_DISCHARGE_POWER, name="Dicharge")
    grid_buy = model.addVars(len(pv_power), lb=0., ub=load_forecast.max(), name="GridBuy")
    grid_sell = model.addVars(len(pv_power), lb=0., ub=pv_forecast.max(), name="GridSell")
    self_consumption = model.addVars(len(pv_power), lb=0., ub=max(load_forecast.max(), pv_forecast.max()), name="SelfConsumption")
    battery_charge = model.addVars(len(pv_power) + 1, lb=0., ub=BAT_CAPACITY, name="BatteryCharge")

    # Binary variable to prevent simultaneous charging and discharging
    is_charging = model.addVars(len(pv_power), vtype=GRB.BINARY, name="IsCharging")  # If 1, then in charging mode, if 0, then in discharge mode
    is_idle = model.addVars(len(pv_power), vtype=GRB.BINARY, name="IsIdle")
    
    # Binary variable to prevent simultaneous selling and buying
    is_buying = model.addVars(len(pv_power), vtype=GRB.BINARY, name="IsBuying")

    # Degradation cost variables
    delta_SOC = model.addVars(len(pv_power), lb=0., ub=BAT_CAPACITY, name="DeltaSOC")
    cyclic_aging_factor = model.addVars(len(pv_power), lb=0., ub=1., name="Cyclic_AF")
    float_aging_factor = model.addVars(len(pv_power), lb=0., ub=1., name="Float_AF")
    total_aging_factor = model.addVars(len(pv_power), lb=0., ub=1., name="Total_AF")
    delta_SOH = model.addVars(len(pv_power), lb=0., ub=1., name="Delta_SOH")

    # Binary variables to indicate the start of a new charging/discharging period
    start_charge_bool = model.addVars(len(pv_power), vtype=GRB.BINARY, name="StartChargeBool")
    start_discharge_bool = model.addVars(len(pv_power), vtype=GRB.BINARY, name="StartDischargeBool")
    # Binary variables to indicate the end of a charging/discharging period
    end_charge_bool = model.addVars(len(pv_power), vtype=GRB.BINARY, name="EndChargeBool")
    end_discharge_bool = model.addVars(len(pv_power), vtype=GRB.BINARY, name="EndDischargeBool")

    # End-point battery charges for calculating the half-cycle delta_SOC
    start_charge = model.addVars(len(pv_power), lb=0., ub=BAT_CAPACITY, name="StartCharge")
    start_discharge = model.addVars(len(pv_power), lb=0., ub=BAT_CAPACITY, name="StartDischarge")
    
    # Objective function    
    objective_expr = (
        quicksum((electricity_price[t]*1.24 + MARGIN + FIXED_COST) * grid_buy[t] for t in range(len(pv_power))) -
        quicksum((electricity_price[t] - MARGIN) * grid_sell[t] for t in range(len(pv_power))) +
        quicksum(delta_SOH[t] * (CAPEX * 100) for t in range(len(pv_power)))
    )

    model.setObjective(objective_expr, GRB.MINIMIZE)

    # Constraints
    model.addConstr(battery_charge[0] == battery.ch)  # Fix the first (=current) battery charge value

    # Initialize variables for warm start
    if previous_values is not None:
        for t in range(len(pv_power)):
            charge[t].Start = previous_values["charge"][t]
            discharge[t].Start = previous_values["discharge"][t]
            battery_charge[t].Start = previous_values["battery_charge"][t]
            grid_buy[t].Start = previous_values["grid_buy"][t]
            grid_sell[t].Start = previous_values["grid_sell"][t]
            self_consumption[t].Start = previous_values["self_consumption"][t]
            delta_SOH[t].Start = previous_values["delta_SOH"][t]
            is_charging[t].Start = previous_values["is_charging"][t]
            is_buying[t].Start = previous_values["is_buying"][t]
            delta_SOC[t].Start = previous_values["delta_SOC"][t]
            cyclic_aging_factor[t].Start = previous_values["cyclic_aging_factor"][t]
            float_aging_factor[t].Start = previous_values["float_aging_factor"][t]
            total_aging_factor[t].Start = previous_values["total_aging_factor"][t]
            start_charge_bool[t].Start = previous_values["start_charge_bool"][t]
            start_discharge_bool[t].Start = previous_values["start_discharge_bool"][t]
            end_charge_bool[t].Start = previous_values["end_charge_bool"][t]
            end_discharge_bool[t].Start = previous_values["end_discharge_bool"][t]
            start_charge[t].Start = previous_values["start_charge"][t]
            start_discharge[t].Start = previous_values["start_discharge"][t]

    # Formulate the constraints
    for t in range(len(pv_power)):
        
        model.addConstr(self_consumption[t] <= pv_forecast[t])
        model.addConstr(self_consumption[t] <= load_forecast[t])
        model.addConstr(battery_charge[t+1] == battery_charge[t] + charge[t] * CH_EFF - discharge[t] * (1/DCH_EFF))

        # Charge constraints
        model.addConstr(charge[t] <= (BAT_CAPACITY - battery_charge[t]) * (1/CH_EFF))
        model.addConstr(charge[t] <= pv_forecast[t] - self_consumption[t])
        # Discharge constraints
        model.addConstr(discharge[t] <= battery_charge[t] * DCH_EFF)
        model.addConstr(discharge[t] <= load_forecast[t] - self_consumption[t])

        # Prevent simultaneous charging and discharging
        M = 500
        model.addConstr(charge[t] <= M * is_charging[t])
        model.addConstr(discharge[t] <= M * (1-is_charging[t]))
        model.addConstr(charge[t] <= M * (1 - is_idle[t]))
        model.addConstr(discharge[t] <= M * (1 - is_idle[t]))

        # If charge[t] > 0 or discharge[t] > 0, then is_idle[t] == 0
        model.addConstr(charge[t] + discharge[t] >= 1e-3 * (1 - is_idle[t]))

        # Auxillary variables for detecting change-points
        if t >= (len(pv_power)-1):
            model.addConstr(start_charge_bool[t] == 0)
            model.addConstr(start_discharge_bool[t] == 0)
        elif t > 0:
            model.addConstr(is_charging[t] - is_charging[t-1] <= 1 - is_idle[t])
            model.addConstr(is_charging[t-1] - is_charging[t] <= 1 - is_idle[t])

            model.addConstr(start_charge_bool[t] >= is_charging[t] - is_charging[t-1])
            model.addConstr(start_charge_bool[t] <= is_charging[t] - is_charging[t-1] + 1)
            
            model.addConstr(start_discharge_bool[t] >= is_charging[t-1] - is_charging[t])
            model.addConstr(start_discharge_bool[t] <= is_charging[t-1] - is_charging[t] + 1)

            model.addConstr(end_charge_bool[t] >= is_charging[t-1] - is_charging[t])
            model.addConstr(end_charge_bool[t] <= is_charging[t-1] - is_charging[t] + 1)

            model.addConstr(end_discharge_bool[t] >= is_charging[t] - is_charging[t-1])
            model.addConstr(end_discharge_bool[t] <= is_charging[t] - is_charging[t-1] + 1)

            model.addConstr(start_charge_bool[t] <= 1 - is_idle[t])
            model.addConstr(start_discharge_bool[t] <= 1 - is_idle[t])
            model.addConstr(end_charge_bool[t] <= 1 - is_idle[t])
            model.addConstr(end_discharge_bool[t] <= 1 - is_idle[t])

        else:  # t==0
            model.addConstr(start_charge_bool[t] == is_charging[t])
            model.addConstr(start_discharge_bool[t] == (1-is_charging[t]))

            model.addConstr(end_charge_bool[t] <= (1-is_charging[t]))
            model.addConstr(end_charge_bool[t] <= previous_charge_bool)
            model.addConstr(end_charge_bool[t] >= (1-is_charging[t]) + previous_charge_bool - 1)

            model.addConstr(end_discharge_bool[t] <= is_charging[t])
            model.addConstr(end_discharge_bool[t] <= previous_discharge_bool)
            model.addConstr(end_discharge_bool[t] >= is_charging[t] + previous_discharge_bool - 1)

        if t>0:
            # Change-point battery charges
            model.addConstr(start_charge[t] <= battery_charge[t] + M * (1 - start_charge_bool[t]))
            model.addConstr(start_charge[t] >= battery_charge[t] - M * (1 - start_charge_bool[t]))
            model.addConstr(start_discharge[t] <= battery_charge[t] + M * (1 - start_discharge_bool[t]))
            model.addConstr(start_discharge[t] >= battery_charge[t] - M * (1 - start_discharge_bool[t]))

            # Move start-point charge value to the end-point
            model.addConstr(start_charge[t] <= start_charge[t-1] + M * (1 - is_charging[t-1]))
            model.addConstr(start_charge[t] >= start_charge[t-1] - M * (1 - is_charging[t-1]))
            model.addConstr(start_discharge[t] <= start_discharge[t-1] + M * (is_charging[t-1]))
            model.addConstr(start_discharge[t] >= start_discharge[t-1] - M * (is_charging[t-1]))
        else: # t==0
            # Use previous iterations' realized cycle starting charge and boolean indicating charge or dicharge 
            model.addConstr(start_charge[t] == previous_start_charge * previous_charge_bool)
            model.addConstr(start_discharge[t] == previous_start_discharge * previous_discharge_bool)
                     
        # Prevent simultaneous buying and selling
        M = 500  # Big-M constant
        model.addConstr(grid_buy[t] <= M*is_buying[t])
        model.addConstr(grid_sell[t] <= M*(1-is_buying[t]))
        model.addConstr(grid_buy[t] >= load_forecast[t] - self_consumption[t] - discharge[t])
        model.addConstr(grid_sell[t] <= pv_forecast[t] - self_consumption[t] - charge[t])

        # Calculate delta SOC
        if t >= (len(pv_power)-1):
            # Charging
            model.addConstr(delta_SOC[t] >= (battery_charge[t] - start_charge[t]))   
            # Discharging
            model.addConstr(delta_SOC[t] >= (start_discharge[t] - battery_charge[t]))
        else:
            # Charging
            model.addConstr(delta_SOC[t] >= (battery_charge[t] - start_charge[t]) - M * (1 - end_charge_bool[t]))
            # Discharging
            model.addConstr(delta_SOC[t] >= (start_discharge[t] - battery_charge[t]) - M * (1 - end_discharge_bool[t]))
            
        #***********Cyclig degradation ****************************************************************************************
        # The one value within one of the ranges corresponds to the delta SOC

        # Cyclic degradation
        model.addGenConstrPWL(
            delta_SOC[t],  # input variable (x)
            cyclic_aging_factor[t],         # output variable (y)
            cyclic_breakpoints,             # x breakpoints
            cyclic_values,                  # y values
            "CyclicDegradation_{}".format(t)
        )

        #***********Float degradation ****************************************************************************************
        # The one value within one of the ranges corresponds to the delta SOC
        model.addGenConstrPWL(
            battery_charge[t],          # input variable (x)
            float_aging_factor[t],      # output variable (y)
            float_breakpoints,          # x breakpoints
            float_values,               # y values
            "FloatDegradation_{}".format(t)
        )

        #***********Total degradation ****************************************************************************************
        model.addConstr(total_aging_factor[t] >= float_aging_factor[t] / (BATTERY_LIFTIME * 8760))
        model.addConstr(total_aging_factor[t] >= cyclic_aging_factor[t] / 100 / 2) # Divide by 100 since parameters obtained 
                                                                                   # from AF curve in the unit of %
                                                                                   # Divide by 2 to get aging factor corresponding 
                                                                                   # to full cycle
        model.addConstr(delta_SOH[t] == BAT_CAPACITY * total_aging_factor[t] * (1 - EOL) / (13.5*0.2))

    #***********Solve optimization****************************************************************************************
    # Solve optimization and convert variables to arrays 
    model.optimize()
    if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL:
        pass
    else:
        print('Infeasible')
    charge = np.array([charge[i].X for i in range(len(charge))])
    discharge = np.array([discharge[i].X for i in range(len(discharge))])
    grid_buy = np.array([grid_buy[i].X for i in range(len(grid_buy))])
    grid_sell = np.array([grid_sell[i].X for i in range(len(grid_sell))])
    self_consumption = np.array([self_consumption[i].X for i in range(len(self_consumption))])
    battery_charge = np.array([battery_charge[i].X for i in range(len(battery_charge))])
    delta_SOH = np.array([delta_SOH[i].X for i in range(len(cyclic_aging_factor))])
    
    previous_values = {
        "charge": charge,
        "discharge": discharge,
        "battery_charge": battery_charge,
        "grid_buy": grid_buy,
        "grid_sell": grid_sell,
        "self_consumption": self_consumption,
        "delta_SOH": delta_SOH,
        "is_charging": [is_charging[t].X for t in range(len(is_charging))],
        "is_buying": [is_buying[t].X for t in range(len(is_buying))],
        "delta_SOC": [delta_SOC[t].X for t in range(len(delta_SOC))],
        "cyclic_aging_factor": [cyclic_aging_factor[t].X for t in range(len(cyclic_aging_factor))],
        "float_aging_factor": [float_aging_factor[t].X for t in range(len(float_aging_factor))],
        "total_aging_factor": [total_aging_factor[t].X for t in range(len(total_aging_factor))],
        "start_charge_bool": [start_charge_bool[t].X for t in range(len(start_charge_bool))],
        "start_discharge_bool": [start_discharge_bool[t].X for t in range(len(start_discharge_bool))],
        "end_charge_bool": [end_charge_bool[t].X for t in range(len(end_charge_bool))],
        "end_discharge_bool": [end_discharge_bool[t].X for t in range(len(end_discharge_bool))],
        "start_charge": [start_charge[t].X for t in range(len(start_charge))],
        "start_discharge": [start_discharge[t].X for t in range(len(start_discharge))]
    }

    return charge, discharge, grid_buy, grid_sell, self_consumption, battery_charge, delta_SOH, previous_values


def progbar(count_value, total, suffix=''):
    bar_length = 100
    filled_up_Length = int(round(bar_length* count_value / float(total)))
    percentage = round(100.0 * count_value/float(total),1)
    bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
    sys.stdout.write('[%s] %s%s ...%s\r' %(bar, percentage, '%', suffix))
    sys.stdout.flush()


def MPC(pv_power, pv_forecast, consumption, load_forecast, electricity_price, electricity_price_forecast, 
                    datetime, MARGIN, FIXED_COST, battery, bat_cost=True):
    """
    A function for rolling optimization: optimizing is performed using a horizon period of 24 hours and optimization will be performed every hour.
    Inputs:
        pv_power: (np array) production for the given input data (in kW)
        pv_forecast: (np array) forecasted production for the given input data (in kW)
        consumption: (np array) consumption for the given input data (in kW)
        load_forecast: (np array) forecasted consumption for the given input data (in kW)
        electricity_price: (np array) electricity spot price for the given input data (in EUR/kWh)
        electricity_price_forecast: (np array) 2D array containing day-ahead spot prices from 1 to 24 next hours
        datetime: (np array of datetime objects) the timestamps for the given input data
        MARGIN: (float) the margin added to electricity price
        FIXED_COST: (float) the fixed electricity price component (inc. VAT, TAX, and distribution fee)
        battery: instance of 'Battery' class
        bat_cost: (boolean) True if battery usage costs should be included in the optimization (MPC-BC strategy), otherwise False
    
    Returns:
        A tuple of pandas dataframe including the optimized actions for the given input period and the total costs as float
    """

    horizon_length = 24  # Optimization horizon in hours

    #***********************************************************************************************
    # Initialize arrays for saving the realized decisions
    #***********************************************************************************************
    pv_house = np.array([])
    pv_bat = np.array([])
    pv_grid = np.array([])
    bat_house = np.array([])
    grid_house = np.array([])
    pv_wasted = np.array([])
    to_house_wasted = np.array([])
    bat_charge = np.array([])
    costs = np.array([])
    profits = np.array([])
    delta_SOH = np.array([])

    #bat_charge_start = 0
    battery.ch = 0.0
    
    charge_bool = 0
    discharge_bool = 0
    start_charge = start_discharge = 0
    
    previous_values = None
    #***********************************************************************************************
    # Loop through each hour
    #***********************************************************************************************
    for current_hour in range(0, len(pv_power)):
        # Initialize iteration parameters
        horizon_start = current_hour
        horizon_end = min(current_hour + horizon_length, len(pv_power))
        #iter_resolution = min(horizon_length, len(pv_power)-current_hour)

        # Combine current spot price and next 24 hour spot prices
        current_and_predicted_electricity_price = np.append(electricity_price[current_hour],
                                                            electricity_price_forecast[current_hour])
        #***********************************************************************************************
        # Perform optimization for the current horizon
        #***********************************************************************************************
        if bat_cost:  # Use MPC-BC

            # Save previous charge before performing actions
            if current_hour == 0:
                previous_battery_charge = 0 
            else:
                previous_battery_charge = battery.ch
            # Perform optimization and obtain next battery state
            opt_charge, opt_discharge, \
            opt_grid_buy, opt_grid_sell, \
            opt_self_consumption, opt_battery_charge, opt_delta_SOH, previous_values = MILP_batcost(pv_power[horizon_start:horizon_end],
                                                                    pv_forecast[horizon_start:horizon_end], 
                                                                    consumption[horizon_start:horizon_end], 
                                                                    load_forecast[horizon_start:horizon_end],
                                                                    current_and_predicted_electricity_price,
                                                                    battery, MARGIN, FIXED_COST, 
                                                                    start_charge, charge_bool, 
                                                                    start_discharge, discharge_bool,
                                                                    previous_values)
            # Set charge and discharge booleans based on realized charge and discharge actions 
            if opt_charge[0] > 0:
                if discharge_bool == 1:
                    start_charge = previous_battery_charge
                    start_discharge = 0
                charge_bool = 1
                discharge_bool = 0
            elif opt_discharge[0] > 0:
                if charge_bool == 1:
                    start_discharge = previous_battery_charge
                    start_charge = 0
                discharge_bool = 1
                charge_bool = 0

        else: # Otherwise use MPC
            opt_charge, opt_discharge, \
            opt_grid_buy, opt_grid_sell, \
            opt_self_consumption, opt_battery_charge, opt_delta_SOH = MILP(pv_power[horizon_start:horizon_end],
                                                            pv_forecast[horizon_start:horizon_end], 
                                                            consumption[horizon_start:horizon_end], 
                                                            load_forecast[horizon_start:horizon_end],
                                                            current_and_predicted_electricity_price, 
                                                            battery, MARGIN, FIXED_COST)

        #***********************************************************************************************
        # Save first instances of optimized results results to arrays
        #***********************************************************************************************
        pv_house = np.append(pv_house,      opt_self_consumption[0])
        pv_bat = np.append(pv_bat,          opt_charge[0])
        pv_grid = np.append(pv_grid,        opt_grid_sell[0])
        bat_house = np.append(bat_house,    opt_discharge[0])
        grid_house = np.append(grid_house,  opt_grid_buy[0])
        pv_wasted = np.append(pv_wasted,    0.0)
        to_house_wasted = np.append(to_house_wasted, np.maximum(0.0, 
                                                                opt_discharge[0]+opt_self_consumption[0]+
                                                                opt_grid_buy[0]-consumption[current_hour]))
        delta_SOH = np.append(delta_SOH, opt_delta_SOH[0])
        #bat_charge = np.append(bat_charge, opt_battery_charge[0])
        bat_charge = np.append(bat_charge, battery.ch)
        costs = np.append(costs, (opt_grid_buy[0] * (electricity_price[current_hour]*1.24 + MARGIN + FIXED_COST))/100)
        profits = np.append(profits, (opt_grid_sell[0] * (electricity_price[current_hour] - MARGIN))/100)

        # Update the battery charge for the next period 
        # (index 1 refers to the next battery charge, index 0 refers to the current charge)
        try:
            battery.update(charge_power=opt_charge[0], discharge_power=opt_discharge[0])
            
        except IndexError:  # Prevent the error that would occur if the last optimization 
            pass            # horizon has a length of one
        
        # Gap the charge, if max capacity of battery is degradated
        battery.ch = min(battery.ch, battery.C)

        progbar(current_hour, len(pv_power))

    results = np.array([pv_house, pv_bat, bat_house, grid_house, pv_grid, 
                        pv_wasted, to_house_wasted, bat_charge, costs, profits, delta_SOH]).T

    return results, (costs.sum()-profits.sum())


def MILP_whole_year(pv_power, pv_forecast, consumption, load_forecast, electricity_price, 
                    datetime, MARGIN, FIXED_COST, battery):
    """
    A function for optimizing the whole year with MILP.
    Inputs:
        pv_power: (np array) production for the given input data (in kW)
        pv_forecast: (np array) forecasted production for the given input data (in kW)
        consumption: (np array) consumption for the given input data (in kW)
        load_forecast: (np array) forecasted consumption for the given input data (in kW)
        electricity_price: (np array) electricity spot price for the given input data (in EUR/kWh)
        datetime: (np array of datetime objects) the timestamps for the given input data
        MARGIN: (float) the margin added to electricity price
        FIXED_COST: (float) the fixed electricity price component (inc. VAT, TAX, and distribution fee)
        battery: instance of 'Battery' class
    
    Returns:
        A pandas dataframe including the optimized actions for the given input period
    """

    battery.ch = 0.0
    
    opt_charge, opt_discharge, \
    opt_grid_buy, opt_grid_sell, \
    opt_self_consumption, opt_battery_charge, opt_delta_SOH = MILP(pv_power, pv_forecast, consumption, load_forecast,
                                                    electricity_price, battery, MARGIN, FIXED_COST)

    #***********************************************************************************************
    # Save optimized results to arrays
    #***********************************************************************************************
    bat_charge = np.array([])
    for i in range(len(opt_battery_charge)-1):
        battery.update(charge_power=opt_charge[i], discharge_power=opt_discharge[i], include_float_degradation=True)
        bat_charge = np.append(bat_charge, battery.ch)
    pv_house = opt_self_consumption
    pv_bat = opt_charge
    pv_grid = opt_grid_sell
    bat_house = opt_discharge
    grid_house = opt_grid_buy
    pv_wasted = np.zeros(grid_house.shape)
    to_house_wasted = np.zeros(grid_house.shape)
    delta_SOH = battery.deltaSOH_history
    costs = (opt_grid_buy * (electricity_price*1.24 + MARGIN + FIXED_COST))/100
    profits = (opt_grid_sell * (electricity_price - MARGIN))/100

    results = np.array([pv_house, pv_bat, bat_house, grid_house, pv_grid, 
                        pv_wasted, to_house_wasted, bat_charge, costs, profits, delta_SOH]).T

    return results, (costs.sum()-profits.sum())


def MILP_BC_whole_year(pv_power, pv_forecast, consumption, load_forecast, electricity_price, 
                    datetime, MARGIN, FIXED_COST, battery):
    """
    A function for optimizing the whole year with MILP with battery usage costs in the loss function.
    Inputs:
        pv_power: (np array) production for the given input data (in kW)
        pv_forecast: (np array) forecasted production for the given input data (in kW)
        consumption: (np array) consumption for the given input data (in kW)
        load_forecast: (np array) forecasted consumption for the given input data (in kW)
        electricity_price: (np array) electricity spot price for the given input data (in EUR/kWh)
        datetime: (np array of datetime objects) the timestamps for the given input data
        MARGIN: (float) the margin added to electricity price
        FIXED_COST: (float) the fixed electricity price component (inc. VAT, TAX, and distribution fee)
        battery: instance of 'Battery' class
    
    Returns:
        A pandas dataframe including the optimized actions for the given input period
    """
    battery.ch = 0.0
    charge_bool = 0
    discharge_bool = 0
    start_charge = start_discharge = 0
    
    opt_charge, opt_discharge, opt_grid_buy, opt_grid_sell, \
    opt_self_consumption, opt_battery_charge, opt_delta_SOH = MILP_batcost(pv_power, pv_forecast, consumption, load_forecast, electricity_price, 
                                                                           battery, MARGIN, FIXED_COST, start_charge, charge_bool, 
                                                                           start_discharge, discharge_bool)

    #***********************************************************************************************
    # Save optimized results to arrays
    #***********************************************************************************************
    bat_charge = np.array([])
    for i in range(len(opt_charge)):
        battery.update(charge_power=opt_charge[i], discharge_power=opt_discharge[i], include_float_degradation=False)
        bat_charge = np.append(bat_charge, battery.ch)
    pv_house = opt_self_consumption
    pv_bat = opt_charge
    pv_grid = opt_grid_sell
    bat_house = opt_discharge
    grid_house = opt_grid_buy
    pv_wasted = np.zeros(grid_house.shape)
    to_house_wasted = np.zeros(grid_house.shape)
    delta_SOH = opt_delta_SOH
    costs = (opt_grid_buy * (electricity_price*1.24 + MARGIN + FIXED_COST))/100
    profits = (opt_grid_sell * (electricity_price - MARGIN))/100

    results = np.array([pv_house, pv_bat, bat_house, grid_house, pv_grid, 
                        pv_wasted, to_house_wasted, bat_charge, costs, profits, delta_SOH]).T

    return results, (costs.sum()-profits.sum())
