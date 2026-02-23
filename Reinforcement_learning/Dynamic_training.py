import numpy as np
import pandas as pd
import torch
import os
import sys
from RL_lr_dec import NN

cost_func = sys.argv[-2]
if cost_func == 'batcost':
    from Costs_and_losses import Loss_bat_cost as Loss, Interpret, cyclic_aging_factor, float_aging_factor
elif cost_func == 'nobatcost':
    from Costs_and_losses import Loss, Interpret, cyclic_aging_factor, float_aging_factor

sys.path.insert(0, '..')
from Functions import read_data
#from Find_best_model import find_best_params

torch.set_num_threads(80)

Nhours = 7*24
BATTERY_SIZE = 13.5 # Usable battery size
MARGIN = 0.4 # [c/kWh]
# Transmission + electricity tax (inc. VAT) [2018, 2019, 2020, 2021, 2022, 2023]
FIXED_COSTS = [5.69, 5.90, 5.90, 5.44, 5.01, 5.01] # c/kWh
BAT_CAPEX = 9000
END_OF_LIFE = 0.8  # Percentual capacity in the end of life [-]

load_profile = sys.argv[1]  # E.g. L9
file_name_start = sys.argv[2]  # The file name until the _year_UTC03.csv (e.g. L9_real_4kWp)
model_name = sys.argv[3]  # Name of the trained model to be used
scaler = sys.argv[4]  # Name of the trained scaler to be used
save_name = sys.argv[-1]


def infere_data_dynamic(year, include_battery_degradation=False):
    global BATTERY_SIZE
    #****************************************************************
    # Load the model
    #****************************************************************
    scaler_x = torch.load(os.path.join('Results', load_profile, scaler))
    model = NN(Nin=78, widFCN=100, depFCN=3, Nout=1)
    if year == 2020:
        model.load_state_dict(torch.load(os.path.join('Results', load_profile, model_name)))
    else:
        model.load_state_dict(torch.load(os.path.join('Results', load_profile, f'model_{save_name}_{year-1}.pt')))
    model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    #****************************************************************
    # Load the data
    #****************************************************************
    data_x, spot, future_spot, pv_data, pv_forec, load_data, load_forec, battery_sh, datetime = \
        read_data(f'{file_name_start}_{year}_UTC03.csv', first_month=1, last_month=12, 
                  return_datetime=True, multiply_data=1, forecast='naive')

    FIXED_COST = FIXED_COSTS[year-2018]

    #****************************************************************
    # Initialize the result vectors
    #****************************************************************
    pv_house = np.array([])
    pv_bat = np.array([])
    pv_grid = np.array([])
    bat_house = np.array([])
    grid_house = np.array([])
    costs = np.array([])
    profits = np.array([])
    pv_wasted = np.array([])
    to_house_wasted = np.array([])
    cost = 0
    profit = 0 
    bat_charge = np.array([])
    
    # Initialize arrays for histories
    inputs = np.zeros(78)
    recommendations = np.array([])
    spots = np.array([])
    prods = np.array([])
    loads = np.array([])
    bats = np.array([])
    bat_stor_vals = np.array([])
    battery = torch.tensor(0, dtype=torch.float32)

    # Initialize arrays for degradation calculation
    operation_mode = torch.tensor(np.ones(1), requires_grad=True, dtype=torch.float32)
    delta_SOC = torch.tensor(np.zeros(1), requires_grad=True, dtype=torch.float32)
    aging_factors = np.array([])
    aging_factors_cyclic = np.array([])
    aging_factors_float = np.array([])
    deltaSOH_cyclic = np.array([])
    deltaSOH_float = np.array([])
    deltaSOH = np.array([])
    battery_usage_costs = np.array([])
    capacities = np.array([])

    i = -1
    for row in data_x:
        i = i + 1
        # Process the row for the model
        model.eval()
        row = row.reshape(1,-1)
        PV = row[0][0]
        load = row[0][1]
        spot_h = row[0][2]
        p_purch = spot_h*1.24 + MARGIN + FIXED_COST
        p_sell = spot_h - MARGIN

        data_scaled = row.copy()
        data_scaled[0][:-1] = scaler_x.transform(row[0][:-1].reshape(1,-1))
        data_scaled[0][-1] = battery/BATTERY_SIZE
        data_scaled = torch.tensor(data_scaled, dtype=torch.float32)
        # Make prediction for the given hour
        recommendation = model(data_scaled)
        # Save inputs and predictions to the lists
        inputs = np.vstack([inputs, data_scaled.detach().numpy().flatten()])
        # Remove the zero row
        if i == 0:
            inputs = inputs[1:]
        recommendations = np.append(recommendations, recommendation.detach().numpy())
        spots = np.append(spots, spot_h)
        prods = np.append(prods, PV)
        loads = np.append(loads, load)
        bats = np.append(bats, battery.detach().numpy())
        bat_stor_vals = np.append(bat_stor_vals, np.median(row[0][3:8])*1.24)
        battery_prev = battery.clone()

        # Interpret the prediction to obtain the actions
        # (pv to house, pv to battery, pv to grid, battery to hourse, grid to house)
        (pv_to_h, pv_to_b, pv_to_g, b_to_h, g_to_h, pv_waste, battery) = Interpret(recommendation, torch.tensor(PV, dtype=torch.float32), 
                                                                              torch.tensor(load, dtype=torch.float32), battery)

        # Calculate the degradation
        # Check if operation mode (charge, discharge) changed and update half-cycle history
        delta_SOC_new = delta_SOC.clone()
        cyclic_AF = torch.zeros_like(battery)
        # Float degradation
        float_AF = float_aging_factor(battery, BATTERY_SIZE)
        
        # Battery is charged
        charging_condition = battery > battery_prev
        discharging_condition = battery_prev > battery

        # Reset delta SOC if operation mode chages
        delta_SOC_new = torch.where(charging_condition & (operation_mode == -1.), torch.zeros_like(delta_SOC), delta_SOC_new)
        delta_SOC_new = torch.where(discharging_condition & (operation_mode == 1.), torch.zeros_like(delta_SOC), delta_SOC_new)

        # Handle charging condition
        #delta_SOC_new = torch.where(charging_condition & (operation_mode == 1.), delta_SOC_new + (battery - battery_prev), delta_SOC_new)
        delta_SOC_new = torch.where(charging_condition, delta_SOC_new + (battery - battery_prev), delta_SOC_new)
        cyclic_AF = torch.where(charging_condition & (operation_mode == -1.), cyclic_aging_factor(delta_SOC, BATTERY_SIZE), cyclic_AF)
        operation_mode = torch.where(charging_condition, torch.tensor(1., dtype=torch.float32), operation_mode)
        
        # Handle discharging condition
        #delta_SOC_new = torch.where(discharging_condition & (operation_mode == -1.), delta_SOC_new + (battery_prev - battery), delta_SOC_new)
        delta_SOC_new = torch.where(discharging_condition, delta_SOC_new + (battery_prev - battery), delta_SOC_new)
        cyclic_AF = torch.where(discharging_condition & (operation_mode == 1.), cyclic_aging_factor(delta_SOC, BATTERY_SIZE), cyclic_AF)
        operation_mode = torch.where(discharging_condition, torch.tensor(-1., dtype=torch.float32), operation_mode)
        
        aging_factor =  torch.maximum(cyclic_AF, float_AF)  # Degradation is the maximum of the two degradation mechanisms
        
        # Update delta SOC for next iteration
        delta_SOC = delta_SOC_new.clone()
        # Update battery capacity
        if include_battery_degradation:
            BATTERY_SIZE = BATTERY_SIZE * (1 - aging_factor*(1 - END_OF_LIFE)).detach().numpy().astype(float)[0]

        # Calculate battery usage cost
        delta_SOH = BATTERY_SIZE * aging_factor.detach().numpy() * (1 - 0.8) / (13.5*0.2)
        battery_usage_cost = delta_SOH * BAT_CAPEX  # Battery price in euros
        aging_factors = np.append(aging_factors, aging_factor.detach().numpy())
        aging_factors_cyclic = np.append(aging_factors_cyclic, cyclic_AF.detach().numpy())
        aging_factors_float = np.append(aging_factors_float, float_AF.detach().numpy())
        
        if float_AF.detach().numpy()[0] <= cyclic_AF.detach().numpy()[0]:
            deltaSOH_float = np.append(deltaSOH_float, 0)
            deltaSOH_cyclic = np.append(deltaSOH_cyclic, delta_SOH)
        else:
            deltaSOH_float = np.append(deltaSOH_float, delta_SOH)
            deltaSOH_cyclic = np.append(deltaSOH_cyclic, 0)

        deltaSOH = np.append(deltaSOH, delta_SOH) 
        battery_usage_costs = np.append(battery_usage_costs, battery_usage_cost)
        capacities = np.append(capacities, BATTERY_SIZE)

        # Update the model once every week using data from previous two weeks
        if (i % Nhours == 0) and (len(spots) >= 2*Nhours):
            #print(i)
            input = np.array([inputs[ind:ind+Nhours] for ind in range(i-2*Nhours,i-Nhours)])
            s = np.array([spots[ind:ind+Nhours] for ind in range(i-2*Nhours,i-Nhours)])
            p = np.array([prods[ind:ind+Nhours] for ind in range(i-2*Nhours,i-Nhours)])
            l = np.array([loads[ind:ind+Nhours] for ind in range(i-2*Nhours,i-Nhours)])
            bat = np.array([bats[ind:ind+Nhours] for ind in range(i-2*Nhours,i-Nhours)])
            bat_stor_val = np.array([bat_stor_vals[ind:ind+Nhours] for ind in range(i-2*Nhours,i-Nhours)])
            model.train()

            for epoch in range(10):
                loss = Loss(
                    torch.tensor(input, dtype=torch.float32), 
                    torch.tensor(s, dtype=torch.float32),
                    torch.tensor(p, dtype=torch.float32),
                    torch.tensor(l, dtype=torch.float32),
                    torch.tensor(bat, dtype=torch.float32),
                    torch.tensor(bat_stor_val, dtype=torch.float32),
                    model, MARGIN, FIXED_COST)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            print(datetime[i], ':', loss.detach().numpy())

        # Save the actions to the lists
        pv_house = np.append(pv_house, pv_to_h.detach().numpy())
        pv_bat = np.append(pv_bat, pv_to_b.detach().numpy())
        pv_grid = np.append(pv_grid, pv_to_g.detach().numpy())
        bat_house = np.append(bat_house, b_to_h.detach().numpy())
        grid_house = np.append(grid_house, g_to_h.detach().numpy())
        pv_wasted = np.append(pv_wasted, pv_waste.detach().numpy())
        to_house_wasted = np.append(to_house_wasted, np.maximum(0, b_to_h.detach().numpy()+pv_to_h.detach().numpy()-load))
        bat_charge = np.append(bat_charge, battery.detach().numpy())

        profit += (pv_to_g).detach().numpy()[0]*p_sell/100
        cost += (g_to_h).detach().numpy()[0]*p_purch/100
        profits = np.append(profits, (pv_to_g).detach().numpy()[0]*p_sell/100)
        costs = np.append(costs, (g_to_h).detach().numpy()[0]*p_purch/100)


    #****************************************************************
    # Study reference policies
    #****************************************************************
    print('\n\n')
    #print(pv.sum())
    #print(load.max())
    print('No solar system at all cost:',np.sum(load*(spot*1.24+MARGIN+FIXED_COST))/100.0)
    from_grid = (load_data-pv_data)
    wasted = from_grid.copy()
    wasted[from_grid>=0.0] = 0
    from_grid[from_grid<0.0]=0.0
    print('All solar used immediately (possible excess dumped):',np.sum(from_grid*(spot*1.24+MARGIN+FIXED_COST))/100)
    print('Value of dumped electricity:',np.sum(wasted*(spot-MARGIN))/100.0)
    print('\n\n')
    print('Daily electricity bill (EUR):', (cost-profit))

    results = np.array([pv_house, pv_bat, bat_house, grid_house, pv_grid,
                        pv_wasted, to_house_wasted, prods, loads, bat_charge, costs, profits]).T

    results_df = pd.DataFrame(results, columns=['pv_to_house', 'pv_to_bat', 'bat_to_house', 
                                                'grid_to_house', 'pv_to_grid', 'pv_to_bat_grid_wasted', 
                                                'to_house_wasted', 'power', 'load', 'soc', 'cost', 'profit'])
    results_df['spot'] = spot
    results_df['datetime'] = datetime
    results_df['power'] = pv_data
    results_df['load'] = load_data
    results_df['battery_capacity'] = capacities
    results_df['deltaSOH_cyclic'] = deltaSOH_cyclic
    results_df['deltaSOH_float'] = deltaSOH_float
    results_df['battery_deltaSOH'] = deltaSOH
    results_df['battery_usage_cost'] = battery_usage_costs
    results_df['aging_factor_cyclic'] = aging_factors_cyclic
    results_df['aging_factor_float'] = aging_factors_float
    results_df['battery_aging_factor'] = aging_factors
    results_df.to_csv(os.path.join('Results', load_profile, f'{save_name}_{year}_UTC03.csv'), index=False)
    
    save_model_path = os.path.join('Results', load_profile, f'model_{save_name}_{year}.pt')
    torch.save(model.state_dict(), save_model_path)


include_battery_degradation = True

infere_data_dynamic(2020, include_battery_degradation)
infere_data_dynamic(2021, include_battery_degradation)
infere_data_dynamic(2022, include_battery_degradation)
