#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#*******************************
# Import modules
#*******************************
import os
import sys
from copy import deepcopy
from math import sqrt
import time
import numpy as np
import pandas as pd
import gc
import sklearn.neural_network as nn
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

if __name__ == '__main__':
    cost_func = sys.argv[-4]
else:
    print('Using nobatcost as default')
    cost_func = 'nobatcost'

if cost_func == 'batcost':
    from Costs_and_losses import Loss_bat_cost as Loss
elif cost_func == 'nobatcost':
    from Costs_and_losses import Loss
sys.path.insert(0, '..')
from Functions import read_data


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
torch.set_num_threads(80)
torch.manual_seed(0)

# Initialize simulation parameters
#************************************************
Nhours = 7*24  # Length of one episode 
N_ACTIONS = 1

# Battery sizing based on Tesla Powerwall
#************************************************* 
BATTERY_SIZE = 13.5 # Usable battery size

# Electricity price components
#*************************************************
MARGIN = 0.4 # [c/kWh]
# Transmission + electricity tax (inc. VAT) [2018, 2019, 2020, 2021, 2022, 2023]
FIXED_COSTS = [5.69, 5.90, 5.90, 5.44, 5.01, 5.01] # c/kWh

class NN(torch.nn.Module):
    def __init__(self, Nin, widFCN, depFCN, Nout):
        super(NN, self).__init__()

        self.Nin = Nin
        self.widFCN = widFCN
        self.depFCN = depFCN
        self.Nout = Nout

        # Fully connected layers with ReLU activation
        FCN_layers = []
        FCN_layers.append(nn.Linear(self.Nin, self.widFCN))
        FCN_layers.append(nn.ReLU())

        # Additional hidden layers
        for _ in range(self.depFCN):
            FCN_layers.append(nn.Linear(self.widFCN, self.widFCN))
            FCN_layers.append(nn.ReLU())

        self.fcn = nn.Sequential(*FCN_layers)

        # Tanh activation used to map the output between [-1,1]
        self.tanh_layer = nn.Sequential(*(nn.Linear(self.widFCN, (1)), nn.Tanh()))


    def forward(self, x):
        hidden = self.fcn(x)
        y = self.tanh_layer(hidden)

        return torch.reshape(y, (y.shape[0], N_ACTIONS))


def TrainModel(data_x, spot, pv, load, battery_sh, epochs, print_progress, MARGIN, FIXED_COST, early_stopping=True):
    #***********************************************************************
    # Create train-test splits
    #***********************************************************************
    train_size = int(0.8 * (len(data_x)-Nhours))
    np.random.seed(24)
    train_ind = np.random.choice(len(data_x)-Nhours, size=train_size, replace=False)
    test_ind = np.setdiff1d(np.arange(len(data_x)-Nhours),train_ind)

    #***********************************************************************
    # Select train and validation datasets
    #***********************************************************************
    tr_x = np.array([data_x[ind:ind+Nhours,:] for ind in train_ind])
    val_x = np.array([data_x[ind:ind+Nhours,:] for ind in test_ind])
    tr_spot, val_spot = np.array([spot[ind:ind+Nhours] for ind in train_ind]), np.array([spot[ind:ind+Nhours] for ind in test_ind])
    tr_pv, val_pv = np.array([pv[ind:ind+Nhours] for ind in train_ind]), np.array([pv[ind:ind+Nhours] for ind in test_ind])
    tr_load, val_load = np.array([load[ind:ind+Nhours] for ind in train_ind]), np.array([load[ind:ind+Nhours] for ind in test_ind])
    tr_battery_sh, val_battery_sh = np.array([battery_sh[ind:ind+Nhours] for ind in train_ind]), np.array([battery_sh[ind:ind+Nhours] for ind in test_ind])
    tr_x_scaled = tr_x.copy()
    val_x_scaled = val_x.copy()
    scaler_x = StandardScaler()
    scaler_x.fit(data_x[train_ind,:-1])
    tr_x_scaled[:,:,:-1]  = np.array([scaler_x.transform(arr) for arr in tr_x[:,:,:-1]])
    val_x_scaled[:,:,:-1]  = np.array([scaler_x.transform(arr) for arr in val_x[:,:,:-1]])
    # Scale soc with minmax
    tr_x_scaled[:,:,-1] = tr_x[:,:,-1]/BATTERY_SIZE
    val_x_scaled[:,:,-1] = val_x[:,:,-1]/BATTERY_SIZE
    # Calculate the average electric price of the next 9 hours for giving price to the energy recerved in the battery
    tr_bat_storage_value = np.array([np.median(data_x[ind:ind+Nhours,3:8], axis=1) for ind in train_ind])
    tr_bat_storage_value = tr_bat_storage_value * 1.24
    val_bat_storage_value = np.array([data_x[ind:ind+Nhours,3:12].max(axis=1) for ind in test_ind])*0

    #***********************************************************************
    # Convert data to tensors
    #***********************************************************************
    tr_x_scaled = torch.tensor(tr_x_scaled, dtype=torch.float32)
    tr_spot = torch.tensor(tr_spot, dtype=torch.float32)
    tr_pv = torch.tensor(tr_pv, dtype=torch.float32)
    tr_load = torch.tensor(tr_load, dtype=torch.float32)
    tr_battery_sh = torch.tensor(tr_battery_sh, dtype=torch.float32)
    tr_bat_storage_value = torch.tensor(tr_bat_storage_value, dtype=torch.float32)
    val_x_scaled = torch.tensor(val_x_scaled, dtype=torch.float32)
    val_spot = torch.tensor(val_spot, dtype=torch.float32)
    val_pv = torch.tensor(val_pv, dtype=torch.float32)
    val_load = torch.tensor(val_load, dtype=torch.float32)
    val_battery_sh = torch.tensor(val_battery_sh, dtype=torch.float32)
    val_bat_storage_value = torch.tensor(val_bat_storage_value, dtype=torch.float32)

    model    = NN(78, 100, 3, 1)
    # Construct our loss function (mean of squared deviation matrix) and an Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    
    learn = True
    epoch = -1
    early_stop_counter = 0
    best_tr = np.inf
    best_val = np.inf

    while learn:
        model.train()
        epoch        += 1                 
        BS            = 200
        epoch_shuffle = np.random.permutation(np.arange(tr_x.shape[0]))
        #*********************************************
        # Run through the batches in the epoch
        #*********************************************
        for batch in range(int(tr_x.shape[0]/BS)):
            if not batch == int(tr_x.shape[0]/BS)-1: 
                Si = batch*BS
                Ei = (batch+1)*BS
            else:
                  Si = batch*BS
                  Ei = tr_x.shape[0]-1
            #*************************************************************
            # Compute loss
            #*************************************************************        
            loss = Loss(tr_x_scaled[epoch_shuffle][Si:Ei], 
                        tr_spot[epoch_shuffle][Si:Ei],
                        tr_pv[epoch_shuffle][Si:Ei],
                        tr_load[epoch_shuffle][Si:Ei],
                        tr_battery_sh[epoch_shuffle][Si:Ei],
                        tr_bat_storage_value[epoch_shuffle][Si:Ei],
                        model, MARGIN, FIXED_COST)
            #*************************************************************
            # Zero gradients, perform a backward pass, 
            # update the weights, and
            # evaluate train error for the batch
            #*************************************************************
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        #*************************************************************
        # Early stopping, both train and generalization error need to
        # decrease
        #*************************************************************
        if epoch % 1 == 0:
            model.eval()

            # Calculate current train and validation socres (electricity costs)
            current_tr = Loss(tr_x_scaled, tr_spot, tr_pv, tr_load, tr_battery_sh, tr_bat_storage_value*0, model, MARGIN, FIXED_COST).data.numpy()
            current_val = Loss(val_x_scaled, val_spot, val_pv, val_load, val_battery_sh, val_bat_storage_value*0, model, MARGIN, FIXED_COST).data.numpy()
            
            # Check wether the current scores smaller than previous scores
            if (current_val >= best_val) and early_stopping:
                early_stop_counter += 1
            elif current_tr >= best_tr and early_stopping:
                early_stop_counter += 1
            else:
                best_tr = current_tr
                best_val = current_val
                early_stop_counter = 0
                best_model = deepcopy(model)
                best_model_state = best_model.state_dict()
            if print_progress:
                print(f'Tr: {current_tr:.6f}, val: {current_val:.6f}')
            # Stop training if nan values are encountered
            if np.isnan(current_tr) or np.isnan(current_val):
                print('Stopping due to nans')
                learn = False

            # Stop training if early stop condition is met
            if early_stop_counter == 20:
                print('early stop after {} epochs'.format(epoch))
                learn = False
    
        # Stop training after the number of epochs is met
        if epoch % epochs == 0 and epoch > 1:
            learn = False
            
    best_model.eval()

    # Free memory (just in case in order to not cause troubles in the next round)
    del model
    del tr_x
    del val_x
    del tr_x_scaled
    del tr_spot
    del tr_pv
    del tr_load
    del tr_battery_sh
    del tr_bat_storage_value
    del val_x_scaled
    del val_spot
    del val_pv
    del val_load
    del val_battery_sh
    del val_bat_storage_value
    gc.collect()

    return best_tr, best_val, best_model, scaler_x, epoch


if __name__ == '__main__':
    file_name = sys.argv[-3]
    model_name = sys.argv[-2]
    scaler_name = sys.argv[-1]
    data_x, spot, future_spot, pv, pv_forec, load, load_forec, battery_sh, datetime = \
                read_data(file_name, first_month=3, last_month=9, 
                          return_datetime=True, multiply_data=10, forecast='naive')
    
    MARGIN = 0.4 # [c/kWh]
    FIXED_COSTS = [5.69, 5.90, 5.90, 5.44, 5.01, 5.01] # [c/kWh] [2018, 2019, 2020, 2021, 2022, 2023]
    FIXED_COST = FIXED_COSTS[pd.Series(datetime)[0].year-2018]

    start_time = time.time()

    best_tr, best_val, model, scaler, epochs = TrainModel(data_x, spot, pv, load, battery_sh, epochs=2000, print_progress=True, MARGIN=MARGIN, FIXED_COST=FIXED_COST)

    torch.save(model.state_dict(), os.path.join('Results', model_name))
    torch.save(scaler, os.path.join('Results', scaler_name))
    print(f'\nModel saved succesfully after {epochs} epoch!')

    print("--- Execution time: %s seconds ---" % (time.time() - start_time))
