import numpy as np
import sys
from Battery.Inverter import Inverter

class Battery:
    """
    Battery degradation is modelled based on 
    https://www.researchgate.net/publication/293145570_CONCEPT_OF_A_BATTERY_AGING_MODEL_FOR_LITHIUM-ION_BATTERIES_CONSIDERING_THE_LIFETIME_DEPENDENCY_ON_THE_OPERATION_STRATEGY
    """

    def __init__(self, include_degradation):
        # ********************************************************
        # Battery constants
        # ********************************************************
        self.include_degradation = include_degradation  # Boolean
        self.C = 13.5  # battery initial capacity [kWh]
        self.MAX_CH_POWER = 5  # kW
        self.MAX_DCH_POWER = 5  # kW
        self.CAPEX = 9000 # capital costs [EUR]
        self.MIN_CHARGE = 0.0 # minimum charge [kWh]
        self.END_OF_LIFE = 0.8  # Percentual capacity in the end of life [-]
        self.LIFETIME = 15  # Expected years to reach the end-of-life
        # Initialize inverter
        self.inverter = Inverter(rated_power=self.MAX_CH_POWER)
        # ********************************************************
        # Degradation parameters
        # ********************************************************
        # Cyclic
        self.a = 1.2698*10**6
        self.b = -1.3133
        # Float
        self.d = 2
        self.e = -1.2
        self.f = -0.0275
        # Linearized cyclic degradation params (see 'degradation_efficiency_curve.ipynb')
        self.cyclic_degradation_curve_lin_slopes = [0.00015675769, 0.00024698501, 0.00033011383, 0.00040564024]  # Values obtained when y-values multiplied by 100 (in the unit of %)
        self.cyclic_degradation_curve_lin_inters = [0.0, -0.00081204588, -0.002807137746, -0.0072631956]  # Values obtained when y-values multiplied by 100 (in the unit of %)
        self.cyclic_degradation_curve_lin_SOCS = [0, 10, 25, 60]  # Starting points for change-points
        # Linearized float degradation params (see 'degradation_efficiency_curve.ipynb')
        self.float_degradation_curve_lin_slopes = [0.0008390540006103, 0.0023000485806749594, 0.007772392380464037, 0.025749092486947102] # Values represent SOC-aging, not float aging,
                                                                                                                                          # divide these with 10*8760 to get the float params
        self.float_degradation_curve_lin_inters = [0.5199433185834548, 0.48487944866190297, 0.16201116447434732, -1.34803164447023] # Values represent SOC-aging, not float aging,
                                                                                                                                    # divide these with 10*8760 to get the float params
        self.float_degradation_curve_lin_SOCS = [0, 25, 60, 85]  # Starting points for change-points

        # ********************************************************
        # Dynamic variables
        # ********************************************************
        self.ch = 0.0  # Battery state of charge [kWh]
        self.previous_ch = 0.0
        self.capacity_history = np.array([])  # Array containing the capacity 
                                              # which decreases due to ageing
        self.aging_factor_history = np.array([])
        self.aging_factor_cyclic_history = np.array([])
        self.aging_factor_float_history = np.array([])
        self.deltaSOH_history = np.array([])
        self.deltaSOH_cyclic_history = np.array([])
        self.deltaSOH_float_history = np.array([])
        self.operation_mode = 'idle'  # Charge, discharge, or idle
        self.half_cycle_history = np.array([])  # Array containing the battery soc (in kWh) 
                                                # through each half-cycle

    def reset_history(self):
        self.capacity_history = np.array([])  
        self.aging_factor_history = np.array([])
        self.aging_factor_cyclic_history = np.array([])
        self.aging_factor_float_history = np.array([])
        self.deltaSOH_history = np.array([])
        self.deltaSOH_cyclic_history = np.array([])
        self.deltaSOH_float_history = np.array([])
        self.half_cycle_history = np.array([])

    def eff(self, power):
        """
        Function estimating the relation between charge/discharge power and efficiency
        """
        if power == 0:
            return 1  # Prevent devision by 0
        # Add inverter efficiency
        inverter_eff = self.inverter.efficiency(power)
        return np.sqrt(0.9) * inverter_eff
    
    def __charge(self, power):
        self.ch = min(self.ch + power*self.eff(power), self.C)

    def __discharge(self, power):
        self.ch = max(self.ch - power/self.eff(power), 0.0)

    def __soc_degradation(self):
        return 1/(self.d+self.e*np.exp(self.f*(100-self.ch/(self.C+1E-10)*100)))

    def float_degradation(self):
        soc_degr = self.__soc_degradation() / (15*8760)  # Divide by the reference lifetime of 15 years
        return soc_degr

    def __N_life(self, delta_SOC):
        """
        Function for estimating cycle life from DOD, based on
        https://www.researchgate.net/publication/293145570_CONCEPT_OF_A_BATTERY_AGING_MODEL_FOR_LITHIUM-ION_BATTERIES_CONSIDERING_THE_LIFETIME_DEPENDENCY_ON_THE_OPERATION_STRATEGY
        """
        N_max_soc = self.a * delta_SOC ** self.b
        return N_max_soc

    def aging_factor(self, delta_SOC):
        return 1/self.__N_life(delta_SOC*100+1E-20) / 2  # Divide by two since this is a half-cycle, not full cycle

    def cyclic_degradation(self):
        delta_SOC = (self.half_cycle_history.max() - self.half_cycle_history.min())/self.C
        aging_factor = self.aging_factor(delta_SOC)
        return aging_factor
    

    def update(self, charge_power, discharge_power, include_float_degradation=True):
        """
        Idle-mode does not brake the half-cycle. This function is used as default.
        """
        # Check that no charge and discharge at the same time
        if charge_power * discharge_power > 1E-10:  # Use small value instead of 0 due to computation accuracy
            # Close the program
            sys.exit(f'ERROR: charge and discharge occuring simultaneously:\
                     \nCharge: {charge_power}\nDischarge: {discharge_power}')
        
        # Update previous charge and add to history
        self.previous_ch = self.ch
        self.capacity_history = np.append(self.capacity_history, self.C)
        # Update battery charge
        self.__charge(charge_power)
        self.__discharge(discharge_power)

        # Check if operation mode (charge, discharge) changed and update half-cycle history
        if charge_power > discharge_power:
            if self.operation_mode in ['charge', 'idle']:
                self.operation_mode = 'charge'
                # Battery charging, add battery charge to half-cycle history
                self.half_cycle_history = np.append(self.half_cycle_history, self.ch)
                aging_factor_cyc = 0.0
            elif self.operation_mode == 'discharge':
                # Half-cycle change-point, add current minimum battery half-cycle charge to history
                self.half_cycle_history = np.append(self.half_cycle_history, self.previous_ch)
                # Calculate degradation
                aging_factor_cyc = self.cyclic_degradation()
                # Reset half-cycle history
                self.half_cycle_history = np.array([self.previous_ch])
                # Change mode to charge
                self.operation_mode = 'charge'
        elif discharge_power > charge_power:  # Discharging
            if self.operation_mode in ['discharge', 'idle']:
                self.operation_mode = 'discharge'
                # Battery still discharging, add battery charge to half-cycle history
                self.half_cycle_history =  np.append(self.half_cycle_history, self.ch)
                aging_factor_cyc = 0.0
            elif self.operation_mode == 'charge':
                # Half-cycle change-point, add maximum battery half-cycle charge to history
                self.half_cycle_history = np.append(self.half_cycle_history, self.previous_ch)
                # Calculate degradation
                aging_factor_cyc = self.cyclic_degradation()
                # Reset half-cycle history
                self.half_cycle_history = np.array([self.previous_ch])
                # Change mode to discharge
                self.operation_mode = 'discharge'
        else: # Both are zeros
            # Continue battery half-cycle
            self.half_cycle_history = np.append(self.half_cycle_history, self.ch)
            # Calculate degradation
            aging_factor_cyc = 0.0

        if include_float_degradation:
            aging_factor_float = self.float_degradation()
        else:
            aging_factor_float = 0
        aging_factor = max(aging_factor_cyc, aging_factor_float)
        self.aging_factor_history = np.append(self.aging_factor_history, aging_factor)
        self.aging_factor_cyclic_history = np.append(self.aging_factor_cyclic_history, aging_factor_cyc)
        self.aging_factor_float_history = np.append(self.aging_factor_float_history, aging_factor_float)

        if self.include_degradation:
            self.C = self.C * (1 - aging_factor * (1 - self.END_OF_LIFE))
        
        delta_SOH = (self.capacity_history[-1] - self.C)/13.5/0.2
        self.deltaSOH_history = np.append(self.deltaSOH_history, delta_SOH)

        if aging_factor_float <= aging_factor_cyc:
            self.deltaSOH_float_history = np.append(self.deltaSOH_float_history, 0)
            self.deltaSOH_cyclic_history = np.append(self.deltaSOH_cyclic_history, delta_SOH)
        else:
            self.deltaSOH_float_history = np.append(self.deltaSOH_float_history, delta_SOH)
            self.deltaSOH_cyclic_history = np.append(self.deltaSOH_cyclic_history, 0)

            
