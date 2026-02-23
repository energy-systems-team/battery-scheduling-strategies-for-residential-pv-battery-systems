import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import datetime


MARGIN = 0.4 # [c/kWh]
# Transmission + electricity tax (inc. VAT) [2018, 2019, 2020, 2021, 2022, 2023]
FIXED_COSTS = [5.69, 5.90, 5.90, 5.44, 5.01, 5.01] # c/kWh


def get_monthly_profiles(data_input, years, var_names):
    """
    Function for grouping data to hours and months.

    Inputs:
        data_input: Pandas dataframe object with datetimeindex in hourly resolution
        years: List containing years from which data will be used. 
               If many years are given, an average over all years are plotted
        var_names: List of column names of data_input that will be included in the aggregation

    Output:
        a tuple of the following variables:
        hourly_mean_data: 2D numpy array (288 x number of input variables) containing hourly values over all the months 
        var_names: names of the columns of the input data that are used
    """

    data_by_year = dict()
    for year in years:
        data_y = data_input[data_input.index.year == year]
        data_by_month = dict()
        for month in data_y.index.month.unique():
            data = data_y[data_y.index.month == month]
            data_by_month[month] = data.groupby(data.index.hour)[var_names].mean()    
        data_by_year[year] = data_by_month
    data_all = []
    for year_no, year in data_by_year.items():
        data_for_year = []
        for month in year.values():
            data_for_year = data_for_year + list(month.values)
        data_all.append(data_for_year)
    
    hourly_mean_data = np.array(data_all).mean(axis=0)
    return hourly_mean_data, var_names


def plot_monthly_profiles(ax, fig, data_input, years, var_names, offset=9, offsety_scale=1, colors=[], linestyle='solid', linewidth=1.):
    """
    Function for plotting the average daily profiles.

    Inputs:
        ax: Matplotlib Axes object in which the data is plotted
        fig: Matplotlib Figure object
        data_input: Pandas dataframe object with datetimeindex in hourly resolution
        years: List containing years from which data will be used. 
               If many years are given, an average over all years are plotted
        var_names: List of column names of data_input that will be included in the aggregation
        offset: A float value for changing the x-position the month labels
        offsety_scale: A float value for changing the y-position the month labels
        colors: List containing colors for each variable
        linestyle: (str) "solid", "dotted", or "dashed". Linestyle for the line
        linewidth: (float) width of the line

    Output:
        ax: Matplotlib Axes object where the data was plotted
    """

    # Get the monthly data and variable names
    data, var_names = get_monthly_profiles(data_input, years, var_names)

    # Iterate through every variable in the input data
    for i in range(data.shape[1]):
        if len(colors) > 0:
            ax.plot(data[:, i], label=var_names[i], c=colors[i], linestyle=linestyle, linewidth=linewidth)
        else:
            ax.plot(data[:, i], label=var_names[i], linestyle=linestyle, linewidth=linewidth)

    # Plot vertical lines separating each month
    ylims = ax.get_ylim()
    ax.vlines(np.arange(0, 24*12, 24), ylims[0], ylims[1], colors='black', alpha=0.3, linewidth=0.4)

    # Set xlabels (hours)
    xticks = list(range(0,12*24,8))
    ax.set_xticks(xticks)
    ax.set_xticklabels((list(range(0,24))*12)[::8], fontsize=7, rotation=90)
    ax.tick_params(axis='x', length=2)

    ax.legend(loc='upper right', ncol=data.shape[1])

    # Add months to figure
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Position month labels to data. Use offset inputs to fine tune the text positions.
    y_ran = ax.get_ylim()
    for i in range(len(months)):
        if offsety_scale == -1:
            continue
        y_offset = y_ran[0] - (y_ran[1]-y_ran[0])/8*offsety_scale
        ax.text(xticks[::3][i]+offset, y_offset, months[i])

    plt.tight_layout()

    return ax


def plot_monthly_profiles_windowed(data_input, years, var_names, offset=9, offsety_scale=0, colors=[], figsize=(2244/300, 2000/300), fontsize=10, tickfontsize=7):
    """
    Function for plotting the average daily profiles as a windowed figure.

    Inputs:
        data_input: Pandas dataframe object with datetimeindex in hourly resolution
        years: List containing years from which data will be used. 
               If many years are given, an average over all years are plotted
        var_names: List of column names of data_input that will be included in the aggregation
        offset: A float value for changing the x-position the month labels
        offsety_scale: A float value for changing the y-position the month labels
        colors: List containing colors for each variable
        figsize: tuple indicating the pixels

    Output:
        a tuple of the following variables:
        ax: Matplotlib Axes object where the data was plotted
        fig: Matplotlib Figure object
    """

    # Get the monthly data and variable names
    data, var_names = get_monthly_profiles(data_input, years, var_names)
    data_windowed = []

    for i in range(12):
        if i > 0:
            data_windowed.append(data[i*24-1:i*24+24,:])
        else:    
            data_windowed.append(data[i*24:i*24+24,:])

    # Plot profiles
    plt.rc('font', size=fontsize)
    
    fig, ax = plt.subplots(3,4,figsize=figsize, sharex=True, sharey=True, gridspec_kw = {'wspace':0, 'hspace':.1})
    
    xticks = list(range(0,12*24,4))

    j = 0
    for a in [ax[0,0], ax[0,1], ax[0,2], ax[0,3], ax[1,0], ax[1,1], ax[1,2], ax[1,3], ax[2,0], ax[2,1], ax[2,2], ax[2,3]]:
        d = data_windowed[j]
        a.set_xticks(xticks)
        a.set_xticklabels((list(range(0,24))*12)[::4], fontsize=tickfontsize, rotation=0)

        j += 1
        for i in range(0,d.shape[1]):
            a.plot(d[:, i], label=var_names[i]
                    , c=colors[i]
                    )

    ax[0,1].legend(loc='lower center', ncol=data.shape[1], bbox_to_anchor=(1,1))
    plt.tight_layout()

    return ax, fig


def _plot_func(axes, power, load, pv_to_house, pv_to_bat, pv_to_grid, bat_to_house, bat_to_grid, grid_to_house, grid_to_bat, pv_to_bat_grid_wasted, 
               pv_to_grid_wasted, to_house_wasted, spot, soc, datetime, detailed, axis_position=40, net_imports=False):
    if detailed:
        ax0, ax02, ax1, ax12, ax13 = axes[0], axes[1], axes[2], axes[3], axes[4]
    else:
        ax0, ax02, ax13 = axes[0], axes[1], axes[-1]

    datetime = datetime.values

    #***************************************
    # Plot simpler plot
    #***************************************
    if not detailed:
        # Exports
        exports = np.zeros(len(power))
        if pv_to_grid is not None:
            exports = exports + pv_to_grid.values
        if bat_to_grid is not None:
            exports = exports + bat_to_grid.values
        
        # Imports
        imports = np.zeros(len(power))
        if grid_to_house is not None:
            imports = imports + grid_to_house.values
        if grid_to_bat is not None:
            imports = imports + grid_to_bat.values

        if net_imports:
            ax0.plot(datetime, imports-exports, label='Net imports')
            # Add dashed line to see 0-point
            ax0.plot(datetime, np.zeros(datetime.shape), color='black', linestyle='dotted', linewidth=0.5, alpha=0.8)
        else:
            ax0.plot(datetime, -exports, label='Exports')
            ax0.plot(datetime, imports, label='Imports')

        if pv_to_bat is not None:
            pv_to_bat = pv_to_bat.values

        if bat_to_house is not None:
            bat_to_house = bat_to_house.values

    #***************************************
    # Plot detailed plot
    #***************************************
    if detailed:
        # Check and plot each array if it is not None
        if pv_to_house is not None:
            pv_to_house = pv_to_house.values
            ax0.plot(datetime, pv_to_house, label=r'$\mathrm{PV\rightarrow H}$', color='#ff0000')

        if pv_to_bat is not None:
            pv_to_bat = pv_to_bat.values
            ax0.plot(datetime, pv_to_bat, label=r'$\mathrm{PV\rightarrow B}$', color='orange')

        if pv_to_bat_grid_wasted is not None:
            pv_to_bat_grid_wasted = pv_to_bat_grid_wasted.values
            ax0.plot(datetime, pv_to_bat_grid_wasted, label=r'$\mathrm{PV_{wasted}}$', linestyle= '-.', color='#cc9900')

        if pv_to_grid is not None:
            pv_to_grid = pv_to_grid.values
            ax0.plot(datetime, pv_to_grid, label=r'$\mathrm{PV\rightarrow G}$', color='#ff8080')

        if bat_to_house is not None:
            bat_to_house = bat_to_house.values
            ax1.plot(datetime, bat_to_house, label=r'$\mathrm{B\rightarrow H}$', color='#33cc33')

        if bat_to_grid is not None:
            bat_to_grid = bat_to_grid.values
            ax1.plot(datetime, bat_to_grid, label=r'$\mathrm{B\rightarrow G}$', color='#009933')

        if grid_to_house is not None:
            grid_to_house = grid_to_house.values
            ax1.plot(datetime, grid_to_house, label=r'$\mathrm{G\rightarrow H}$', color='#005ce6')

        if grid_to_bat is not None:
            grid_to_bat = grid_to_bat.values
            ax1.plot(datetime, grid_to_bat, label=r'$\mathrm{G\rightarrow B}$', 
                    color='#d9b3ff')

    consumption = load
    production = power

    #*****************************************************
    # Plot component common for detailed and simple plot
    #*****************************************************
    if soc is not None:
        soc = soc.values
        ax02.plot(datetime, soc, label='Bat. charge', color='#732673', linestyle='--')
        ax02.set_ylim(0,14)
        ax02.tick_params(axis='y', length=2)
        if detailed:
            ax12.plot(datetime, soc, label='Bat. charge', color='#732673', linestyle='--')
            ax12.set_ylim(ax1.get_ylim()[0],14)
            ax12.tick_params(axis='y', length=2)
    if spot is not None:
        spot = spot.values
        ax13.plot(datetime, spot, label='Spot', color='grey', linestyle=':')
        ax13.tick_params(axis='y', length=2)

    ax0.plot(datetime, production, label='PV', color='#e6e600')
    ax0.plot(datetime, consumption, label='Load', color='brown'
    )
    ax0.tick_params(axis='y', length=2)
    if detailed:
        ax1.plot(datetime, consumption, label='Load', color='brown'
        )
        ax1.set_xlim(datetime[0], datetime[-1])
        ax1.tick_params(axis='y', length=2)
    
    ax0.set_xlim(datetime[0], datetime[-1])

    if (soc is not None) and (spot is not None):
        ax13.spines['right'].set_position(('outward', axis_position))

    # Set labels, legend, and show the plot
    if detailed:
        ax1.set_xlabel('Hour')
        ax1.set_ylabel('Power (kWh/h)')
    else:
        ax0.set_xlabel('Hour')
    ax0.set_ylabel('Power (kWh/h)')
    if soc is not None:
        ax02.set_ylabel('Bat. charge (kWh)')

    if soc is not None:
        if detailed:
            ax12.set_ylabel('Bat. charge (kWh)')
        if spot is not None:
            ax13.set_ylabel('Spot (c/kWh)')

    if soc is None:
        ax02.set_yticks([])
        if detailed:
            ax12.set_yticks([])

    items = []
    labels = []
    h0, l0 = ax0.get_legend_handles_labels()
    items += h0
    labels += l0
    h02, l02 = ax02.get_legend_handles_labels()
    items += h02
    labels += l02
    if spot is not None:
        h13, l13 = ax13.get_legend_handles_labels()
        items += h13
        labels += l13

    if not detailed:
        ax0.legend(items, labels, ncol=len(items), loc='lower left', bbox_to_anchor=(-0.1,1), handlelength=0.5, columnspacing=0.4)
    else:
        ax0.legend(h0+h02, l0+l02, ncol=len(h0+h02), loc='lower left', bbox_to_anchor=(-0.1,1), handlelength=0.5, columnspacing=0.4)
    ax0.grid(axis='x', alpha=0.3)

    # Format primary x-axis ticks with hourly resolution
    ax0.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

    if detailed:
        h1, l1 = ax1.get_legend_handles_labels()
        h12, l12 = ax12.get_legend_handles_labels()

        ax1.legend(h1+h12+h13, l1+l12+l13, ncol=len(h1+h12+h13), loc='lower left', bbox_to_anchor=(0,1), handlelength=1, columnspacing=0.4)
        ax1.grid(axis='x', alpha=0.3)

        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

        for a, letter in list(zip([ax0, ax1], 
                                ['A', 'B'])):
            x0, xmax = a.set_xlim()
            y0, ymax = a.set_ylim()
            data_width = xmax - x0
            data_height = ymax - y0
            a.text(x0-(0.03*data_width), (y0 + 1*data_height), letter, weight='bold', fontsize=15, va='bottom', ha='right')

    return ax0


def _check_if_column_in_data(data, column):
    try:
        return data[column]
    except KeyError:
        return None


def plot_days(path, day, days, variables, title, save_name, save=False, detailed=True, figsize=None, axis_position=40, net_imports=False, title_y=1):

    year = day.year
    data = pd.read_csv(path, parse_dates=['datetime'])

    # Select variables to be used
    data = data[variables]

    start = day
    end = day + datetime.timedelta(days=days)
    data = data[(data.datetime>=start)&(data.datetime<=end)]

    if detailed:
        if figsize is None:
            figsize=(2244/300, 1500/300)
        fig, (ax0, ax1) = plt.subplots(2,1, figsize=figsize, dpi=300, sharex=False, sharey=True)
        ax02 = ax0.twinx()
        ax12 = ax1.twinx()
        ax13 = ax1.twinx()
    else:
        if figsize is None:
            figsize=(2244/300, 900/300)
        fig, ax0 = plt.subplots(1,1, figsize=figsize, dpi=300)
        ax02 = ax0.twinx()
        if "spot" in variables:
            ax13 = ax0.twinx()
        else:
            ax13 = None
        ax1, ax12 = None, None
    
    axes = [ax0, ax02, ax1, ax12, ax13]

    power = data['power']
    load = data['load']
    pv_to_house = _check_if_column_in_data(data, 'pv_to_house')
    pv_to_bat = _check_if_column_in_data(data, 'pv_to_bat')
    pv_to_grid = _check_if_column_in_data(data, 'pv_to_grid')
    bat_to_house = _check_if_column_in_data(data, 'bat_to_house')
    bat_to_grid = _check_if_column_in_data(data, 'bat_to_grid')
    grid_to_house = _check_if_column_in_data(data, 'grid_to_house')
    grid_to_bat = _check_if_column_in_data(data, 'grid_to_bat')
    pv_to_bat_grid_wasted = _check_if_column_in_data(data, 'pv_to_bat_grid_wasted')
    pv_to_grid_wasted = _check_if_column_in_data(data, 'pv_to_grid_wasted')
    to_house_wasted = _check_if_column_in_data(data, 'to_house_wasted')
    spot = _check_if_column_in_data(data, 'spot')
    soc = _check_if_column_in_data(data, 'soc')
    dates = _check_if_column_in_data(data, 'datetime')

    FIXED_COST = FIXED_COSTS[year-2018]
    try:
        print('Cost from the period:', (grid_to_house*(spot*1.24+FIXED_COST+MARGIN)-pv_to_grid*(spot-MARGIN)).sum()/100)
    except:
        pass
    ax = _plot_func(axes, power, load, pv_to_house, pv_to_bat, pv_to_grid, bat_to_house, bat_to_grid, grid_to_house, grid_to_bat, 
               pv_to_bat_grid_wasted, pv_to_grid_wasted, to_house_wasted, spot, soc, dates, detailed, axis_position, net_imports)
    fig.suptitle(f'{title}', weight='bold', y=title_y)
    plt.tight_layout()
    if save:
        save_path = os.path.join('..', '..', 'Images', 'Ennusteet', save_name)
        plt.savefig(save_path, dpi=300)
    
    return ax, fig


if __name__ == "__main__":
    # Demonstration

    import pandas as pd
    import numpy as np
    from math import pi

    daterange = pd.date_range(start='1/1/2019', end='1/1/2021', freq='1h')
    x1 = np.sin(np.arange(len(daterange))*pi/12)+(np.arange(len(daterange))/10000)**2
    x2 = np.cos(np.arange(len(daterange))*pi/12)+(np.arange(len(daterange))/10000)**2
    data = pd.DataFrame(data=np.array([x1, x2]).T, index=daterange, columns=['sin', 'cos'])
    
    fig, ax = plt.subplots(3, 1, figsize=(10,10))

    ax[0] = plot_monthly_profiles(ax=ax[0], fig=fig, data_input=data, years=[2019], var_names=['sin'])
    ax[1] = plot_monthly_profiles(ax=ax[1], fig=fig, data_input=data, years=[2020], var_names=['cos'])
    ax[2] = plot_monthly_profiles(ax=ax[2], fig=fig, data_input=data, years=[2019, 2020], var_names=['sin', 'cos'])
    plt.show()

    fig, ax = plot_monthly_profiles_windowed(data_input=data, years=[2019], var_names=['sin', 'cos'], colors=['red', 'blue'])
    plt.show()
    

