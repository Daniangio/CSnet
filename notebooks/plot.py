import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

def gaussian(x, mu, sigma, A):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

# Plotting
def plot(names, chemical_shifts, title, figsize=(10,8)):
    plt.figure(figsize=figsize)
    c = list(mcolors.TABLEAU_COLORS.keys())
    for i, (name, y) in enumerate(zip(names, chemical_shifts.T)):
        x = np.arange(len(y))
        plt.plot(x, y, label=name, color=c[i])

        # coefficients = np.polyfit(x, y, 3)
        # poly_function = np.poly1d(coefficients)
        # plt.plot(x, poly_function(x), linestyle='--', color=c[i], label='Gaussian Approximation')
        plt.axhline(y.mean(), xmin=0., xmax=1., linestyle='dashdot', color=c[i], label='Mean')

    # Adding labels and legend
    plt.xlabel('Time')
    plt.ylabel('Chemical Shift')
    plt.title(title)
    plt.legend()

    # Display the plot
    plt.show()

def plot_inference(atom_names, pred, target, figsize=(10,8)):
    plt.figure(figsize=figsize)

    inds =  np.arange(1, pred.shape[-1] + 1)
    plt.scatter(inds, pred, marker='x', color='black', s=30, zorder=3)
    plt.scatter(inds, target, marker='o', color='red', s=30, zorder=3)

    plt.xticks(np.arange(1, len(atom_names) + 1), labels=atom_names)
    plt.xlim(0.25, len(atom_names) + 0.75)
    plt.xlabel('Atom')
    plt.legend(['pred', 'target'])

    # Display the plot
    plt.show()

def plot_violin(atom_names, data_list, ds_names, title, figsize=None, overlap_same: bool = False, ylim=None):
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
        '#4053D3',
        '#DDB310',
        '#B51D14',
        '#00BEFF',
        '#FB49B0',
        '#00B25D',
        '#CACACA',
    ])
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()

    labels = []
    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))
        return violin
    
    for i, (ds_name, data) in enumerate(zip(ds_names, data_list)):
        if overlap_same:
            violins = add_label(plt.violinplot(data, vert=True, showmeans=False, showextrema=False, showmedians=False), ds_name)
        else:
            positions = [(i+1) + (x * len(data_list)) for x in range(data.shape[-1])]
            violins = add_label(plt.violinplot(data, positions=positions, vert=True, showmeans=False, showextrema=False, showmedians=False), ds_name)
        for pc in violins['bodies']:
            pc.set_edgecolor('black')
            pc.set_alpha(.5)
        
        quartile1, medians, quartile3 = np.percentile(data.T, [25, 50, 75], axis=1)

        def adjacent_values(vals, q1, q3):
            upper_adjacent_value = q3 + (q3 - q1) * 1.5
            upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

            lower_adjacent_value = q1 - (q3 - q1) * 1.5
            lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
            return lower_adjacent_value, upper_adjacent_value
        
        whiskers = np.array([
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data.T, quartile1, quartile3)])
        whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

        inds = inds = np.arange(1, len(medians) + 1) if overlap_same else positions
        plt.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
        plt.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        plt.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    # Adding labels and legend
    # set style for the axes
    if overlap_same:
        plt.xticks(np.arange(1, len(atom_names) + 1), labels=atom_names, rotation=90)
        plt.xlim(0.25, len(atom_names) + 0.75)
    else:
        plt.xticks(np.arange(1, (i +  1) * len(atom_names) + 1), labels=np.repeat(atom_names, (i + 1)), rotation=90)
        plt.xlim(0.25, (i +  1) * len(atom_names) + 0.75)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Atom')
    plt.legend(*zip(*labels))
    
    plt.title(title)
    if figsize is None:
        fig.set_size_inches(len(atom_names), 8, forward=True)
        fig.set_dpi(100)

    # Display the plot
    plt.show()

def plot_distribution(atom_names, data_list, ds_names, title, figsize=(10,8), overlap_same: bool = False):
    from scipy.stats import gaussian_kde

    mydict = {
        'cs_FAN.npy': 'active',
        'cs_pAs.npy': 'pAs',
        'cs_INzma.npy': 'inactive',
    }

    color=[
        '#B51D14',
        '#DDB310',
        '#4053D3',
        '#00BEFF',
        '#FB49B0',
        '#00B25D',
        '#CACACA',
    ]
    plt.figure(figsize=figsize)
    
    data_ref = data_list[0]
    data_ref=data_ref.flatten()
    kde = gaussian_kde(data_ref)
    x_vals = np.linspace(min(data_ref), max(data_ref), 1000)
    y_vals = kde(x_vals)
    shift = x_vals[np.argmax(y_vals)]
    for i, (ds_name, data) in enumerate(zip(ds_names, data_list)):
        data=data.flatten()
        kde = gaussian_kde(data)
        x_vals = np.linspace(min(data), max(data), 1000)
        y_vals = kde(x_vals)

        plt.plot(x_vals - shift, y_vals, color=color[i], label=mydict.get(ds_name, ds_name))
        plt.vlines(x_vals[np.argmax(y_vals)] - shift, 0, np.max(y_vals), color=color[i], linestyle='--', lw=1)
    plt.xlabel('Atom')
    
    plt.xlabel(r'$\Delta$ppm')
    plt.title(title)
    plt.legend()
    plt.gca().invert_xaxis()
    plt.xticks([-4., -3., -2., -1., 0., 1.])
    plt.yticks([])
    # plt.box(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # Display the plot
    plt.show()

def plot_violindist(data_list, ds_names, title, figsize=(10,8), center=True):
    from scipy.stats import gaussian_kde

    mydict = {
        'cs_FAN.npy': 'active',
        'cs_pAs.npy': 'pAs',
        'cs_INzma.npy': 'inactive',
    }

    color=[
        '#B51D14',
        '#DDB310',
        '#4053D3',
        '#00BEFF',
        '#FB49B0',
        '#00B25D',
        '#CACACA',
    ]

    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color)

    fig = plt.figure(figsize=figsize)
    
    data_ref = data_list[0]
    data_ref=data_ref.flatten()
    kde = gaussian_kde(data_ref)
    x_vals = np.linspace(min(data_ref), max(data_ref), 1000)
    y_vals = kde(x_vals)
    if center:
        shift = x_vals[np.argmax(y_vals)]
    else:
        shift = 0

    labels = []
    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), label))
        return violin

    y_vals_sum = []
    for i, (ds_name, data) in enumerate(zip(ds_names, data_list)):
        data=data.flatten()
        kde = gaussian_kde(data)
        x_vals = np.linspace(min(data), max(data), 1000)
        y_vals = kde(x_vals)

        positions = [(0.6*i+1)]
        violins = add_label(plt.violinplot(data - shift, positions=positions, vert=False, showmeans=False, showextrema=False, showmedians=False), ds_name)

        for pc in violins['bodies']:
            pc.set_edgecolor('black')
            pc.set_alpha(.5)
        
        quartile1, medians, quartile3 = np.percentile(data - shift, [25, 50, 75], axis=0)

        def adjacent_values(vals, q1, q3):
            upper_adjacent_value = q3 + (q3 - q1) * 1.5
            upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

            lower_adjacent_value = q1 - (q3 - q1) * 1.5
            lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
            return lower_adjacent_value, upper_adjacent_value
        
        whiskers = np.array([
            adjacent_values(data, quartile1, quartile3)
        ])
        whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

        plt.scatter(x_vals[np.argmax(y_vals)] - shift, positions, marker='o', color='white', s=30, zorder=3)
        plt.hlines(positions, quartile1, quartile3, color='k', linestyle='-', lw=5)
        plt.hlines(positions, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

        plt.plot(x_vals - shift, y_vals, color=color[i], label=mydict.get(ds_name, ds_name))
        plt.vlines(x_vals[np.argmax(y_vals)] - shift, 0, (0.6*i+1), color=color[i], linestyle='--', lw=1)
        y_vals_sum.append(y_vals)
    
    y_vals_sum = np.stack(y_vals_sum, axis=0).sum(axis=0)
    plt.plot(x_vals - shift, y_vals_sum, color='black', label='simulated spectrum', linestyle='--', lw=2)
    
    plt.xlabel(r'$\Delta$ppm')
    plt.title(title)
    plt.legend()
    plt.gca().invert_xaxis()
    if center:
        plt.xlim(1.5, -4.)
        plt.xticks([-4., -3., -2., -1., 0., 1.])
    plt.yticks([])
    # plt.box(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # Display the plot
    plt.show()

    # fig.savefig('../media/violin_dist_val_225_CG1.svg')