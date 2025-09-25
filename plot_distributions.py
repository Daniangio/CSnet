import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def extract_chainid_resnum_resname(df):
    df[['chainid', 'resnum', 'resname', 'atomname']] = df['atom_fullname'].str.split('.', expand=True)
    return df

def plot_distributions(*dfs, atom_types=None):
    if atom_types is None:
        atom_types = [None] * len(dfs)
    
    filtered_dfs = []
    for df, atom_type in zip(dfs, atom_types):
        df = extract_chainid_resnum_resname(df)
        if atom_type is not None:
            filtered_dfs.append(df[df['node_type'] == atom_type])
        else:
            filtered_dfs.append(df)
    
    if len(filtered_dfs) == 1:
        df = filtered_dfs[0]
        groups = df.groupby(['chainid', 'resnum', 'resname'])
        colors = plt.cm.get_cmap('tab10', len(groups))
        plt.figure()
        for i, (label, group) in enumerate(groups):
            plt.hist(group['pred'], bins=30, alpha=0.5, label=f'{label}', color=colors(i))
        plt.xlabel('pred')
        plt.ylabel('Frequency')
        plt.title('1D Distribution of pred')
        plt.legend()
        plt.show()
    
    elif len(filtered_dfs) == 2:
        df1, df2 = filtered_dfs
        df1 = df1.sort_values(by=['batch', 'chainid', 'resnum', 'resname'])
        df2 = df2.sort_values(by=['batch', 'chainid', 'resnum', 'resname'])
        merged_df = pd.merge_ordered(df1, df2, on=['batch', 'chainid', 'resnum', 'resname'], suffixes=('_df1', '_df2'))
        
        groups = merged_df.groupby(['chainid', 'resnum', 'resname'])
        colors = plt.cm.get_cmap('tab10', len(groups))
        plt.figure()
        for i, (label, group) in enumerate(groups):
            x = group['pred_df1']
            y = group['pred_df2']
            plt.scatter(x, y, alpha=0.5, label=f'{label}', color=colors(i))

            xy = np.vstack([x, y])
            kde = gaussian_kde(xy)
            
            xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
            zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
            
            levels = np.linspace(zi.min(), zi.max(), 20)
            for j, level in enumerate(levels):
                # alpha = 1.0 if j >= len(levels) - 5 else 0.1
                alpha = 0.1 + j * 0.9 / len(levels)
                color = 'black' if j >= len(levels) - 2 else colors(i)
                plt.contour(xi, yi, zi.reshape(xi.shape), levels=[level], colors=[color], alpha=alpha)

        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()

        plt.xlabel('pred (df1)')
        plt.ylabel('pred (df2)')
        plt.title('2D Distribution of pred')
        plt.legend()
        plt.show()
    else:
        raise ValueError("The function supports only 1 or 2 dataframes.")

# Example usage:
# df1 = pd.DataFrame(...)
# df2 = pd.DataFrame(...)
# plot_distributions(df1, df2, atom_types=[7, 12])
