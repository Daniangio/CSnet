import os
import glob
import numpy as np
import pandas as pd
import plotly.express as px

from os.path import basename
from csnet.utils import DataDict


def get_npz_statistics(
    npz_folder: str,
    feat: str,
    type: str,
):
    filenames = []
    all_x = []
    all_idcs = []
    for filename in glob.glob(os.path.join(npz_folder, "*.npz")):
        filenames.append('.'.join(basename(filename).split('.')[:-1]))
        ds = np.load(filename, allow_pickle=True)
        x = ds[feat]
        batches = len(x)
        x = x.flatten()
        idcs = np.repeat(ds[type], batches)
        all_x.append(x)
        all_idcs.append(idcs)

    df_dict: dict[str, pd.DataFrame] = {}
    for filename, x, idcs in zip(filenames, all_x, all_idcs):
        for index in np.unique(idcs):
            df: pd.DataFrame = df_dict.get(index, None)
            fltr = idcs==index
            value = x[fltr]
            nan_fltr = ~np.isnan(value)
            update_df = pd.DataFrame(data={
                'filename': [filename] * sum(nan_fltr),
                'value': value[nan_fltr],
                })
            if df is None:
                df = update_df
            else:
                df = pd.concat([df, update_df], ignore_index=True)
            df_dict[index] = df
    
    return df_dict

def plot_distribution(statistics: dict, index: int):
    fig = px.histogram(statistics[index], x='value', color='filename', nbins=100)
    fig.show()

def build_config_params(statistics: dict, config_root: str):
    txt = 'type_names:\n'
    type_names = ['-'.join(k) if isinstance(k, tuple) else k for k in DataDict.CHEMICAL_SPECIES2ATOM_TYPE_LIST()]
    for type_name in type_names:
        txt += f'  - {type_name}\n'

    with open(os.path.join(config_root, 'type_names.yaml'), 'w') as f:
        f.write(txt)

    avg_feat_values = []
    for index in range(len(DataDict.CHEMICAL_SPECIES2ATOM_TYPE_LIST())):
        try:
            avg = statistics[index].value.mean()
        except:
            avg = 0.
        avg_feat_values.append(avg)
    avg_feat_values=np.nan_to_num(np.array(avg_feat_values))

    txt = 'per_species_bias:\n'
    for avg_feat_value, type_name in zip(avg_feat_values, type_names):
        txt += f"  - {avg_feat_value: <20} # {type_name}\n"
        #txt += f'  - {}   #{type_name}\n'
    with open(os.path.join(config_root, 'per_species_bias.yaml'), 'w') as f:
        f.write(txt)