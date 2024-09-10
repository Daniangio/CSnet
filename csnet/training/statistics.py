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
    print(f"{len(filenames)} files analysed.")
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

def plot_distribution(statistics: dict, resname: str, atomname: str, src: str = 'true'):
    try:
        index = DataDict.get_atom_type(resname, atomname)
        data = statistics[index]
        if len(data) == 0:
            return
        fig = px.histogram(
            data,
            x='value',
            color='filename',
            nbins=100,
            title=f"{resname}-{atomname}",
        )
        fig.show()
        fig.write_html(f"../media/{resname}-{atomname}-{src}.html")
    except:
        pass

def build_config_params(statistics: dict, config_root: str):
    txt = 'type_names:\n'
    type_names = ['-'.join(k) if isinstance(k, tuple) else k for k in DataDict.CHEMICAL_SPECIES2ATOM_TYPE_LIST()]
    for type_name in type_names:
        txt += f'  - {type_name}\n'

    feat_stats = []
    for index in range(len(DataDict.CHEMICAL_SPECIES2ATOM_TYPE_LIST())):
        try:
            avg = statistics[index].value.mean()
            std = statistics[index].value.std()
        except:
            avg = 0.
            std = 0.
        feat_stats.append([avg, std])
    feat_stats=np.nan_to_num(np.array(feat_stats))

    txt += '\n\nper_type_bias:\n'
    for avg, type_name in zip(feat_stats[:, 0], type_names):
        txt += f"  - {avg: <20} # {type_name}\n"
    
    txt += '\n\nper_type_std:\n'
    for std, type_name in zip(feat_stats[:, 1], type_names):
        txt += f"  - {std: <20} # {type_name}\n"
     
    with open(os.path.join(config_root, 'per_type_stats.yaml'), 'w') as f:
        f.write(txt)