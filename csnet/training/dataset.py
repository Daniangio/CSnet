import os
import re
import glob
import sys
import torch
from pathlib import Path
from tqdm.autonotebook import tqdm

from csnet.utils import DataDict
sys.path.append('..')

import pandas as pd
import numpy as np
import MDAnalysis as mda

from typing import Dict, Generator
from os.path import basename
from csnet.training.nmr import NMR, JOIN_CHAR
from geqtrain.data import AtomicDataDict
from geqtrain.scripts.evaluate import load_model


ADJUST_ATOM_NAMES = {
    'HA1': 'HA3',
    'HB1': 'HB3',
}


def build_dataset(
    pdb_input_folder: str,
    cs_input_folder: str,
    output_folder: str,
):
    print(f"Building dataset info...")
    df = build_info(
        pdb_input_folder=pdb_input_folder,
        cs_input_folder=cs_input_folder,
    )
    # df = df[(df['exp']=='X-RAY') & (df['temp']==298) & (df['ph']>=6.3)  & (df['ph']<=6.7)]
    
    os.makedirs(output_folder, exist_ok=True)
    for index, row in df.iterrows():
        pdb_filename = row.pdb
        cs_filenames = row.cs
        npz_file = os.path.join(output_folder, f"{os.path.basename(pdb_filename).split('.')[0]}.npz")
        if os.path.isfile(npz_file):
            continue

        print(f"Building dataset for {pdb_filename}")
        for ds in get_dataset(
            pdb_filename,
            cs_filenames,
            selection="protein",
        ):
            if ds is not None:
                ds.update({
                    "temp": np.array([row.temp]),
                    "ph": np.array([row.ph]),
                })
                np.savez(npz_file, **ds)
                print(f"{npz_file} dataset saved.")

def build_info(
    pdb_input_folder: str,
    cs_input_folder: str,
):
    template_exp = re.compile(r'(EXPERIMENT TYPE)(\D+):[ \t]+([^ \t]+)', re.I)
    template_ph = re.compile(r'( PH )(\D+)(\d+(?:\.\d+)?)', re.I)
    template_temp = re.compile(r'(TEMPERATURE)(\D+)(\d+)', re.I)

    cs_files = list(glob.glob(f'{cs_input_folder}/*.corr*', recursive=True))
    pdb_codes = [os.path.basename(cs_file)[:4] for cs_file in cs_files]

    records = []
    for pdb_file in glob.glob(f'{pdb_input_folder}/*.pdb*', recursive=True):
        exp = None
        ph = None
        temp = None
        with open(pdb_file, 'r') as f:
            for line in f.readlines():
                match_exp = template_exp.findall(line)
                if len(match_exp) > 0:
                    exp = match_exp[0][-1]
                match_ph = template_ph.findall(line)
                if len(match_ph) > 0:
                    ph = float(match_ph[0][-1])
                match_temp = template_temp.findall(line)
                if len(match_temp) > 0:
                    temp = int(match_temp[0][-1])
                if exp is not None and ph is not None and temp is not None:
                    break

        cs = [cs_files[i] for i,s in enumerate(pdb_codes) if os.path.basename(pdb_file).startswith(s)]

        records.append({
            'pdb': pdb_file,
            'cs': cs,
            'exp': exp,
            'ph': ph,
            'temp': temp,
        })

    info_df = pd.DataFrame.from_records(records)
    info_df.to_csv(os.path.join(pdb_input_folder, "info.csv"))
    return info_df

def fix_atom_naming(name):
    if name[0].isdigit():
        name = name[1:] + name[0]
    return ADJUST_ATOM_NAMES.get(name, name)

def get_atom_idcs_with_high_rmsf(coords: np.ndarray, heavy_atoms_idcs: np.ndarray, protein, rmsf_threshold: float):
    if len(coords) == 1:
        return None
    ref_structure = coords[0][heavy_atoms_idcs]
    all_SD = np.power(coords[1:, heavy_atoms_idcs] - ref_structure[None, ...], 2).sum(axis=-1)

    residcs = []
    for resid, r in enumerate(protein.residues):
        residcs.extend([resid] * r.n_atoms)
    residcs = np.array(residcs)[heavy_atoms_idcs]

    RMSF = np.zeros((len(all_SD), residcs.max() + 1,), dtype=np.float32)
    RMSF[:, residcs] += all_SD

    rmsf_residues_is_high = np.max(RMSF > rmsf_threshold, axis=0)
    high_rmsf_atom_idcs = []
    for resid, (r, discard) in enumerate(zip(protein.residues, rmsf_residues_is_high)):
        high_rmsf_atom_idcs.extend([discard] * r.n_atoms)
    return np.array(high_rmsf_atom_idcs)

def get_dataset(
        pdb_file,
        nmr_files,
        selection: str = "protein",
        rmsf_threshold: float = 1.5
    ) -> Generator[Dict[str, np.ndarray], None, None]:
    ds, mol = get_structure(pdb_file, selection=selection)

    # Build DataFrame from nmr file(s)
    chain = ds.get("atom_chains")[0]
    pdb_code = basename(pdb_file).split('.')[0].split(JOIN_CHAR)[1]
    if len(pdb_code) == 5:
        chain = pdb_code[-1]
    nmr_dfs = []
    for nmr_file in nmr_files:
        try:
            nmr = NMR(nmr_file, chain=chain)
        except Exception as e:
            raise Exception(f"Could not read NMR file {nmr_file}.") from e
        cs_df = nmr.atom_chem_shift_df
        if cs_df is not None:
            nmr_dfs.append(cs_df)
    
    if len(nmr_dfs) == 0:
        raise Exception(f"Found no nmr chemical shift values in all nmr files [{','.join(nmr_files)}] for pdb {pdb_file}")
    
    nmr_df = pd.concat(nmr_dfs)

    # If there are multiple assignments to the same atom, take the mean
    mean_df = nmr_df.groupby('Atom_fullname', as_index=False).mean(numeric_only=True)
    std_df = nmr_df.groupby('Atom_fullname', as_index=False).std(numeric_only=True)

    mean_std_df = mean_df.merge(std_df, on='Atom_fullname', how='inner', suffixes=('', '_std'))
    unique_atom_fullname_df = nmr_df.drop_duplicates(subset='Atom_fullname', keep="first")

    merged_df = unique_atom_fullname_df.merge(mean_std_df, on='Atom_fullname', how='inner', suffixes=('', '_mean'))
    merged_df = merged_df.drop(["Val"], axis=1)
    cs_val_col = merged_df.pop("Val_mean")
    merged_df.insert(0, "Val", cs_val_col)
    nmr_df = merged_df

    # Match nmr DataFrame and pdb Atom_fullname
    rows = []
    for atom_fullname in ds.get("atom_fullnames"):
        soft_atom_fullname = JOIN_CHAR.join(atom_fullname.split(JOIN_CHAR)[:-1])
        split_values = nmr_df['Atom_fullname'].str.split(JOIN_CHAR)
        nmr_entry_idx = nmr_df[split_values.apply(lambda x: x[:3] == soft_atom_fullname.split(JOIN_CHAR))].index
        nmr_exists = len(nmr_entry_idx) > 0
        if nmr_exists:
            idx = nmr_entry_idx[0]
            row = nmr_df.loc[idx].values.tolist()
        else:
            resnum, resname, atomname, chainid = atom_fullname.split(JOIN_CHAR)
            row = []
            for col in nmr_df.columns:
                if col == "Comp_index_ID":
                    val = int(resnum)
                elif col == "Comp_ID":
                    val = resname
                elif col == "Atom_ID":
                    val = atomname
                elif col == "Resnumber":
                    val = resnum
                elif col == "Atom_fullname":
                    val = atom_fullname
                elif col == "Entity_ID":
                    val = chainid
                else:
                    val = np.nan
                row.append(val)
        rows.append(row)

    cs_df = pd.DataFrame(rows, columns=nmr_df.columns)
    cs_df.loc[cs_df.Val_std > 0.1, "Val"] = np.nan # Drop nmr assignments with high variability

    heavy_atoms_idcs = np.argwhere([atom_name.startswith('H') for atom_name in ds.get("atom_names")]).flatten()
    discard_atom_idcs = get_atom_idcs_with_high_rmsf(ds.get(AtomicDataDict.POSITIONS_KEY), heavy_atoms_idcs, mol, rmsf_threshold=rmsf_threshold)

    # Extract chemical shifts
    cs = cs_df.Val.values.astype(np.float32)
    if discard_atom_idcs is not None:
        cs[discard_atom_idcs] = np.nan
    cs = np.repeat(cs[None, ...], len(ds.get(AtomicDataDict.POSITIONS_KEY)), axis=0)

    # Build dataset
    ds.update({
        "chemical_shifts": cs,
    })

    yield ds

def get_structure(topology, trajectories = [], selection=None):
    try:
        u = mda.Universe(topology, *trajectories)
        if selection is None:
            selection = "all"
        mol = u.select_atoms(selection)
    except Exception as e:
        raise Exception (f"Could not load {topology}") from e

    atom_resnumbers = []
    atom_resnames = []
    atom_names = []
    atom_chains = []
    atom_fullnames = []
    atom_types = []

    def get_chainID(atom):
        try:
            return atom.chainID
        except:
            return 'A'
        
    def get_element(atom):
        try:
            return atom.element
        except:
            return atom.type
    
    for atom in mol.atoms:
        atom_resnumber = atom.resnum
        atom_resname = atom.resname
        atom_name = fix_atom_naming(atom.name)
        atom_chain = get_chainID(atom)
        atom_fullname = JOIN_CHAR.join([str(atom_resnumber), atom_resname, atom_name, atom_chain])

        atom_resnumbers.append(atom_resnumber)
        atom_resnames.append(atom_resname)
        atom_names.append(atom_name)
        atom_chains.append(atom_chain)
        atom_fullnames.append(atom_fullname)
        element = get_element(atom)
        atom_types.append(DataDict.get_atom_type(atom_resname, atom_name, element, verbose=True))
    
    atom_resnumbers = np.array(atom_resnumbers)
    atom_resnames = np.array(atom_resnames)
    atom_names = np.array(atom_names)
    atom_chains = np.array(atom_chains)
    atom_fullnames = np.array(atom_fullnames)
    atom_types = np.array(atom_types)

    coords = []
    for ts in u.trajectory:
        coords.append(mol.positions)
    coords = np.stack(coords, axis=0)

    # - Remove clashing atoms - #
    mask = np.triu_indices(coords.shape[1], k=1)
    lengths = np.linalg.norm(coords[:, mask[0]] - coords[:, mask[1]], axis=-1)
    keep_edges = np.all(lengths > 0.5, axis=0)
    keep_idcs = np.union1d(np.unique(mask[0][keep_edges]), np.unique(mask[1][keep_edges]))

    # Build dataset
    ds = {
        AtomicDataDict.POSITIONS_KEY: coords[:, keep_idcs],
        AtomicDataDict.NODE_TYPE_KEY: atom_types[keep_idcs],
        "atom_resnumbers": atom_resnumbers[keep_idcs],
        "atom_resnames": atom_resnames[keep_idcs],
        "atom_names": atom_names[keep_idcs],
        "atom_chains": atom_chains[keep_idcs],
        "atom_fullnames": atom_fullnames[keep_idcs],
    }
    
    return ds, mol

def build_input_data(pdb_file, r_max: float, traj_files=[], selection="all"):
    dataset, _ = get_structure(pdb_file, traj_files=traj_files, selection=selection)
    return prepare_dataset(dataset, r_max=r_max)
    
def prepare_dataset(dataset, r_max):
    coords = torch.from_numpy(dataset.get("coords"))
    node_types = torch.from_numpy(dataset.get("atom_types"))
    batch = torch.zeros(coords.shape[-2], dtype=torch.long)
    
    return [{
        AtomicDataDict.POSITIONS_KEY: pos,
        f"{AtomicDataDict.POSITIONS_KEY}_slices": torch.tensor([0, len(pos)]),
        AtomicDataDict.NODE_TYPE_KEY: node_types,
        AtomicDataDict.EDGE_INDEX_KEY: get_edge_index(positions=pos, r_max=r_max),
        AtomicDataDict.BATCH_KEY: batch,
    }
    for pos in coords
    ], dataset

def get_edge_index(positions: torch.Tensor, r_max: float):
    dist_matrix = torch.norm(positions[:, None, ...] - positions[None, ...], dim=-1).fill_diagonal_(torch.inf)
    return torch.argwhere(dist_matrix <= r_max).T.long()

def run_inference(
    model_path: str,
    test_regex: str,
    device: str = 'cpu',
    output_dir: str = '../inference',
    selection: str = "protein",
):
    model, config = load_model(Path(model_path), device=device)
    for input_file in glob.glob(test_regex):
        print(f"Running inference on {input_file}")
        if input_file.endswith('.npz'):
            batches, dataset = prepare_dataset(np.load(input_file, allow_pickle=True), r_max=config.get('r_max'))
        else:
            batches, dataset = build_input_data(input_file, traj_files=[], selection=selection, r_max=config.get('r_max'))
        cs = []
        for batch in tqdm(batches):
            try:
                for v in batch.values():
                    if isinstance(v, torch.Tensor):
                        v.to(device)
                cs.append(evaluate(
                    model=model,
                    batch=batch,
                    node_out_keys=[AtomicDataDict.NODE_OUTPUT_KEY]
                )[AtomicDataDict.NODE_OUTPUT_KEY].cpu().numpy())
            except:
                pass
        cs = np.stack(cs, axis=0)
        ds = dict(dataset)
        ds['chemical_shifts_pred'] = cs
        os.makedirs(output_dir, exist_ok=True)
        np.savez(os.path.join(output_dir, basename(input_file)), **ds)