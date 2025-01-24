import logging
import os
import yaml
import pynmrstar
import requests
import shutil
import urllib.request
import numpy as np
import pandas as pd
import difflib
import mdtraj as md

import MDAnalysis as mda
# from MDAnalysis.analysis import dihedrals

from typing import List, Optional, Union
from collections import OrderedDict
from pathlib import Path

from pdbfixer import PDBFixer
from openmm.app import PDBFile, ForceField
from sklearn.ensemble import IsolationForest
from geqtrain.utils import ATOMIC_NUMBER_MAP

import warnings
warnings.filterwarnings('ignore')


def search_keys(data, search_terms):
    """
    Recursively search for keys in a dictionary or list that match the given search terms.
    
    :param data: Dictionary or list to search
    :param search_terms: List of key substrings to search for
    :return: Dictionary of found keys and their values
    """
    found = {}
    
    if isinstance(data, dict):
        for key, value in data.items():
            if any(term == key.lower() for term in search_terms):
                found[key] = value
            if isinstance(value, (dict, list)):
                found.update(search_keys(value, search_terms))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                found.update(search_keys(item, search_terms))
    
    return found

def get_pdb_experimental_details(pdb_code):
    """
    Fetches experimental details, searching for pH and temperature in the metadata.
    
    :param pdb_code: The PDB code of the structure (e.g., '1abc')
    :return: A dictionary containing matched keys and their values
    """
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_code}"
    
    # Make a GET request to fetch the PDB metadata
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error: Unable to fetch data for PDB code {pdb_code}. Status code: {response.status_code}")
        return None
    
    pdb_data = response.json()
    
    # Define search terms for pH and temperature
    ph_terms = ['p_h', 'ph']
    temp_terms = ['temp']
    
    # Search for pH and temperature keys in the JSON
    ph_details = search_keys(pdb_data, ph_terms)
    temp_details = search_keys(pdb_data, temp_terms)
    
    # Combine results
    details = {
        "ph": list(ph_details.values())[0],
        "temp": list(temp_details.values())[0],
    }
    
    return details


class AtomTypeAssigner:
    def __init__(self, config, atom_type_map=None):
        # Load configuration from YAML
        with open(config, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.key_formats      : dict = self.config.get("key_formats", {"atom_types": "{resname}.{atomname}"})
        self.groups           : list = self.config.get("groups", [])
        self.accepted_resnames: list = self.config.get("accepted_resnames", [])
        self.remap_resname    : str  = self.config.get("remap_resname", "MOL")
        self.atom_type_map           = dict()
        self.next_atom_type_dict     = dict()
        self.group_map        : dict = self._prepare_groups()

        self.atom_type_map_is_fixed = self.load_atom_type_map(atom_type_map)
    
    def filename(self, data_root):
        return os.path.join(data_root, 'atom_type_map.yaml')
    
    def save_atom_type_map(self, data_root):
        with open(self.filename(data_root), 'w') as f:
            yaml.safe_dump(self.atom_type_map, f)

    def load_atom_type_map(self, atom_type_map):
        try:
            if atom_type_map and os.path.isfile(atom_type_map):
                with open(atom_type_map, 'r') as f:
                    atom_type_map = dict(yaml.safe_load(f))
                    for k, v in atom_type_map.items():
                        new_v = {_k.upper(): _v for _k, _v in v.items()}
                        self.atom_type_map[k] = new_v
                return True
        except:
            logging.warning(f"Failed loading file {atom_type_map}")
        return False
    
    def _prepare_groups(self) -> dict:
        """
        Prepares a mapping from key to group index to ensure all keys in a group
        get the same atom type.
        """
        group_map = dict()
        for group in self.groups:
            for key in group:
                group_map[key] = group[0]  # Assign group leader
        return group_map

    def remap_atom(self, atom):
        """
        Remap an atom's properties if its resname is not in the accepted list.
        """
        if atom['resname'] not in self.accepted_resnames:
            atom['resname'] = self.remap_resname
            atom['atomname'] = atom['element']  # Map atomname to element
        return atom

    def generate_key(self, atom, key_format):
        """
        Generate a key for the atom based on the key_format.
        """
        atom = self.remap_atom(atom)  # Apply remapping rules
        key = key_format.format(**atom).upper()
        return self.group_map.get(key, key)  # Map to group leader if applicable

    def assign_atom_types(self, atoms):
        """
        Assign atom types to a list of atoms based on the config and update
        the atom_type_map dynamically.
        """
        atom_types = {}
        for atom in atoms:
            for format_name, key_format in self.key_formats.items():
                key = self.generate_key(atom, key_format)
                atom_type_map = self.atom_type_map.get(format_name, dict())
                if key not in atom_type_map:
                    if self.atom_type_map_is_fixed:
                        # Find the most similar atom type in the map
                        closest_matches = difflib.get_close_matches(key.upper(), atom_type_map.keys(), n=5)
                        filtered_matches = [match for match in closest_matches if all(a[0] == b[0] for a, b in zip(match.split('.'), key.upper().split('.')))]
                        if filtered_matches:
                            new_key = filtered_matches[0]
                            logging.warning(f"Using a loaded atom_type_map, but found a new atom_type: {key}. Assigning a similar atom_type: {new_key}")
                            key = new_key
                        else: raise ValueError(f"Unknown atom type {key} and no similar type found in atom_type_map.")
                    else:
                        atom_type_map[key] = self.next_atom_type_dict.get(format_name, 0)
                        self.next_atom_type_dict[format_name] = self.next_atom_type_dict.get(format_name, 0) + 1
                        self.atom_type_map[format_name] = atom_type_map
                atom_types_list: list = atom_types.get(format_name, [])
                atom_types_list.append(atom_type_map[key])
                atom_types[format_name] = atom_types_list
        return atom_types
    
    def read_atom_types(self, atom, format_name, key_format, atom_type):
        key = self.generate_key(atom, key_format)
        atom_type_map = self.atom_type_map.get(format_name, dict())
        if key not in atom_type_map:
            atom_type_map[key] = atom_type
        self.atom_type_map[format_name] = atom_type_map

class NMRDatasetBuilder:
    def __init__(self, config, atom_type_map=None):
        self.atom_type_assigner = AtomTypeAssigner(config, atom_type_map)
        self.match_columns = ['PDB_FILENAME', 'PDB_PH', 'PDB_TEMP', 'NMR_FILENAME', 'NMR_PH', 'NMR_TEMP', 'CHAINID', 'RESNUM', 'RESNAME']
        self._dataset_info = []
        self._last_dataset_info_len = 0
        self.__dataset_info = None
        self.statistics = {}
        self.outliers = None

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
    
    @property
    def dataset_info(self):
        if self.__dataset_info is None or len(self._dataset_info) != self._last_dataset_info_len:
            if len(self._dataset_info) == 0: return None
            logging.info('- Building dataset info file...')
            self._last_dataset_info_len = len(self._dataset_info)
            __dataset_info = pd.concat(self._dataset_info, ignore_index=True)
            self.__dataset_info = __dataset_info.groupby(self.match_columns, dropna=False).first().reset_index()
        return self.__dataset_info
    
    def npz_datadir(self, data_root):
        return os.path.join(data_root, 'npz')
    
    def npz_excluded_datadir(self, data_root):
        return os.path.join(data_root, 'npz_excluded')
    
    def npz_filename(self, data_root, pdbcode, bmrbid=None):
        return os.path.join(self.npz_datadir(data_root), self.format_npz(pdbcode, bmrbid))
    
    def npz_excluded_filename(self, data_root, pdbcode, bmrbid):
        return os.path.join(self.npz_excluded_datadir(data_root), self.format_npz(pdbcode, bmrbid))
    
    def stem(self, filename):
        return Path(str(filename)).stem
    
    def format_npz(self, pdbcode, bmrbid=None):
        if bmrbid is None:
            return f'{self.stem(pdbcode)}.npz'
        return f'{self.stem(pdbcode)}.{self.stem(bmrbid)}.npz'
    
    def dataset_info_filename(self, data_root):
        return os.path.join(data_root, 'dataset_info.csv')
    
    def save_dataset_info(self, data_root):
        self.dataset_info.to_csv(self.dataset_info_filename(data_root))
        logging.info(f'Dataset info file saved at {self.dataset_info_filename(data_root)}')
    
    def node_type_stats(self, data_root):
        return os.path.join(data_root, 'node_type_statistics.yaml')
    
    def build(self, nmr2pdb: Union[List, str], slicing=None, data_root: str = './'):
        if isinstance(nmr2pdb, str):
            df = pd.read_csv(nmr2pdb)
            df = df.rename(columns=lambda x: x.strip())
            nmr2pdb = [(pdb, nmr) for pdb, nmr in zip(df['pdb'], df['nmr'])]
        
        os.makedirs(data_root, exist_ok=True)
        os.makedirs(self.npz_datadir(data_root), exist_ok=True)
        logging.info(f'--- BUILDING DATASET ---')
        for pdbcode, bmrbid in nmr2pdb:
            if os.path.isfile(self.npz_filename(data_root, pdbcode, bmrbid)) or \
               os.path.isfile(self.npz_excluded_filename(data_root, pdbcode, bmrbid)):
                try:    pdb = dict(np.load(self.npz_filename(data_root, pdbcode, bmrbid)))
                except: pdb = dict(np.load(self.npz_excluded_filename(data_root, pdbcode, bmrbid)))
                self.update_dataset_info(pdb, pdbcode, bmrbid, atom_type_assigner_is_missing=True)
                continue
            try:
                nmr_cs, nmr_ph, nmr_temp = self.getcs(bmrbid, datadir=os.path.join(data_root, 'bmrb.cs'))
            except Exception as e:
                logging.warning(e)
                continue
            pdb = self.getpdb(pdbcode, datadir=os.path.join(data_root, 'rcsb.pdb'), slicing=slicing)
            if pdb is None:
                continue

            pdb = self.align_and_assign_chemical_shifts(pdb, nmr_cs)

            if nmr_ph is not None:
                 pdb.update({'nmr_ph': nmr_ph,})
            if nmr_temp is not None:
                 pdb.update({'nmr_temp': nmr_temp,})

            self.update_dataset_info(pdb, pdbcode, bmrbid)

            np.savez(self.npz_filename(data_root, pdbcode, bmrbid), **pdb)
            logging.info(f'File {self.npz_filename(data_root, pdbcode, bmrbid)} saved!')
        
        self.atom_type_assigner.save_atom_type_map(data_root)
        self.save_dataset_info(data_root)
    
    def build_inference(self, topology: str, traj: Optional[str]=None, slicing=None, ph=7.0, temperature=298., data_root: str = './'):
        os.makedirs(data_root, exist_ok=True)
        os.makedirs(self.npz_datadir(data_root), exist_ok=True)
        logging.info(f'--- BUILDING DATASET ---')
        
        pdb = self.parseinput(topology, traj=traj, slicing=slicing, ph=ph, temperature=temperature)
        p = Path(topology)
        np.savez(self.npz_filename(data_root, p.stem), **pdb)
        logging.info(f'File {self.npz_filename(data_root, p.stem)} saved!')
        return pdb
    
    def update_dataset_info(self, pdb: dict, pdbcode, bmrbid, atom_type_assigner_is_missing=False):
        data = []

        for items in zip(
            pdb.get('atom_fullnames'),
            pdb.get('atom_numbers'),
            pdb.get('chemical_shifts')[0],
            *[pdb.get(format_name) for format_name in self.atom_type_assigner.key_formats]
        ):
            vars = {
                'fullname'  : items[0],
                'atomnumber': items[1],
                'cs'        : items[2],
                **{
                    (format_name, key_format): _type
                    for (format_name, key_format), _type
                    in zip(self.atom_type_assigner.key_formats.items(), items[3:])
                }
            }
            
            chainID, resnum, resname, atomname = vars.get('fullname').split('.')
            element = {v: k for k, v in ATOMIC_NUMBER_MAP.items()}[vars.get('atomnumber')]
            
            if atom_type_assigner_is_missing:
                atom_data = {
                    "chainID"  : chainID,
                    "resnumber": resnum,
                    "resname"  : resname,
                    "atomname" : atomname,
                    "element"  : element,
                }

                for format_name_key_format_tuple, atom_type in vars.items():
                    if not isinstance(format_name_key_format_tuple, tuple) : continue
                    format_name, key_format = format_name_key_format_tuple
                    self.atom_type_assigner.read_atom_types(atom_data, format_name, key_format, atom_type.item())
            
            if np.isnan(vars.get('cs')):
                continue
            
            def cast_to_item(x):
                return x.item() if isinstance(x, np.ndarray) else x
            record = {k: cast_to_item(v) for k,v in {
                'PDB_FILENAME': self.stem(pdbcode),
                'PDB_PH'      : pdb.get('pdb_ph'  , np.NaN),
                'PDB_TEMP'    : pdb.get('pdb_temp', np.NaN),
                'NMR_FILENAME': self.stem(bmrbid),
                'NMR_PH'      : pdb.get('nmr_ph'  , np.NaN),
                'NMR_TEMP'    : pdb.get('nmr_temp', np.NaN),
                'CHAINID'     : str(chainID),
                'RESNUM'      : resnum,
                'RESNAME'     : resname,
                atomname      : vars.get('cs'),
            }.items()}

            data.append(record)
        self._dataset_info.append(pd.DataFrame(data))

    def getcs(self, bmrbid, datadir):
        """
        Get NMR experimental data from file save in local computer.
        First check if the BMRB file is downloaded, if not download it to local.
        """
        if isinstance(bmrbid, str):
            bmrbid  = bmrbid.strip()
        if os.path.isfile(bmrbid):
            file_path = bmrbid
        else:
            os.makedirs(datadir, exist_ok=True)
            file_path = os.path.join(datadir, f'{bmrbid}.str')
            if not os.path.exists(file_path): #check file exitst
                entrydownload = pynmrstar.Entry.from_database(bmrbid, convert_data_types=True) #convert_data_types to import number as floats
                entrydownload.write_to_file(file_path)
        # convert_data_types to import number as floats
        entry = pynmrstar.Entry.from_file(file_path, convert_data_types=True)
        # entry.write_to_file(f'{bmrbid}')
        # cs_result_sets = [] # To store all chemical shift present in the BMRB file
        cs = OrderedDict() # To store the final data
        for chemical_shift_loop in entry.get_loops_by_category("Atom_chem_shift"):
            for record in chemical_shift_loop.get_tag(['Comp_index_ID', 'Comp_ID', 'Atom_ID', 'Atom_type', 'Val', 'Val_err', 'Auth_asym_ID']):
                cs[f'{record[6] or "A"}.{record[0]}.{record[1]}.{record[2]}'] = float(record[4])
        
        def reorder_ordered_dict(odict):
            """
            Reorder an OrderedDict based on CHAINID and RESNUM.
            """
            # Extract keys and sort by CHAINID and RESNUM
            sorted_keys = sorted(odict.keys(), key=lambda k: (k.split(".")[0], int(k.split(".")[1])))
            
            # Create a new OrderedDict with sorted keys
            sorted_odict = OrderedDict((key, odict[key]) for key in sorted_keys)
            return sorted_odict
        
        cs = reorder_ordered_dict(cs)

        # Extract sample conditions
        ph, temp = None, None
        for saveframe in entry.get_saveframes_by_category("sample_conditions"):
            # print(f"Sample Condition ID: {saveframe['ID'][0]}")
            variables = saveframe.get_loop("_Sample_condition_variable")
            for row in variables:
                if ph is None and row[0].lower() == 'ph':
                    ph = float(row[1])
                elif temp is None and row[0].lower() == 'temperature':
                    temp = float(row[1])
        return cs, ph, temp

    def getpdb(self, pdbcode: str, datadir: str, slicing = None, downloadurl = "https://files.rcsb.org/download/"):
        """
        Downloads a PDB file from the Internet and saves it in a data directory.
        :param pdbcode: The standard PDB ID e.g. '3ICB' or '3icb'. You can also provide directly a pdb filename (ending in .pdb) which is already locally stored.
        :param datadir: The directory where the downloaded file will be saved
        :param downloadurl: The base PDB download URL, cf.
            `https://www.rcsb.org/pages/download/http#structures` for details
        :return: the full path to the downloaded PDB file or None if something went wrong
        """
        try:
            if os.path.isfile(pdbcode):
                outfnm = pdbcode
            else:
                os.makedirs(datadir, exist_ok=True)
                if pdbcode.endswith(".pdb"):
                    pdbfn = pdbcode
                    pdbcode = Path(pdbcode).stem
                    url = None
                else:
                    pdbfn = pdbcode + ".pdb"
                    url = downloadurl + pdbfn
                outfnm = os.path.join(datadir, pdbfn)
                if url is not None and not os.path.isfile(outfnm):
                    urllib.request.urlretrieve(url, outfnm)
            return self.parsepdb(pdbcode, outfnm, slicing=slicing)
        except Exception as err:
            logging.warning(f"Skipping {pdbcode} | {str(err)}")
            return None

    def parsepdb(self, pdbcode, pdb, slicing):
        pdb_info = self.get_pdb_info(pdbcode, pdb)
        
        if not pdb_info.get('pdb_has_hydrogens', False):
            # Add Hydrogens
            forcefield = ForceField('amber14-all.xml', 'amber14/tip3p.xml')
            fixer = PDBFixer(pdb)
            fixer.findMissingResidues()
            fixer.findNonstandardResidues()
            fixer.replaceNonstandardResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.removeHeterogens(keepWater=False)
            fixer.addMissingHydrogens(pdb_info.get('pdb_ph', 7.0), forcefield=forcefield)
            PDBFile.writeFile(fixer.topology, fixer.positions, open(pdb, 'w'))
        
        data = self.read_traj(pdb, slicing=slicing, remove_clashes=True)
        data.update(pdb_info)
        return data
    
    def parseinput(self, topology, traj=None, slicing=None, ph=None, temperature=None):
        info = {}
        if ph is not None: info['pdb_ph']   = ph
        if temperature is not None: info['pdb_temp'] = temperature

        data = self.read_traj(topology, trajectory=traj, slicing=slicing)
        data.update(info)
        return data

    def read_traj(self, topology, trajectory: Optional[str]=None, slicing=None, remove_clashes=False):
        u = mda.Universe(topology, trajectory) if trajectory is not None else mda.Universe(topology)
        mol = u.select_atoms('all')

        atom_chains     = []
        atom_resnumbers = []
        atom_resnames   = []
        atom_names      = []
        atom_fullnames  = []
        atom_data       = []
        atom_numbers    = []

        def get_chainID(atom):
            try:    return atom.chainID
            except: return 'A'
        
        for atom in mol.atoms:
            atom_chain = get_chainID(atom)
            atom_resnumber = atom.resnum
            atom_resname = atom.resname
            atom_name = atom.name
            atom_fullname = '.'.join([atom_chain, str(atom_resnumber), atom_resname, atom_name])

            atom_chains.append(atom_chain)
            atom_resnumbers.append(atom_resnumber)
            atom_resnames.append(atom_resname)
            atom_names.append(atom_name)
            atom_fullnames.append(atom_fullname)

            try:    atom_element = atom.element
            except: atom_element = atom.type
            atom_numbers.append(ATOMIC_NUMBER_MAP.get(atom_element.upper()))

            atom_data.append({
                "chainID"  : atom_chain,
                "resnumber": atom_resnumber,
                "resname"  : atom_resname,
                "atomname" : atom_name,
                "element"  : atom_element,
            })
        
        atom_chains     = np.array(atom_chains)
        atom_resnumbers = np.array(atom_resnumbers)
        atom_resnames   = np.array(atom_resnames)
        atom_names      = np.array(atom_names)
        atom_fullnames  = np.array(atom_fullnames)
        atom_numbers    = np.array(atom_numbers, dtype=np.int8)
        atom_types      = {k: np.array(v) for k, v in self.atom_type_assigner.assign_atom_types(atom_data).items()}

        
        if trajectory is None:
            traj = md.load(topology)
        elif trajectory.endswith('.xtc'):
            traj = md.load_xtc(trajectory, top=topology)
        elif trajectory.endswith('.trr'):
            traj = md.load_trr(trajectory, top=topology)
        sasa = md.shrake_rupley(traj, probe_radius=1.4, mode='residue')
        dssp = md.compute_dssp(traj, simplified=True)

        coords, atom_sasa, atom_ss = [], [], []
        if slicing is None: slicing = slice(0, None, 1)
        for ts in u.trajectory[slicing]:
            coords.append(mol.positions)
            frame_atom_sasa = np.zeros((u.atoms.n_atoms), dtype=np.float32)
            frame_atom_ss   = np.empty((u.atoms.n_atoms), dtype=str)
            for idx, residue in enumerate(u.residues):
                ss = dssp[ts.frame, idx]
                asa = sasa[ts.frame, idx]
                for atom in residue.atoms:
                    atom_index = atom.index
                    frame_atom_sasa[atom_index] = asa
                    frame_atom_ss[atom_index] = ss
            atom_sasa.append(frame_atom_sasa)
            atom_ss.append(frame_atom_ss)
        coords = np.stack(coords, axis=0)
        # Check overlapping (soft algorithm)
        dists = np.linalg.norm(coords[:, 1:] - coords[:, :-1], axis=-1)
        if np.any(dists < 0.5):
            raise Exception(f"Found clashing atoms (distance < 0.5 Angstrom)")

        atom_sasa = np.stack(atom_sasa)
        atom_ss = np.stack(atom_ss)
        ss_mapping = {'C': 0, 'E': 1, 'H': 2, 'N': 3}
        atom_ss_int = np.vectorize(ss_mapping.get)(atom_ss)

        # rama = dihedrals.Ramachandran(mol).run()
        # janin = dihedrals.Janin(mol).run()

        # - Remove clashing atoms - #
        if remove_clashes:
            mask = np.triu_indices(coords.shape[1], k=1)
            lengths = np.linalg.norm(coords[:, mask[0]] - coords[:, mask[1]], axis=-1)
            keep_edges = np.all(lengths > 0.5, axis=0)
            keep_idcs = np.union1d(np.unique(mask[0][keep_edges]), np.unique(mask[1][keep_edges]))
        else:
            keep_idcs = np.arange(coords.shape[1])

        # Build dataset
        data = {
            "coords"         : coords[:, keep_idcs],
            "atom_chains"    : atom_chains[keep_idcs],
            "atom_resnumbers": atom_resnumbers[keep_idcs],
            "atom_resnames"  : atom_resnames[keep_idcs],
            "atom_names"     : atom_names[keep_idcs],
            "atom_fullnames" : atom_fullnames[keep_idcs],
            "atom_numbers"   : atom_numbers[keep_idcs],
            "atom_ss"        : atom_ss_int[:, keep_idcs],
            "atom_sasa"      : atom_sasa[:, keep_idcs],
        }
        atom_types = {k: v[keep_idcs] for k, v in atom_types.items()}
        data.update(atom_types)
        return data
    
    def get_pdb_info(self, pdbcode, pdb) -> dict:
        try:
            ph, temperature, pdb_has_hydrogens = None, None, False
            line_limit_for_searching_hydrogens = 20
            with open(pdb, 'r') as pdb_file:
                for line in pdb_file:
                    # Look for pH in REMARK 210 lines
                    if 'REMARK 210  PH' in line:
                        # Split the line and find numeric values
                        parts = line.split(':')
                        if len(parts) > 1:
                            ph_values = [float(val.strip()) for val in parts[1].split(';') if val.strip().replace('.', '').isdigit()]
                            if ph_values:
                                ph = ph_values[0]

                    # Look for temperature in REMARK 210 lines
                    if 'REMARK 210  TEMPERATURE' in line:
                        # Split the line and find numeric values
                        parts = line.split(':')
                        if len(parts) > 1:
                            temp_values = [float(val.strip()) for val in parts[1].split(';') if val.strip().replace('.', '').isdigit()]
                            if temp_values:
                                temperature = temp_values[0]
                    
                    # Check if the line is an ATOM or HETATM record
                    if line.startswith(('ATOM', 'HETATM')):
                        # PDB format: columns 13-14 typically contain the atom name
                        atom_name = line[12:14].strip()
                        
                        # Check if the atom name starts with 'H' (hydrogen)
                        if atom_name.startswith('H'):
                            pdb_has_hydrogens = True
                        
                        line_limit_for_searching_hydrogens -= 1
                    
                    # Stop searching once we have both values
                    if temperature is not None and ph is not None and (pdb_has_hydrogens or line_limit_for_searching_hydrogens<=0):
                        break
            
            if temperature is None or ph is None:
                try:
                    details = get_pdb_experimental_details(pdbcode)
                    ph = details.get('ph', None)
                    temperature = details.get('temp', None)
                except:
                    pass        
        except FileNotFoundError:
            print(f"Error: File {pdb} not found.")
        except Exception as e:
            print(f"An error occurred while parsing the PDB file: {e}")
        
        info = {
            'pdb_has_hydrogens': pdb_has_hydrogens,
        }
        if ph is not None:
            info['pdb_ph']   = ph
        if temperature is not None:
            info['pdb_temp'] = temperature
        
        return info
    
    def align_and_assign_chemical_shifts(self, pdb: dict, nmr: OrderedDict):
        
        def align_nmr2pdb(pdb: dict, nmr: OrderedDict):

            from Bio import pairwise2

            three_to_one = {
                # Amino acids
                "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
                "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
                "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
                "TRP": "W", "TYR": "Y",
                # Nucleotides
                "DA": "A", "DC": "C", "DG": "G", "DT": "T",  # DNA
                "A": "A", "C": "C", "G": "G", "U": "U"       # RNA
            }

            def parse_residues(array):
                """
                Extract the residue sequence as one-letter codes and the corresponding CHAINIDs.
                """
                sequences = {}
                residue_atoms = {}
                for entry in array:
                    parts = entry.split(".")
                    chain_id = parts[0]  # Extract CHAINID
                    resnum = parts[1]    # Extract residue number
                    resname = parts[2]   # Extract 3-letter residue name
                    one_letter = three_to_one.get(resname, "?")  # Map to 1-letter code
                    
                    residue_key = f"{chain_id}.{resnum}.{resname}"
                    if chain_id not in sequences:
                        sequences[chain_id] = []
                    if residue_key not in residue_atoms:
                        residue_atoms[residue_key] = []
                        sequences[chain_id].append(one_letter)
                    residue_atoms[residue_key].append(entry)
                
                return {k: ''.join(v) for k, v in sequences.items()}, residue_atoms

            def align_and_renumber(pdb_atoms, nmr_atoms, chain_id):
                """
                Perform sequence alignment and renumber residues for a single chain.
                Update all atoms for a residue.
                """
                pdb_residues, pdb_atoms_dict = parse_residues(pdb_atoms)
                nmr_residues, nmr_atoms_dict = parse_residues(nmr_atoms)
                
                pdb_sequence = pdb_residues[chain_id]
                nmr_sequence = nmr_residues[chain_id]
                
                # Align residue names using sequence alignment
                alignment = pairwise2.align.globalxx(pdb_sequence, nmr_sequence, one_alignment_only=True)
                aligned_pdb, aligned_nmr, _, _, _ = alignment[0]
                
                renumbered_pdb = []
                renumbered_nmr = []
                first_nmr_res_assigned = False
                pdb_off, pdb_start_off = 0, 0
                nmr_off, nmr_start_off = 0, 0
                
                pdb_chain_residues = [key for key in pdb_atoms_dict if key.startswith(chain_id)]
                nmr_chain_residues = [key for key in nmr_atoms_dict if key.startswith(chain_id)]
                
                for index, (pdb_res, nmr_res) in enumerate(zip(aligned_pdb, aligned_nmr)):
                    skip = None
                    if pdb_res == "-":
                        skip = 'pdb'
                        pdb_start_off += 1
                    elif nmr_res == "-":
                        skip = 'nmr'
                        if first_nmr_res_assigned:
                            nmr_off += 1
                        else:
                            nmr_start_off += 1
                    
                    new_resnum = '-1'
                    if skip != 'pdb':
                        pdb_key = pdb_chain_residues[index - pdb_off - pdb_start_off]
                        new_resnum = str(int(pdb_key.split(".")[1]) + pdb_off)
                        for atom in pdb_atoms_dict[pdb_key]:
                            fields = atom.split(".")
                            fields[1] = new_resnum
                            renumbered_pdb.append(".".join(fields))
                    if skip != 'nmr':
                        first_nmr_res_assigned = True
                        nmr_key = nmr_chain_residues[index - nmr_off - nmr_start_off]
                        for atom in nmr_atoms_dict[nmr_key]:
                            fields = atom.split(".")
                            fields[1] = new_resnum
                            renumbered_nmr.append(".".join(fields))
                
                return renumbered_pdb, renumbered_nmr

            def renumber_residues_with_chains(pdb, nmr):
                """
                Align and renumber residues in pdb and nmr arrays considering CHAINID.
                """
                chains = np.array([entry.split(".")[0] for entry in pdb])
                _, idx = np.unique(chains, return_index=True)
                chains = chains[np.sort(idx)]
                renumbered_pdb = []
                renumbered_nmr = []
                renumbered_nmr_values = []
                
                for chain_id in chains:
                    pdb_chain = [entry for entry in pdb if entry.startswith(chain_id)]
                    nmr_tuple_chain = [(k, v) for k, v in nmr.items() if k.startswith(chain_id)]
                    nmr_chain = [entry[0] for entry in nmr_tuple_chain]
                    nmr_values_chain = [entry[1] for entry in nmr_tuple_chain]
                    renumbered_nmr_values.extend(nmr_values_chain)
                    
                    if pdb_chain and nmr_chain:
                        pdb_res, nmr_res = align_and_renumber(pdb, nmr, chain_id)
                        renumbered_pdb.extend(pdb_res)
                        renumbered_nmr.extend(nmr_res)
                    else:
                        # If one of the chains is completely missing, add entries unchanged
                        renumbered_pdb.extend(pdb_chain)
                        renumbered_nmr.extend(nmr_chain)
                
                return np.array(renumbered_pdb), np.array(renumbered_nmr), np.array(renumbered_nmr_values)

            # Run the alignment and renumbering
            return renumber_residues_with_chains(pdb, nmr)

        pdb_atom_fullnames = pdb.get('atom_fullnames')
        aligned_pdb_atom_fullnames, aligned_nmr_atom_fullnames, aligned_nmr_values = align_nmr2pdb(pdb_atom_fullnames, nmr)
        # Find indices which turn pdb_atom_fullnames to aligned_pdb_atom_fullnames
        sort_idx = pdb_atom_fullnames.argsort()
        alignment_indices = sort_idx[np.searchsorted(pdb_atom_fullnames , aligned_pdb_atom_fullnames, sorter = sort_idx)]

        # Update pdb by reindexing all keys
        for key, value in pdb.items():
            if not isinstance(value, np.ndarray):
                continue
            for dim in range(value.ndim):
                if value.shape[dim] == len(alignment_indices):
                    indexer = [slice(None)] * dim + [alignment_indices]
                    pdb[key] = value[tuple(indexer)]
                    break
        
        # Assign chemical shifts
        nmr = {aligned_k: v for aligned_k, v in zip(aligned_nmr_atom_fullnames, aligned_nmr_values)}
        nmr_cs_list = [nmr[atom_fullname] if atom_fullname in nmr else np.nan for atom_fullname in aligned_pdb_atom_fullnames]
        num_structures = len(pdb.get('coords'))
        chemical_shifts = np.tile(np.array(nmr_cs_list), (num_structures, 1))
        pdb['chemical_shifts'] = chemical_shifts
        
        return pdb

    def filter_npz_datasets(self, data_root: str = './', ph_range   = [5., 9.], ph_max_diff = 2., temp_range = [278, 350], temp_max_diff = 20.):
        if self.dataset_info is None:
            self.build_dataset_info_from_npz(data_root)
        
        df = self.dataset_info
        logging.info(f'- Filtering npz datasets -')
        os.makedirs(self.npz_excluded_datadir(data_root), exist_ok=True)
        for pdbcode in df["PDB_FILENAME"].unique():
            for bmrbid in df[df["PDB_FILENAME"] == pdbcode]["NMR_FILENAME"].unique():
                exclude = False
                try:
                    row = df[(df["PDB_FILENAME"] == pdbcode) & (df["NMR_FILENAME"] == bmrbid)].iloc[0]
                    pdb_ph = float(row['PDB_PH'])
                    nmr_ph = float(row['NMR_PH'])
                    if nmr_ph < ph_range[0] or nmr_ph > ph_range[1]:
                        exclude=True
                        logging.info(f"PH of NMR {bmrbid}: {nmr_ph}. Excluding from dataset.")
                    diff_ph = np.abs(pdb_ph - nmr_ph)
                    if diff_ph > ph_max_diff:
                        exclude=True
                        logging.info(f"PH difference between PDB {pdbcode} and NMR {bmrbid}: {diff_ph}. Excluding from dataset.")
                except: pass

                try:
                    pdb_temp = float(row['PDB_TEMP'])
                    nmr_temp = float(row['NMR_TEMP'])
                    if nmr_temp < temp_range[0] or nmr_temp > temp_range[1]:
                        exclude=True
                        logging.info(f"TEMP of NMR {bmrbid}: {nmr_temp}. Excluding from dataset.")
                    diff_temp = np.abs(pdb_temp - nmr_temp)
                    if diff_temp > temp_max_diff:
                        exclude=True
                        logging.info(f"TEMP difference between PDB {pdbcode} and NMR {bmrbid}: {diff_temp}. Excluding from dataset.")
                except: pass

                if exclude and os.path.isfile(self.npz_filename(data_root, pdbcode, bmrbid)):
                    shutil.move(self.npz_filename(data_root, pdbcode, bmrbid), self.npz_excluded_filename(data_root, pdbcode, bmrbid))

    def build_statistics(self, data_root: str = './', rebuild=False):
        logging.info(f'--- BUILDING STATISTICS ---')
        if rebuild or self.dataset_info is None:
            if os.path.isfile(self.dataset_info_filename(data_root)):
                self.__dataset_info = pd.read_csv(self.dataset_info_filename(data_root))
                logging.info(f'Dataset info file loaded from {self.dataset_info_filename(data_root)}')
            else:
                self.build_dataset_info_from_npz(data_root, save=not rebuild)

        df = self.dataset_info
        statistics = {}
        keep_cols = []
        for col in [col for col in df.columns if col not in self.match_columns]:
            if df[col].mean() is not np.nan:
                keep_cols.append(col)
        grouped = df.groupby(['RESNAME'])[keep_cols]
        grouped_means = grouped.mean()
        resname_atomname_pairs = [
            (index, col) for index, row in grouped_means.iterrows()
            for col, value in row.items() if not np.isnan(value)
        ]

        for resname_atomname in resname_atomname_pairs:
            resname, atomname = resname_atomname
            statistics[resname_atomname] = df.loc[grouped.groups[resname]][self.match_columns + [atomname]]
        
        self.statistics = statistics

    def build_dataset_info_from_npz(self, data_root, save=True):
        atom_type_assigner_is_missing = not self.atom_type_assigner.load_atom_type_map(data_root)
        logging.info(f'- Loading data from npz -')
        self._dataset_info = []
        for filename in os.listdir(self.npz_datadir(data_root)):
            pdbcode, bmrbid = Path(filename).stem.split('.')
            pdb = dict(np.load(self.npz_filename(data_root, pdbcode, bmrbid)))
            self.update_dataset_info(pdb, pdbcode, bmrbid, atom_type_assigner_is_missing)
        if save:
            self.save_dataset_info(data_root)
    
    def extract_outliers(self):

        logging.info(f'--- EXTRACTING OUTLIERS ---')
        outliers = []
        for key, df in self.statistics.items():
            resname, atomname = key
            df = df.dropna(subset=[atomname])
            if len(df) < 10:
                continue  # Skip small datasets

            # Use IsolationForest to detect outliers
            clf = IsolationForest(contamination=0.005, random_state=42)
            df['outlier'] = clf.fit_predict(df[[atomname]])

            # Calculate mean and standard deviation
            mean = df[atomname].mean()
            std = df[atomname].std()

            # Identify outliers based on IsolationForest and 3*std distance from mean
            df['distance_from_mean'] = np.abs(df[atomname] - mean)
            df['is_outlier'] = (df['outlier'] == -1) & (df['distance_from_mean'] > 4 * std)

            outliers_df = df[df['is_outlier']]
            outliers_df = outliers_df.drop(columns=['outlier', 'distance_from_mean', 'is_outlier'])
            outliers_df = outliers_df.dropna(axis=1, how='all')
            outliers_df = outliers_df.rename(columns={atomname: 'CS'})
            outliers_df = outliers_df.assign(ATOMNAME=atomname)
            if len(outliers_df) > 0:
                outliers.append(outliers_df)
        
        if len(outliers) == 0:
            logging.info(f'- No outliers found -')
            return
        self.outliers = pd.concat(outliers, ignore_index=True)

    def remove_outliers(self, data_root: str = './'):
        logging.info(f'--- REMOVING OUTLIERS FROM NPZ DATASETS ---')
        if self.outliers is None or len(self.outliers) == 0:
            logging.info(f'- No outliers found. Either there are no outliers or you did not call the "extract_outliers()" method -')
            return
        outliers = self.outliers
        for pdbcode in outliers['PDB_FILENAME'].unique():
            pdbcode_outliers = outliers[outliers['PDB_FILENAME'] == pdbcode]
            for bmrbid in pdbcode_outliers['NMR_FILENAME'].unique():
                if not os.path.isfile(self.npz_filename(data_root, pdbcode, bmrbid)):
                    continue
                df = pdbcode_outliers[pdbcode_outliers['NMR_FILENAME'] == bmrbid]
                ds = dict(np.load(self.npz_filename(data_root, pdbcode, bmrbid)))
                atom_fullnames = ds['atom_fullnames']
                for _, row in df.iterrows():
                    query = f"{row['CHAINID']}.{row['RESNUM']}.{row['RESNAME']}.{row['ATOMNAME']}"
                    fltr = np.argwhere(atom_fullnames == query).flatten()
                    try:
                        error = f"Mismatch between outlier value ({row['CS']}) and npz dataset values ({ds['chemical_shifts'][:, fltr].flatten()}). " +\
                        f"pdbcode: {pdbcode} bmrbid: {bmrbid} index: {fltr.flatten()} name: {atom_fullnames[fltr]}"
                        assert np.all(ds['chemical_shifts'][:, fltr] == row['CS']), error
                    except AssertionError as e:
                        if np.all(np.isnan(ds['chemical_shifts'][:, fltr])):
                            logging.debug(f"Filename {self.format_npz(pdbcode, bmrbid)}: chemical shifts for outlier atom '{query}' were already excluded.")
                        else:
                            raise e
                    ds['chemical_shifts'][:, fltr] = np.nan
                np.savez(self.npz_filename(data_root, pdbcode, bmrbid), **ds)

    def build_config_params(self, data_root: str = './'):
        logging.info(f'--- BUILDING CONFIG PARAMS ---')
        type_names_txt    = 'type_names:\n'
        per_type_bias_txt = 'per_type_bias:\n'
        per_type_std_txt  = 'per_type_std:\n'

        for format_name, atom_type_map in self.atom_type_assigner.atom_type_map.items():
            if format_name != 'atom_types':
                continue
            for type_name, atom_type in sorted(atom_type_map.items(), key=lambda x: x[1]):
                resname, atomname = type_name.split('.')
                type_names_txt    += f'  - {type_name:12}# | {atom_type}\n'

                keys = self.get_keys_from_statistics(resname, atomname)
                values = []
                for key in keys:
                    stats = self.statistics.get(key)
                    stat_resname, stat_atomname = key
                    values.append(stats[stat_atomname].values)

                if len(values) > 0:
                    values = np.concatenate(values)
                    values = values[~np.isnan(values)]
                    per_type_bias_txt += f'  - {np.nan_to_num(values.mean(), 0.):8.3f}    # {type_name:12}| {atom_type}\n'
                    per_type_std_txt  += f'  - {max(np.nan_to_num(values.std(),  1.), 1.e-3):8.3f}    # {type_name:12}| {atom_type}\n'
                else:
                    per_type_bias_txt += f'  - {0.:8.3f}    # {type_name:12}| {atom_type}\n'
                    per_type_std_txt  += f'  - {1.:8.3f}    # {type_name:12}| {atom_type}\n'
            
        txt = f'{type_names_txt}\n\n{per_type_bias_txt}\n\n{per_type_std_txt}'
        
        with open(self.node_type_stats(data_root), 'w') as f:
            f.write(txt)
    
    def get_keys_from_statistics(self, resname, atomname):
        for format_name, key_format in self.atom_type_assigner.key_formats.items():
            if format_name == 'atom_types':
                break
        keys = []
        for key in self.statistics.keys():
            stat_resname, stat_atomname = key
            atom = {
                'resname' : stat_resname,
                'atomname': stat_atomname,
                'element' : stat_atomname,
            }
            type_atomname = self.atom_type_assigner.generate_key(atom, key_format)
            if f"{resname}.{atomname}" == type_atomname:
                keys.append(key)
        return keys
    
    def plot_distribution(self, resnames: List[str], atomnames: Optional[List[str]] = None, save=False):
        import plotly.express as px
        for resname in resnames:
            for atomname in atomnames:
                try:
                    data = self.statistics.get((resname, atomname))
                    if len(data) == 0:
                        return
                    fig = px.histogram(
                        data,
                        x=atomname,
                        color='PDB_FILENAME',
                        marginal="rug", # can be `box`, `violin`
                        hover_data=data.columns,
                        nbins=100,
                        title=f"{resname}-{atomname}",
                        
                    )
                    fig.show()
                    if save:
                        fig.write_html(f"../media/{resname}-{atomname}.html")
                except:
                    pass