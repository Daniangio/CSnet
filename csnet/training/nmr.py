from typing import Optional
import pandas as pd
import shlex
import re


JOIN_CHAR = '_'

ATOM_NAME_EXTENSIONS = {
        "THR_HG2_": ["THR_HG21_", "THR_HG22_", "THR_HG23_"],
        "ALA_HB_": ["ALA_HB1_", "ALA_HB2_", "ALA_HB3_"],
        "VAL_HG1_": ["VAL_HG11_", "VAL_HG12_", "VAL_HG13_"],
        "VAL_HG2_": ["VAL_HG21_", "VAL_HG22_", "VAL_HG23_"],
        "ILE_HD1_": ["ILE_HD11_", "ILE_HD12_", "ILE_HD13_"],
        "ILE_HG2_": ["ILE_HG21_", "ILE_HG22_", "ILE_HG23_"],
        "LEU_HD1_": ["LEU_HD11_", "LEU_HD12_", "LEU_HD13_"],
        "LEU_HD2_": ["LEU_HD21_", "LEU_HD22_", "LEU_HD23_"],
        "ASN_HD2_": ["ASN_HD21_", "ASN_HD22_"],
        "ABA_HG_": ["ABA_HG1_", "ABA_HG2_", "ABA_HG3_"],
        "MET_HE_": ["MET_HE1_", "MET_HE2_", "MET_HE3_"],
        "LYS_HZ_": ["LYS_HZ1_", "LYS_HZ2_", "LYS_HZ3_"],
    }


class NMR:

    audit_df: Optional[pd.DataFrame] = None
    chem_shift_ref_df: Optional[pd.DataFrame] = None
    atom_chem_shift_df: Optional[pd.DataFrame] = None

    def __init__(
        self,
        nmr_file,
        chain = 'A',
    ) -> None:
        self._read(nmr_file, chain = chain)

    def _adjust_atom_naming(self, row):
        for atom_name in ATOM_NAME_EXTENSIONS:
            if atom_name in row['Atom_fullname']:
                new_rows = []
                for new_name in ATOM_NAME_EXTENSIONS.get(atom_name): # Create new rows
                    new_row = row.copy()
                    new_row['Atom_fullname'] = row['Atom_fullname'].replace(atom_name, new_name)
                    new_rows.append(new_row)
                return pd.DataFrame(new_rows)
        return pd.DataFrame([row])  # Return the original row

    def _read(self, nmr_file, chain = 'A',):
        with open(nmr_file) as f:
            _section = ""
            data_columns = []
            atom_chem_shift_df_list = []

            for line in f.readlines():
                if line.startswith("loop_"):
                    _section = "data_columns"
                    data_columns = []
                    data_rows = []
                    continue
                if _section.startswith("data"):
                    if line.startswith("_Audit"):
                        _section = "data_columns_Audit"
                        data_columns.append(line.split('.')[-1].strip())
                    elif line.startswith("_Chem_shift_ref"):
                        _section = "data_columns_Chem_shift_ref"
                        data_columns.append(line.split('.')[-1].strip())
                    elif line.startswith("_Atom_chem_shift"):
                        _section = "data_columns_Atom_chem_shift"
                        data_columns.append(line.split('.')[-1].strip())
                    elif re.match("( )*stop_", line):
                        if _section == "data_rows_Audit":
                            self.audit_df = pd.DataFrame.from_records(data_rows, columns=data_columns, index="Revision_ID")
                        elif _section == "data_rows_Chem_shift_ref":
                            self.chem_shift_ref_df = pd.DataFrame.from_records(data_rows, columns=data_columns)
                        elif _section == "data_rows_Atom_chem_shift":
                            df = pd.DataFrame.from_records(data_rows, columns=data_columns, index="ID")
                            df["Val"] = df["Val"].astype(float)
                            try:
                                df["Resnumber"] = df["Original_PDB_residue_no"].astype(int).astype(str)
                            except ValueError:
                                try:
                                    df["Resnumber"] = df["Seq_ID"].astype(int).astype(str)
                                except KeyError:
                                    df["Resnumber"] = df["Auth_seq_ID"].astype(int).astype(str)
                            except:
                                df["Resnumber"] = (df["Val"] * 0).astype(int).astype(str)
                            try:
                                df["Entity_ID"] = df["Entity_ID"].replace({'.': 1}).astype(int)
                                df["Entity_ID"] = (df["Entity_ID"] - 1).astype(str)
                                df["Atom_fullname"] = df[
                                    ["Resnumber", "Comp_ID", "Atom_ID", "Entity_ID"]
                                ].agg(JOIN_CHAR.join, axis=1)
                            except:
                                df["Atom_fullname"] = df[
                                    ["Resnumber", "Original_PDB_residue_name", "Original_PDB_atom_name"]
                                ].assign(Chain = chain).agg(JOIN_CHAR.join, axis=1)

                            # Fix naming
                            df["Atom_fullname"] = df["Atom_fullname"].str.replace('_ox', '').str.replace('_(\d+)\s*(.+)_(\d+)', r'_\2\1_\3', regex=True)
                            df = pd.concat(df.apply(self._adjust_atom_naming, axis=1).tolist(), ignore_index=True)

                            atom_chem_shift_df_list.append(df)
                        _section = ""
                        continue
                    elif len(data_columns) > 0:
                        try:
                            splits = shlex.split(line)
                        except:
                            continue
                        if len(splits) == len(data_columns):
                            _section = "data_rows_" + JOIN_CHAR.join(_section.split(JOIN_CHAR)[2:])
                            data_rows.append(splits)
                        else:
                            continue
            
            self.atom_chem_shift_df = pd.concat(atom_chem_shift_df_list)