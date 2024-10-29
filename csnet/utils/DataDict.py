from cgmap._keys import *  # noqa: F403, F401
from .._keys import *  # noqa: F403, F401

# Also import the module to use in TorchScript, this is a hack to avoid bug:
# https://github.com/pytorch/pytorch/issues/52312
from .. import _keys


def get_atom_type(atom_resname, atom_name, element=None, verbose=False, method='full'):
    atom2atom_type = ATOM2ATOM_TYPE(method)
    if method in ['full', 'reduced']:
        atom_type = atom2atom_type.get((atom_resname, atom_name), atom2atom_type.get(element, None))
    elif method == 'base':
        atom_type = atom2atom_type.get(element, None)
    else:
        raise Exception(f'Unrecognized method "{method}". Use one among ["full", "reduced", "base"]')
    if atom_type is None:
        if verbose:
            print(f"Unrecognized atom type for {atom_resname}, {atom_name}, {element}.")
        return atom2atom_type.get("X")
    return atom_type

def get_species(key, chemical_species, method):
        
    def format(resname, chemical_species):
        return f"{resname}.{chemical_species}"
    
    if method == 'full' and isinstance(key, tuple):
        return format(key[0], chemical_species)
    if method == 'base' and isinstance(key, tuple):
        return None
    return chemical_species

def CHEMICAL_SPECIES2ATOM_TYPE_LIST(method):
    chemical_species2atom_type_set = set()
    for key, chemical_species in ATOM2CHEMICAL_SPECIES.items():
        species = get_species(key, chemical_species, method)
        if species is not None:
            chemical_species2atom_type_set.add(species)
    return sorted(list(chemical_species2atom_type_set))

def ATOM2ATOM_TYPE(method):
    chemical_species2atom_type_list = CHEMICAL_SPECIES2ATOM_TYPE_LIST(method)
    atom2atom_type = {}
    for key, chemical_species in ATOM2CHEMICAL_SPECIES.items():
        species = get_species(key, chemical_species, method)
        if species is not None:
            atom2atom_type[key] = chemical_species2atom_type_list.index(species)
    return {k: v for k,v in sorted(atom2atom_type.items(), key=lambda x: x[1])}

ATOM2CHEMICAL_SPECIES = {

    # GENERIC

    "X":  "x",

    "D":  "d",
    "H":  "h",
    "Be": "be",
    "C":  "c",
    "N":  "n",
    "O":  "o",
    "F":  "f",
    "Si": "si",
    "P":  "p",
    "S":  "s",
    "Cl": "cl",
    "Ca": "ca",
    "V":  "v",
    "Fe": "fe",
    "Cu": "cu",
    "Zn": "zn",
    "Ga": "ga",
    "As": "as",
    "Se": "se",
    "Br": "br",
    "Cd": "cd",
    "Hg": "hg",

    # ALANINA

    ("ALA", "N"):   "n-ammide",
    ("ALA", "H"):   "h-ammide",
    ("ALA", "CA"):  "c-alpha",
    ("ALA", "HA"):  "h-alpha",
    ("ALA", "CB"):  "c-beta-ter",
    ("ALA", "HB1"): "h-beta-metil",
    ("ALA", "HB2"): "h-beta-metil",
    ("ALA", "HB3"): "h-beta-metil",
    ("ALA", "C"):   "c-carbonilico",
    ("ALA", "O"):   "o-carbonilico",

    ("ALA", "H2"):  "h-zeta-ammina",
    ("ALA", "H3"):  "h-zeta-ammina",
    ("ALA", "OXT"): "o-delta-carbossile", # Selezionato per similarità
    
    # ARGININA

    ("ARG", "N"): "n-ammide",
    ("ARG", "H"): "h-ammide",
    ("ARG", "CA"): "c-alpha",
    ("ARG", "HA"): "h-alpha",
    ("ARG", "CB"): "c-beta",
    ("ARG", "HB2"): "h-beta-alifatico",
    ("ARG", "HB3"): "h-beta-alifatico",
    ("ARG", "CG"): "c-gamma",
    ("ARG", "HG2"): "h-gamma-alifatico",
    ("ARG", "HG3"): "h-gamma-alifatico",
    ("ARG", "CD"): "c-delta",
    ("ARG", "HD2"): "h-delta-alifatico",
    ("ARG", "HD3"): "h-delta-alifatico",
    ("ARG", "NE"): "n-epsilon-guanidinio",
    ("ARG", "HE"): "h-epsilon-guanidinio",
    ("ARG", "CZ"): "c-zeta-guanidinio",
    ("ARG", "NH1"): "n-eta-guanidinio",
    ("ARG", "HH11"): "h-eta-guanidinio",
    ("ARG", "HH12"): "h-eta-guanidinio",
    ("ARG", "NH2"): "n-eta-guanidinio",
    ("ARG", "HH21"): "h-eta-guanidinio",
    ("ARG", "HH22"): "h-eta-guanidinio",
    ("ARG", "C"): "c-carbonilico",
    ("ARG", "O"): "o-carbonilico",

    ("ARG", "H2"): "h-zeta-ammina",
    ("ARG", "H3"): "h-zeta-ammina",
    ("ARG", "HXT"): "h-delta-carbossile", # Selezionato per similarità
    ("ARG", "OXT"): "o-delta-carbossile", # Selezionato per similarità

    # ASPARAGINA

    ("ASN", "N"): "n-ammide",
    ("ASN", "H"): "h-ammide",
    ("ASN", "CA"): "c-alpha",
    ("ASN", "HA"): "h-alpha",
    ("ASN", "CB"): "c-beta",
    ("ASN", "HB2"): "h-beta-alifatico",
    ("ASN", "HB3"): "h-beta-alifatico",
    ("ASN", "CG"): "c-gamma-ammide",
    ("ASN", "OD1"): "o-delta-ammide",
    ("ASN", "ND2"): "n-delta-ammide",
    ("ASN", "HD21"): "h-delta-ammide",
    ("ASN", "HD22"): "h-delta-ammide",
    ("ASN", "C"): "c-carbonilico",
    ("ASN", "O"): "o-carbonilico",

    ("ASN", "H2"): "h-zeta-ammina",
    ("ASN", "H3"): "h-zeta-ammina",
    ("ASN", "OXT"): "o-delta-carbossile", # Selezionato per similarità

    # ASPARTATO

    ("ASP", "N"): "n-ammide",
    ("ASP", "H"): "h-ammide",
    ("ASP", "CA"): "c-alpha",
    ("ASP", "HA"): "h-alpha",
    ("ASP", "CB"): "c-beta",
    ("ASP", "HB2"): "h-beta-alifatico",
    ("ASP", "HB3"): "h-beta-alifatico",
    ("ASP", "CG"): "c-gamma-carbossile",
    ("ASP", "OD1"): "o-delta-carbossile",
    ("ASP", "OD2"): "o-delta-carbossile",
    ("ASP", "HD2"): "h-delta-carbossile",
    ("ASP", "C"): "c-carbonilico",
    ("ASP", "O"): "o-carbonilico",

    ("ASP", "H2"): "h-zeta-ammina",
    ("ASP", "H3"): "h-zeta-ammina",
    ("ASP", "OXT"): "o-delta-carbossile", # Selezionato per similarità

    # CISTEINA

    ("CYS", "N"): "n-ammide",
    ("CYS", "H"): "h-ammide",
    ("CYS", "CA"): "c-alpha",
    ("CYS", "HA"): "h-alpha",
    ("CYS", "CB"): "c-beta",
    ("CYS", "HB2"): "h-beta-alifatico",
    ("CYS", "HB3"): "h-beta-alifatico",
    ("CYS", "SG"): "s-gamma",
    ("CYS", "HG"): "h-gamma-solfuro",
    ("CYS", "C"): "c-carbonilico",
    ("CYS", "O"): "o-carbonilico",

    ("CYS", "H2"): "h-zeta-ammina",
    ("CYS", "H3"): "h-zeta-ammina",
    ("CYS", "HXT"): "h-delta-carbossile", # Selezionato per similarità
    ("CYS", "OXT"): "o-delta-carbossile", # Selezionato per similarità

    ("CYX", "N"): "n-ammide",
    ("CYX", "H"): "h-ammide",
    ("CYX", "CA"): "c-alpha",
    ("CYX", "HA"): "h-alpha",
    ("CYX", "CB"): "c-beta",
    ("CYX", "HB2"): "h-beta-alifatico",
    ("CYX", "HB3"): "h-beta-alifatico",
    ("CYX", "SG"): "s-gamma-ponte",
    ("CYX", "C"): "c-carbonilico",
    ("CYX", "O"): "o-carbonilico",

    ("CYX", "H2"): "h-zeta-ammina",
    ("CYX", "H3"): "h-zeta-ammina",
    ("CYX", "OXT"): "o-delta-carbossile", # Selezionato per similarità

    # GLUTAMMINA

    ("GLN", "N"): "n-ammide",
    ("GLN", "H"): "h-ammide",
    ("GLN", "CA"): "c-alpha",
    ("GLN", "HA"): "h-alpha",
    ("GLN", "CB"): "c-beta",
    ("GLN", "HB2"): "h-beta-alifatico",
    ("GLN", "HB3"): "h-beta-alifatico",
    ("GLN", "CG"): "c-gamma",
    ("GLN", "HG2"): "h-gamma-alifatico",
    ("GLN", "HG3"): "h-gamma-alifatico",
    ("GLN", "CD"): "c-delta-ammide",
    ("GLN", "OE1"): "o-epsilon-ammide",
    ("GLN", "NE2"): "n-epsilon-ammide",
#    ("GLN", "HE11"): "",
#    ("GLN", "HE12"): "",
    ("GLN", "HE21"): "h-epsilon-ammide",
    ("GLN", "HE22"): "h-epsilon-ammide",
    ("GLN", "C"): "c-carbonilico",
    ("GLN", "O"): "o-carbonilico",

    ("GLN", "H2"): "h-zeta-ammina",
    ("GLN", "H3"): "h-zeta-ammina",
    ("GLN", "OXT"): "o-delta-carbossile", # Selezionato per similarità

    # GLUTAMMATO

    ("GLU", "N"): "n-ammide",
    ("GLU", "H"): "h-ammide",
    ("GLU", "CA"): "c-alpha",
    ("GLU", "HA"): "h-alpha",
    ("GLU", "CB"): "c-beta",
    ("GLU", "HB2"): "h-beta-alifatico",
    ("GLU", "HB3"): "h-beta-alifatico",
    ("GLU", "CG"): "c-gamma",
    ("GLU", "HG2"): "h-gamma-alifatico",
    ("GLU", "HG3"): "h-gamma-alifatico",
    ("GLU", "CD"): "c-delta-carbossile",
    ("GLU", "OE1"): "o-epsilon-carbossile",
    ("GLU", "OE2"): "o-epsilon-carbossile",
    ("GLU", "HE2"): "h-epsilon-carbossile",
    ("GLU", "C"): "c-carbonilico",
    ("GLU", "O"): "o-carbonilico",

    ("GLU", "H2"): "h-zeta-ammina",
    ("GLU", "H3"): "h-zeta-ammina",
    ("GLU", "OXT"): "o-delta-carbossile", # Selezionato per similarità

    # GLICINA

    ("GLY", "N"): "n-ammide",
    ("GLY", "H"): "h-ammide",
    ("GLY", "CA"): "c-alpha",
    ("GLY", "HA2"): "h-alpha",
    ("GLY", "HA3"): "h-alpha",
    ("GLY", "C"): "c-carbonilico",
    ("GLY", "O"): "o-carbonilico",

    ("GLY", "H2"): "h-zeta-ammina",
    ("GLY", "H3"): "h-zeta-ammina",
    ("GLY", "OXT"): "o-delta-carbossile", # Selezionato per similarità

    # ISTIDINA

    ("HID", "N"): "n-ammide",
    ("HID", "H"): "h-ammide",
    ("HID", "CA"): "c-alpha",
    ("HID", "HA"): "h-alpha",
    ("HID", "CB"): "c-beta",
    ("HID", "HB2"): "h-beta-alifatico",
    ("HID", "HB3"): "h-beta-alifatico",
    ("HID", "CG"): "c-gamma-aromatico",
    ("HID", "ND1"): "n-delta-aromatico",
    ("HID", "HD1"): "h-delta-aromatico-n-imidazolo",
    ("HID", "CE1"): "c-epsilon-aromatico",
    ("HID", "HE1"): "h-epsilon-aromatico",
    ("HID", "NE2"): "n-epsilon-aromatico",
    ("HID", "CD2"): "c-delta-aromatico",
    ("HID", "HD2"): "h-delta-aromatico",
    ("HID", "C"): "c-carbonilico",
    ("HID", "O"): "o-carbonilico",

    ("HID", "H2"): "h-zeta-ammina",
    ("HID", "H3"): "h-zeta-ammina",
    ("HID", "OXT"): "o-delta-carbossile", # Selezionato per similarità

    ("HIE", "N"): "n-ammide",
    ("HIE", "H"): "h-ammide",
    ("HIE", "CA"): "c-alpha",
    ("HIE", "HA"): "h-alpha",
    ("HIE", "CB"): "c-beta",
    ("HIE", "HB2"): "h-beta-alifatico",
    ("HIE", "HB3"): "h-beta-alifatico",
    ("HIE", "CD2"): "c-delta-aromatico",
    ("HIE", "HD2"): "h-delta-aromatico",
    ("HIE", "CG"): "c-gamma-aromatico",
    ("HIE", "NE2"): "n-epsilon-aromatico",
    ("HIE", "HE2"): "h-epsilon-aromatico-n-imidazolo",
    ("HIE", "ND1"): "n-delta-aromatico",
    ("HIE", "CE1"): "c-epsilon-aromatico",
    ("HIE", "HE1"): "h-epsilon-aromatico",
    ("HIE", "C"): "c-carbonilico",
    ("HIE", "O"): "o-carbonilico",

    ("HIE", "H2"): "h-zeta-ammina",
    ("HIE", "H3"): "h-zeta-ammina",
    ("HIE", "OXT"): "o-delta-carbossile", # Selezionato per similarità

    ("HIS", "N"): "n-ammide",
    ("HIS", "H"): "h-ammide",
    ("HIS", "CA"): "c-alpha",
    ("HIS", "HA"): "h-alpha",
    ("HIS", "CB"): "c-beta",
    ("HIS", "HB2"): "h-beta-alifatico",
    ("HIS", "HB3"): "h-beta-alifatico",
    ("HIS", "CD2"): "c-delta-aromatico",
    ("HIS", "HD2"): "h-delta-aromatico",
    ("HIS", "CG"): "c-gamma-aromatico",
    ("HIS", "NE2"): "n-epsilon-aromatico",
    ("HIS", "HE2"): "h-epsilon-aromatico",
    ("HIS", "ND1"): "n-delta-aromatico",
    ("HIS", "HD1"): "h-delta-aromatico-n-imidazolo",
    ("HIS", "CE1"): "c-epsilon-aromatico",
    ("HIS", "HE1"): "h-epsilon-aromatico-n-imidazolo",
    ("HIS", "C"): "c-carbonilico",
    ("HIS", "O"): "o-carbonilico",

    ("HIS", "H2"): "h-zeta-ammina",
    ("HIS", "H3"): "h-zeta-ammina",
    ("HIS", "OXT"): "o-delta-carbossile", # Selezionato per similarità

    ("ILE", "N"): "n-ammide",
    ("ILE", "H"): "h-ammide",
    ("ILE", "CA"): "c-alpha",
    ("ILE", "HA"): "h-alpha",
    ("ILE", "CB"): "c-beta",
    ("ILE", "HB"): "h-beta-alifatico",
    ("ILE", "CG2"): "c-gamma-ter",
    ("ILE", "HG21"): "h-gamma-metil",
    ("ILE", "HG22"): "h-gamma-metil",
    ("ILE", "HG23"): "h-gamma-metil",
    ("ILE", "CG1"): "c-gamma",
    ("ILE", "HG12"): "h-gamma-alifatico",
    ("ILE", "HG13"): "h-gamma-alifatico",
    ("ILE", "CD1"): "c-delta-ter",
    ("ILE", "HD11"): "h-delta-metil",
    ("ILE", "HD12"): "h-delta-metil",
    ("ILE", "HD13"): "h-delta-metil",
    ("ILE", "C"): "c-carbonilico",
    ("ILE", "O"): "o-carbonilico",

    ("ILE", "H2"): "h-zeta-ammina",
    ("ILE", "H3"): "h-zeta-ammina",
    ("ILE", "OXT"): "o-delta-carbossile", # Selezionato per similarità

    # LEUCINA

    ("LEU", "N"): "n-ammide",
    ("LEU", "H"): "h-ammide",
    ("LEU", "CA"): "c-alpha",
    ("LEU", "HA"): "h-alpha",
    ("LEU", "CB"): "c-beta",
    ("LEU", "HB2"): "h-beta-alifatico",
    ("LEU", "HB3"): "h-beta-alifatico",
    ("LEU", "CG"): "c-gamma",
    ("LEU", "HG"): "h-gamma-alifatico",
    ("LEU", "CD1"): "c-delta-ter",
    ("LEU", "HD11"): "h-delta-metil",
    ("LEU", "HD12"): "h-delta-metil",
    ("LEU", "HD13"): "h-delta-metil",
    ("LEU", "CD2"): "c-delta-ter",
    ("LEU", "HD21"): "h-delta-metil",
    ("LEU", "HD22"): "h-delta-metil",
    ("LEU", "HD23"): "h-delta-metil",
    ("LEU", "C"): "c-carbonilico",
    ("LEU", "O"): "o-carbonilico",

    ("LEU", "H2"): "h-zeta-ammina",
    ("LEU", "H3"): "h-zeta-ammina",
    ("LEU", "OXT"): "o-delta-carbossile", # Selezionato per similarità

    # LISINA

    ("LYS", "N"): "n-ammide",
    ("LYS", "H"): "h-ammide",
    ("LYS", "CA"): "c-alpha",
    ("LYS", "HA"): "h-alpha",
    ("LYS", "CB"): "c-beta",
    ("LYS", "HB2"): "h-beta-alifatico",
    ("LYS", "HB3"): "h-beta-alifatico",
    ("LYS", "CG"): "c-gamma",
    ("LYS", "HG2"): "h-gamma-alifatico",
    ("LYS", "HG3"): "h-gamma-alifatico",
    ("LYS", "CD"): "c-delta",
    ("LYS", "HD2"): "h-delta-alifatico",
    ("LYS", "HD3"): "h-delta-alifatico",
    ("LYS", "CE"): "c-epsilon",
    ("LYS", "HE2"): "h-epsilon-alifatico",
    ("LYS", "HE3"): "h-epsilon-alifatico",
    ("LYS", "NZ"): "n-zeta-ammina",
    ("LYS", "HZ1"): "h-zeta-ammina",
    ("LYS", "HZ2"): "h-zeta-ammina",
    ("LYS", "HZ3"): "h-zeta-ammina",
    ("LYS", "C"): "c-carbonilico",
    ("LYS", "O"): "o-carbonilico",

    ("LYS", "H2"): "h-zeta-ammina",
    ("LYS", "H3"): "h-zeta-ammina",
    ("LYS", "OXT"): "o-delta-carbossile", # Selezionato per similarità

    # METIONINA

    ("MET", "N"): "n-ammide",
    ("MET", "H"): "h-ammide",
    ("MET", "CA"): "c-alpha",
    ("MET", "HA"): "h-alpha",
    ("MET", "CB"): "c-beta",
    ("MET", "HB2"): "h-beta-alifatico",
    ("MET", "HB3"): "h-beta-alifatico",
    ("MET", "CG"): "c-gamma",
    ("MET", "HG2"): "h-gamma-alifatico",
    ("MET", "HG3"): "h-gamma-alifatico",
    ("MET", "SD"): "s-delta",
    ("MET", "CE"): "c-epsilon-ter",
    ("MET", "HE1"): "h-epsilon-metil",
    ("MET", "HE2"): "h-epsilon-metil",
    ("MET", "HE3"): "h-epsilon-metil",
    ("MET", "C"): "c-carbonilico",
    ("MET", "O"): "o-carbonilico",

    ("MET", "H2"): "h-zeta-ammina",
    ("MET", "H3"): "h-zeta-ammina",
    ("MET", "OXT"): "o-delta-carbossile", # Selezionato per similarità

    # FENILALANINA

    ("PHE", "N"): "n-ammide",
    ("PHE", "H"): "h-ammide",
    ("PHE", "CA"): "c-alpha",
    ("PHE", "HA"): "h-alpha",
    ("PHE", "CB"): "c-beta",
    ("PHE", "HB2"): "h-beta-alifatico",
    ("PHE", "HB3"): "h-beta-alifatico",
    ("PHE", "CG"): "c-gamma-aromatico",
    ("PHE", "CD1"): "c-delta-aromatico",
    ("PHE", "HD1"): "h-delta-aromatico",
    ("PHE", "CE1"): "c-epsilon-aromatico",
    ("PHE", "HE1"): "h-epsilon-aromatico",
    ("PHE", "CZ"): "c-zeta-aromatico",
    ("PHE", "HZ"): "h-zeta-aromatico",
    ("PHE", "CD2"): "c-delta-aromatico",
    ("PHE", "HD2"): "h-delta-aromatico",
    ("PHE", "CE2"): "c-epsilon-aromatico",
    ("PHE", "HE2"): "h-epsilon-aromatico",
    ("PHE", "C"): "c-carbonilico",
    ("PHE", "O"): "o-carbonilico",

    ("PHE", "H2"): "h-zeta-ammina",
    ("PHE", "H3"): "h-zeta-ammina",
    ("PHE", "OXT"): "o-delta-carbossile", # Selezionato per similarità

    # PROLINA

    ("PRO", "N"): "n-ammide",
    ("PRO", "CD"): "c-delta-pro",
    ("PRO", "HD2"): "h-delta-alifatico-pro",
    ("PRO", "HD3"): "h-delta-alifatico-pro",
    ("PRO", "CA"): "c-alpha",
    ("PRO", "HA"): "h-alpha",
    ("PRO", "CB"): "c-beta",
    ("PRO", "HB2"): "h-beta-alifatico",
    ("PRO", "HB3"): "h-beta-alifatico",
    ("PRO", "CG"): "c-gamma",
    ("PRO", "HG2"): "h-gamma-alifatico",
    ("PRO", "HG3"): "h-gamma-alifatico",
    ("PRO", "C"): "c-carbonilico",
    ("PRO", "O"): "o-carbonilico",

    ("PRO", "H2"): "h-zeta-ammina",
    ("PRO", "H3"): "h-zeta-ammina",
    ("PRO", "OXT"): "o-delta-carbossile", # Selezionato per similarità

    # SERINA

    ("SER", "N"): "n-ammide",
    ("SER", "H"): "h-ammide",
    ("SER", "CA"): "c-alpha",
    ("SER", "HA"): "h-alpha",
    ("SER", "CB"): "c-beta",
    ("SER", "HB2"): "h-beta-alifatico",
    ("SER", "HB3"): "h-beta-alifatico",
    ("SER", "OG"): "o-gamma-alcol",
    ("SER", "HG"): "h-gamma-alcol",
    ("SER", "C"): "c-carbonilico",
    ("SER", "O"): "o-carbonilico",

    ("SER", "H2"): "h-zeta-ammina",
    ("SER", "H3"): "h-zeta-ammina",
    ("SER", "HXT"): "h-delta-carbossile", # Selezionato per similarità
    ("SER", "OXT"): "o-delta-carbossile", # Selezionato per similarità

    # TREONINA

    ("THR", "N"): "n-ammide",
    ("THR", "H"): "h-ammide",
    ("THR", "CA"): "c-alpha",
    ("THR", "HA"): "h-alpha",
    ("THR", "CB"): "c-beta",
    ("THR", "HB"): "h-beta-alifatico",
    ("THR", "OG1"): "o-gamma-alcol",
    ("THR", "HG1"): "h-gamma-alcol",
    ("THR", "CG2"): "c-gamma-ter",
    ("THR", "HG21"): "h-gamma-metil",
    ("THR", "HG22"): "h-gamma-metil",
    ("THR", "HG23"): "h-gamma-metil",
    ("THR", "C"): "c-carbonilico",
    ("THR", "O"): "o-carbonilico",

    ("THR", "H2"): "h-zeta-ammina",
    ("THR", "H3"): "h-zeta-ammina",
    ("THR", "OXT"): "o-delta-carbossile", # Selezionato per similarità

    # TRIPTOFANO

    ("TRP", "N"): "n-ammide",
    ("TRP", "H"): "h-ammide",
    ("TRP", "CA"): "c-alpha",
    ("TRP", "HA"): "h-alpha",
    ("TRP", "CB"): "c-beta",
    ("TRP", "HB2"): "h-beta-alifatico",
    ("TRP", "HB3"): "h-beta-alifatico",
    ("TRP", "CG"): "c-gamma-aromatico",
    ("TRP", "CD1"): "c-delta-aromatico",
    ("TRP", "HD1"): "h-delta-aromatico",
    ("TRP", "NE1"): "n-epsilon-aromatico",
    ("TRP", "HE1"): "h-epsilon-aromatico-n-indolo",
    ("TRP", "CE2"): "c-epsilon-aromatico",
    ("TRP", "CD2"): "c-delta-aromatico",
    ("TRP", "CE3"): "c-delta-aromatico",
    ("TRP", "HE3"): "h-epsilon-aromatico",
    ("TRP", "CZ3"): "c-zeta-aromatico",
    ("TRP", "HZ3"): "h-zeta-aromatico",
    ("TRP", "CZ2"): "c-zeta-aromatico",
    ("TRP", "HZ2"): "h-zeta-aromatico",
    ("TRP", "CH2"): "c-eta-aromatico",
    ("TRP", "HH2"): "h-eta-aromatico",
    ("TRP", "C"): "c-carbonilico",
    ("TRP", "O"): "o-carbonilico",

    ("TRP", "H2"): "h-zeta-ammina",
    ("TRP", "H3"): "h-zeta-ammina",
    ("TRP", "OXT"): "o-delta-carbossile", # Selezionato per similarità

    # TIROSINA

    ("TYR", "N"): "n-ammide",
    ("TYR", "H"): "h-ammide",
    ("TYR", "CA"): "c-alpha",
    ("TYR", "HA"): "h-alpha",
    ("TYR", "CB"): "c-beta",
    ("TYR", "HB2"): "h-beta-alifatico",
    ("TYR", "HB3"): "h-beta-alifatico",
    ("TYR", "CG"): "c-gamma-aromatico",
    ("TYR", "CD1"): "c-delta-aromatico",
    ("TYR", "HD1"): "h-delta-aromatico",
    ("TYR", "CE1"): "c-epsilon-aromatico",
    ("TYR", "HE1"): "h-epsilon-aromatico",
    ("TYR", "CZ"): "c-zeta-aromatico",
    ("TYR", "OH"): "o-eta-tyr",
    ("TYR", "HH"): "h-eta-alcol-tyr",
    ("TYR", "CD2"): "c-delta-aromatico",
    ("TYR", "HD2"): "h-delta-aromatico",
    ("TYR", "CE2"): "c-epsilon-aromatico",
    ("TYR", "HE2"): "h-epsilon-aromatico",
    ("TYR", "C"): "c-carbonilico",
    ("TYR", "O"): "o-carbonilico",

    ("TYR", "H2"): "h-zeta-ammina",
    ("TYR", "H3"): "h-zeta-ammina",
    ("TYR", "OXT"): "o-delta-carbossile", # Selezionato per similarità

    # VALINA

    ("VAL", "N"): "n-ammide",
    ("VAL", "H"): "h-ammide",
    ("VAL", "CA"): "c-alpha",
    ("VAL", "HA"): "h-alpha",
    ("VAL", "CB"): "c-beta",
    ("VAL", "HB"): "h-beta-alifatico",
    ("VAL", "CG1"): "c-gamma-ter",
    ("VAL", "HG11"): "h-gamma-metil",
    ("VAL", "HG12"): "h-gamma-metil",
    ("VAL", "HG13"): "h-gamma-metil",
    ("VAL", "CG2"): "c-gamma-ter",
    ("VAL", "HG21"): "h-gamma-metil",
    ("VAL", "HG22"): "h-gamma-metil",
    ("VAL", "HG23"): "h-gamma-metil",
    ("VAL", "C"): "c-carbonilico",
    ("VAL", "O"): "o-carbonilico",

    ("VAL", "H2"): "h-zeta-ammina",
    ("VAL", "H3"): "h-zeta-ammina",
    ("VAL", "OXT"): "o-delta-carbossile", # Selezionato per similarità

    ("ACE", "H1"): "h-metil-cap",
    ("ACE", "CH3"): "c-metil-cap",
    ("ACE", "H2"): "h-metil-cap",
    ("ACE", "H3"): "h-metil-cap",
    ("ACE", "C"): "c-carbonilico",
    ("ACE", "O"): "o-carbonilico",

    ("NME", "N"): "n-ammide",
    ("NME", "H"): "h-ammide",
    ("NME", "CH3"): "c-metil-cap",
    ("NME", "HH31"): "h-metil-cap",
    ("NME", "HH32"): "h-metil-cap",
    ("NME", "HH33"): "h-metil-cap",

}