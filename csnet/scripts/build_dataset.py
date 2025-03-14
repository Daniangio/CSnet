#!/usr/bin/env python3
import argparse
from csnet.utils.dataset import NMRDatasetBuilder

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Build NMR Dataset from configuration files')
    
    # Required arguments
    parser.add_argument('-c',
                        '--config', 
                        type=str, 
                        required=True,
                        help='Path to atom type configuration YAML file')
    
    parser.add_argument('-n',
                        '--nmr2pdb', 
                        type=str,
                        required=True,
                        help='Path to nmr2pdb CSV file')
    
    # Optional arguments
    parser.add_argument('-m',
                        '--max-structures', 
                        type=int, 
                        default=None, 
                        help='Maximum number of structures to process (default: None, take all structures)')
    
    parser.add_argument('-d',
                        '--data-root', 
                        type=str, 
                        default='./', 
                        help='Output folder path (default: "./")')
    
    parser.add_argument('-a',
                        '--atom-type-map', 
                        type=str, 
                        default=None, 
                        help='Folder path where to find atom_type_assigner.yaml (default: None == use data_root)')
    
    # Parse arguments
    args = parser.parse_args()

    # Initialize the NMRDataset with the config file
    dataset_builder = NMRDatasetBuilder(args.config, args.atom_type_map)
    options = {"ignore_dihedrals_concat": True}

    # Build the dataset
    dataset_builder.build(args.nmr2pdb, slicing=slice(0, args.max_structures), data_root=args.data_root, options=options)
    dataset_builder.filter_npz_datasets(data_root=args.data_root)
    dataset_builder.build_statistics(data_root=args.data_root)
    dataset_builder.extract_outliers()
    dataset_builder.remove_outliers(data_root=args.data_root)
    dataset_builder.build_statistics(data_root=args.data_root, rebuild=True)
    dataset_builder.build_config_params(data_root=args.data_root)

if __name__ == "__main__":
    main()