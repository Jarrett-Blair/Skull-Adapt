# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 12:48:20 2024

@author: blair
"""
import os
import shutil

# Replace "_" with "-" for scans, and change parts[:2] to parts[:3]
def get_collection_num(filename):
    # Split the filename using underscores
    parts = filename.split('_')
    
    # Combine the first three parts to get Genus-species_collectionnumber
    genus_species_collection = '_'.join(parts[:2])
    
    return genus_species_collection

def get_unique_combinations(directory):
    # Get all file names in the directory
    files = os.listdir(directory)
    
    # Extract Genus-species_collectionnumber from each filename
    combinations = [get_collection_num(file) for file in files]
    
    # Get unique combinations
    unique_combinations = list(set(combinations))
    
    return unique_combinations

parent_dir = r"C:\Users\blair\OneDrive - UBC\Skulls\test_split1"
species = os.listdir(parent_dir)
sp_dict = {key: [] for key in species}

for sp in species:
    sub_dir = os.path.join(parent_dir, sp)
    sp_dict[sp] = get_unique_combinations(sub_dir)
    

sub_sp_dict = {key: original_list[4:6] for key, original_list in sp_dict.items()}

def copy_files(src_dir, dest_dir, sub_sp_dict):

    # Iterate through each subdirectory in the dictionary
    for sp, indivs in sub_sp_dict.items():
        src_subdir = os.path.join(src_dir, sp)

        # Iterate through each file in the subdirectory
        for filename in os.listdir(src_subdir):
            # Check if any subsection is present in the filename
            if any(indiv in filename for indiv in indivs):
                source_path = os.path.join(src_subdir, filename)
                destination_path = os.path.join(dest_dir, sp, filename)

                # Copy the file to the destination directory
                shutil.copy2(source_path, destination_path)
                

dest_dir = r"C:\Users\blair\OneDrive - UBC\Skulls\test-split1.5"

copy_files(parent_dir, dest_dir, sub_sp_dict)



## This section is to see how many files will be moved
# from collections import defaultdict

# def count_files(parent_dir, sp_dict):
#     # Create a defaultdict to store the count for each subsection
#     subsection_count = defaultdict(int)

#     # Iterate through each subdirectory in the dictionary
#     for subdirectory, subsections in sp_dict.items():
#         source_subdirectory = os.path.join(parent_dir, subdirectory)

#         # Iterate through each file in the subdirectory
#         for filename in os.listdir(source_subdirectory):
#             # Check if any subsection is present in the filename
#             if any(subsection in filename for subsection in subsections):
#                 # Increment the count for the current subsection
#                 subsection_count[subdirectory] += 1

#     return subsection_count

# sub_sp_dict = {key: original_list[4:6] for key, original_list in sp_dict.items()}
# file_count = count_files(parent_dir, sub_sp_dict)


## This sees how many files were actually moved
# def count_files_in_subdirectories(directory_path):
#     # Get a list of subdirectories
#     subdirectories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]

#     # Dictionary to store the counts
#     file_counts = {}

#     # Iterate through subdirectories and count files
#     for subdirectory in subdirectories:
#         subdirectory_path = os.path.join(directory_path, subdirectory)
#         files_in_subdirectory = [f for f in os.listdir(subdirectory_path) if os.path.isfile(os.path.join(subdirectory_path, f))]
#         file_counts[subdirectory] = len(files_in_subdirectory)

#     return file_counts

# # Example usage:
# directory_path = dest_dir
# result = count_files_in_subdirectories(directory_path)