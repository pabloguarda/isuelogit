from __future__ import annotations

import os
from printer import blockPrinting

#============================================================================
# FILEPATHS
#=============================================================================


# @blockPrinting
def set_main_dir(dir = None):

    if dir is None:
        # Set working directory
        # os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        dir = os.getcwd()

    # print('main dir:', dir)

    dirs = {'main_dir': dir}

    # Paths to folders or files
    dirs = {'folder_pablo_networks': dirs['main_dir'] + "/input/public/networks/pablo/",
            'folder_tntp_networks': dirs['main_dir'] + "/input/public/networks/github/",
            'Colombus_network':"/Users/pablo/OneDrive/university/cmu/2-research/datasets/private/bin-networks/Columbus",
            'Fresno_network': '/Users/pablo/OneDrive/university/cmu/2-research/datasets/private/od-fresno-sac/SR41',
            'Sacramento_network': '/Users/pablo/OneDrive/university/cmu/2-research/datasets/private/od-fresno-sac/sac',
            'Fresno_Sac_networks': '/Users/pablo/OneDrive/university/cmu/2-research/datasets/private/od-fresno-sac',

            # Folder to read and write data
            'input_folder': dirs['main_dir'] + "/input/",
            'output_folder': dirs['main_dir'] + "/output/",
            }

    # Folder to read and write network data
    dirs['read_network_data'] = dirs['input_folder'] + "/network-data/"
    dirs['write_network_data'] = dirs['output_folder'] + "/network-data/"

    return dirs

dirs = set_main_dir()

# =============================================================================
# GIS OPTIONS
# =============================================================================

gis_options= {'crs_ca':2228,
              'crs_mercator': 4326}

# gis_options= {'crs_ca':2228,
#               'crs_mercator': 4326,
#               'inrix_matching': {},
#               'data_processing': {},
#               'buffer_size': {},
#               'matching_stats':{}
#               }



