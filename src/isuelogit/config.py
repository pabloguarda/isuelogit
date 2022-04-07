from __future__ import annotations

import os
import psutil
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

# =============================================================================
# Memory usage
# =============================================================================

def get_cpu_usage_pct():
    """
    Obtains the system's average CPU load as measured over a period of 500 milliseconds.
    :returns: System CPU load as a percentage.
    :rtype: float
    """
    return psutil.cpu_percent(interval=0.5)


def get_cpu_frequency():
    """
    Obtains the real-time value of the current CPU frequency.
    :returns: Current CPU frequency in MHz.
    :rtype: int
    """
    return int(psutil.cpu_freq().current)

def get_cpu_temp():
    """
    Obtains the current value of the CPU temperature.
    :returns: Current value of the CPU temperature if successful, zero value otherwise.
    :rtype: float
    """
    # Initialize the result.
    result = 0.0
    # The first line in this file holds the CPU temperature as an integer times 1000.
    # Read the first line and remove the newline character at the end of the string.
    if os.path.isfile('/sys/class/thermal/thermal_zone0/temp'):
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            line = f.readline().strip()
        # Test if the string is an integer as expected.
        if line.isdigit():
            # Convert the string with the CPU temperature to a float in degrees Celsius.
            result = float(line) / 1000
    # Give the result back to the caller.
    return result

def get_ram_usage():
    """
    Obtains the absolute number of RAM bytes currently in use by the system.
    :returns: System RAM usage in bytes.
    :rtype: int
    """
    return int(psutil.virtual_memory().total - psutil.virtual_memory().available)

def get_ram_total():
    """
    Obtains the total amount of RAM in bytes available to the system.
    :returns: Total system RAM in bytes.
    :rtype: int
    """
    return int(psutil.virtual_memory().total)

def get_ram_usage_pct():
    """
    Obtains the system's current RAM usage.
    :returns: System RAM usage as a percentage.
    :rtype: float
    """
    return psutil.virtual_memory().percent


def get_swap_usage():
    """
    Obtains the absolute number of Swap bytes currently in use by the system.
    :returns: System Swap usage in bytes.
    :rtype: int
    """
    return int(psutil.swap_memory().used)

def get_swap_usage_pct():
    """
    Obtains the system's current Swap usage.
    :returns: System Swap usage as a percentage.
    :rtype: float
    """
    return psutil.swap_memory().percent