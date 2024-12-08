import numpy as np
from numpy.linalg import inv
from scipy.stats import zscore
import os
from General.HelperFunctions import particiapnt_to_char
from General.init import *

def insert_zeros_in_removed_places(vector, electrodes_to_remove):
    for electrode in electrodes_to_remove:
        vector = np.insert(vector, electrode, 0)
    return vector

def find_first_match_index(vector, value):
    for i, v in enumerate(vector):
        if v == value:
            return i
    return -1

def find_match_indexes(vector, value):
    indexes = []
    for i, v in enumerate(vector):
        if v == value:
            indexes.append(i)
    return indexes

def find_first_match_index_with_prioritized_values(vector, values):
    for value in values:
        index = find_first_match_index(vector, value)
        if index != -1:
            return index
    return -1

def find_matched_index(vector, values, new_w):
    for value in values:
        indexes = find_match_indexes(vector, value)
        if ( len(indexes) == 0):
            continue
        if ( len(indexes) == 1):
            return indexes[0]
        
        scores = [zscore(new_w[:,index])[value-1] for index in indexes]
        return indexes[np.argmax(scores)]
    return -1

def find_matched_index_zscore_method(new_w, value):
    return np.argmax(zscore(new_w, axis=0)[value-1])

def GetComponentByElecrodesNumber(w, electrodes_channel_defintion, removed_electrodes_zero_based=[]):
    number_of_channels = 16 - len(removed_electrodes_zero_based)
    # calculations for the heatmap
    inverse = np.absolute(inv(w)) 
    new_w = np.zeros((16,number_of_channels))
    for i in range(number_of_channels):
        new_w[:,i] = insert_zeros_in_removed_places(inverse[:,i], removed_electrodes_zero_based)

    #value to send:
    chosen_electrode_definition = electrodes_channel_defintion[0]
    for value in electrodes_channel_defintion:
        zero_based_value = value - 1
        if ( zero_based_value in removed_electrodes_zero_based):
            continue
        else:
            chosen_electrode_definition = value
            break
    
    return find_matched_index_zscore_method(new_w, chosen_electrode_definition)

def GetNoseComponent(w, removed_electrodes_zero_based):
    return GetComponentByElecrodesNumber(w, [9, 8], removed_electrodes_zero_based)

def GetOrbicularisOculiComponent(w, removed_electrodes_zero_based):
    return GetComponentByElecrodesNumber(w, [10,11,12], removed_electrodes_zero_based)

def GetZygomaticusComponent(w, removed_electrodes_zero_based):
    return GetComponentByElecrodesNumber(w, [6, 5, 7, 3, 4], removed_electrodes_zero_based)

def GetFrontalisComponent(w, removed_electrodes_zero_based):
    return GetComponentByElecrodesNumber(w, [14, 13, 15, 16], removed_electrodes_zero_based)

def GetPlatysimaComponent(w, removed_electrodes_zero_based):
    return GetComponentByElecrodesNumber(w, [2], removed_electrodes_zero_based)

def GetRisorius(w, removed_electrodes_zero_based):
    a = GetComponentByElecrodesNumber(w, [4], removed_electrodes_zero_based)
    if ( a == -1):
        return 0
    return a


def get_special_component_indexes(w, list_of_faulty_channels=[]):
    special_comps_getters = [GetNoseComponent, GetOrbicularisOculiComponent, GetZygomaticusComponent, GetFrontalisComponent, GetPlatysimaComponent, GetRisorius]
    return [getter(w, list_of_faulty_channels) for getter in special_comps_getters]

def get_special_component_names():
    return ["nasalis", "orbicularis_oculi", "zygomaticus_major", "frontalis", "platysima", "risorius"]

def get_component_analysis_folder(data_path, session, p, dwt=False):
    if (dwt):
        return os.path.join(data_path, session, "DWT_hila_db15", particiapnt_to_char(p))
    return os.path.join(data_path, session, "IC_w_and_photos", particiapnt_to_char(p))

def open_w_if_exisis(session, p, dwt=False):
    try:
        return np.load(os.path.join(get_component_analysis_folder(data_path, session, p, dwt), "w.npy"))
    except:
        return None

def open_whiten_if_exisis(session, p, dwt=False):
    try:
        return np.load(os.path.join(get_component_analysis_folder(data_path, session, p, dwt), "white.npy"))
    except:
        return None

def save_new_w(session, p, w):
    folder = get_component_analysis_folder(data_path, session, p)
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "w.npy"), w)

def load_special_components_if_exists(data_path, session, p, dwt=False):
    folder = get_component_analysis_folder(data_path, session, p, dwt)
    file_path = os.path.join(folder, "components_chosen.txt")
    #if file exists:
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return list(map(int, file.read().split("\n")))
    return None