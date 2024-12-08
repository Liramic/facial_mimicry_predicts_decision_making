import os
from General.HelperFunctions import GetSessions, cleanSpace
import numpy as np
from General.init import *
import pandas as pd
from PsychData.LoadChoicesJson import loadChoiceFromSession
from tqdm import tqdm
from General.HelperFunctions import getPathsFromSessionFolder
from EDF.EdfAnalyzer import EdfAnalyzer
from ComponentAnalysis.ComponentExtractor import open_w_if_exisis, open_whiten_if_exisis, load_special_components_if_exists, get_special_component_indexes, GetLipCornerDownPuller, GetComponentByElecrodesNumber
from Analysis.original_actress_readings import export_actress_chunked_data_with_dwt
from scipy import signal
from ComponentAnalysis.Denoise import wavelet_denoising, whiten, center

def run_emg_analysis(main_corrolation_func, rms_size_ms, downsampleWindowInMs, headers, csv_out_path, with_actress=False):
    sessions_list = GetSessions(data_path)
    all_correlations = []

    if (with_actress):
        actress_chunks = export_actress_chunked_data_with_dwt(downsampleWindowInMs, rms_size_ms)
    
    for session in tqdm(sessions_list):
        breaked = False
        print(session)
        both_componenets = [[],[]]
        both_chunks = [[],[]]

        path_and_corrections = getPathsFromSessionFolder(os.path.join(data_path, session))
        for p in [A,B]:
            Y, chunks, freq = EdfAnalyzer.Read(path_and_corrections[p][0], path_and_corrections[p][1], downsampleWindowInMs)
            Y = wavelet_denoising(Y, freq, 'db15')
            Y, _ = center(Y)
            
            whiten_mat = open_whiten_if_exisis(session, p, dwt=True)
            if (whiten_mat is None):
                Y = whiten(Y)
            else:
                Y = np.dot(whiten_mat, Y)

            w = open_w_if_exisis(session, p, dwt=True)
            
            if ( w is None):
                w, independentComponents = EdfAnalyzer.ICA(Y, 16, whiten=False)
                #save_new_w(session, p, w)
            else:
                independentComponents = np.matmul(w,Y)


            independentComponents = EdfAnalyzer.window_rms(independentComponents, rms_size_ms, freq)

            hz = int(1000/downsampleWindowInMs)
            independentComponents = signal.resample(independentComponents, int(independentComponents.shape[1] / freq * hz), axis=1)

            #select components:
            special_comps_index = load_special_components_if_exists(data_path, session, p, dwt=True)
            if (special_comps_index is None):
                special_comps_index = get_special_component_indexes(w)

            independentComponents = independentComponents[special_comps_index, :]

            both_chunks[p] = EdfAnalyzer.combine_callibration_ticks(chunks)
            both_componenets[p] = independentComponents
            
        if (breaked):
            continue
        
        if ( with_actress ):
            corr_results = main_corrolation_func(session, both_componenets, both_chunks, actress_chunks)
        else:
            corr_results = main_corrolation_func(session, both_componenets, both_chunks)

        choices = loadChoiceFromSession(data_path, session)
        for i in range(len(corr_results)):
            result = corr_results[i]
            key = result[1]
            key_splits = key.split("_")
            if ( len(key_splits) > 1):
                story_number = key_splits[1]
                if ( story_number != ''):
                    story_number = story_number.split('.')[0]
                    for choice in choices:
                        if ( story_number == choice[0] ):
                            corr_results[i] = result + choice[1:]
                            break

        all_correlations += corr_results
        del independentComponents
        cleanSpace()
        print("finished session " + session)

    if (csv_out_path!= None):
        df = pd.DataFrame(all_correlations)
        df.columns = headers
        df.to_csv(csv_out_path)