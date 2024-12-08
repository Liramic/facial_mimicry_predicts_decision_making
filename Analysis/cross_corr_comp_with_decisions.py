import os
import numpy as np
from General.init import *
from General.CorrolationMethods import get_max_xcorr_pearson_1d
from ComponentAnalysis.ComponentExtractor import get_special_component_names
from Analysis.analysis_runner import run_emg_analysis

rms_size_ms = 50
downsampleWindowInMs = 5
windowForCorrInMs = 3000
max_lag_ms = 3000
numOfPixelsForXCorrWindow = windowForCorrInMs // downsampleWindowInMs
numOfPixelsForXCorrStep = numOfPixelsForXCorrWindow

def get_vector_len(vec):
    if len(vec.shape) == 1:
        return vec.shape[0]
    return vec.shape[1]

def get_minimal_vecs_len(vec1, vec2):
    return min(get_vector_len(vec1), get_vector_len(vec2))

def make_vectors_comperable(a, b):
    min_length = get_minimal_vecs_len(a, b)
    if len(a.shape) == 1:
        return a[:min_length], b[:min_length]
    return a[:, :min_length], b[:, :min_length]

def corelation_lags_measurments(correlations, lags):
    return [np.mean(correlations), np.mean(lags), np.max(correlations)]

def corelation_lags_measurments_headers(title):
    return [f"{title}_mean_corr", f"{title}_mean_lag", f"{title}_max_corr"]


def add_mean_component_headers():
    res = [f"mean_{comp}_{p}" for comp in get_special_component_names() for p in ["A", "B"]] +\
            [f"max_{comp}_{p}" for comp in get_special_component_names() for p in ["A", "B"]]
    return res

def add_mean_components(mat_a, mat_b):
    mats = [np.mean(mat_a, axis=1), np.mean(mat_b, axis=1)]
    maxs = [np.max(mat_a, axis=1), np.max(mat_b, axis=1)]
    size = len(mats[0])
    res = [mats[p][i] for i in range(size) for p in [A, B]] + \
          [maxs[p][i] for i in range(size) for p in [A, B]]
    return res

def wrap_in_default_headers(headers):
    return ["session", "key", "action"] + headers + ['whoseReading', 'isChoiceA', 'isChoiceB', 'rtChoiceA', 'rtChoiceB', 'isOther', 'ChoseTogether']

def get_main_corrolation_of_single_comps_headers():
    comps_names = get_special_component_names()
    headers = []
    for comp in comps_names:
        headers += corelation_lags_measurments_headers(comp)
    headers += add_mean_component_headers()
    return headers

def main_corrolation_of_single_comps(current_session, independentComponents, chunks):
    results = []
    reading_order = [A, A, B, B, A, A, B, B]

    n = independentComponents[A].shape[0]

    for key in chunks[0]:
        chunk_data = [[], []]
        action = key.split("_")[0]
        current_reader = None
        if action.lower() == "reading":
            current_reader = reading_order.pop(0)
        else:
            continue

        for participant in [A, B]:
            chunk = chunks[participant][key]
            chunk_start = int(chunk.Start.Time)
            chunk_end = int(chunk.End.Time)
            chunk_data[participant] = independentComponents[participant][:, chunk_start:chunk_end]

        mat1 = chunk_data[A]
        mat2 = chunk_data[B]

        mean_comps_measurements = add_mean_components(chunk_data[A], chunk_data[B])

        max_len = min(mat1.shape[1], mat2.shape[1])

        corrs = [[] for i in range(n)]
        lags = [[] for i in range(n)]

        for s in range(0, max_len - numOfPixelsForXCorrWindow, numOfPixelsForXCorrStep):
            e = s + numOfPixelsForXCorrWindow
            direction = "bi-directional"
            if action.lower() == "reading":
                direction = "one-direction"
            for i in range(n):
                if action.lower() == "reading" and current_reader == A:
                    max_corr, max_ind = get_max_xcorr_pearson_1d(mat2[i], mat1[i], max_lag_ms,
                                                        downsampleWindowInMs, s, e, direction, weighted_corr=False, lowess=False)
                else:
                    max_corr, max_ind = get_max_xcorr_pearson_1d(mat1[i], mat2[i], max_lag_ms,
                                                        downsampleWindowInMs, s, e, direction, weighted_corr=False, lowess=False)
                corrs[i].append(max_corr)
                lags[i].append(max_ind)

        cor_results = []
        for i in range(n):
            cor_results += corelation_lags_measurments(corrs[i], lags[i])

        current_result = [current_session, key, action] + \
            cor_results + \
            mean_comps_measurements
        
        results.append(current_result)
    return results

def corrolation_each_comp_with_actress_headers():
    return corr_lag_measurments_n_headers("A_B") + \
    corr_lag_measurments_n_headers("actress_A") + \
    corr_lag_measurments_n_headers("actress_B") + \
    add_mean_component_headers(True) + \
    [f"mean_actress_{comp}" for comp in get_special_component_names()]

def corr_lag_measurments_n_headers(title):
    comps_names = get_special_component_names()
    res_headers = []
    for i in range(len(comps_names)):
        res_headers += corelation_lags_measurments_headers(f"{title}_{comps_names[i]}")
    return res_headers

def corr_lag_measurments_n(corrs, lags):
    results = []
    for i in range(len(corrs)):
        results += corelation_lags_measurments(corrs[i], lags[i])
    return results

def corrolation_each_comp_with_actress(current_session, independentComponents, chunks, actress_chunks):
    results = []

    for key in chunks[0]:
        chunk_data = [[], []]
        action = key.split("_")[0]

        if action.lower() != "listening":
            continue

        story_number = key.split("_")[1].split(".")[0]
        mat_actress = actress_chunks[story_number].data

        for participant in [A, B]:
            chunk = chunks[participant][key]
            chunk_start = int(chunk.Start.Time)
            chunk_end = int(chunk.End.Time)
            chunk_data[participant] = independentComponents[participant][:, chunk_start:chunk_end]
        
        mean_comps_measurements = add_mean_components(chunk_data[A], chunk_data[B])

        mat1 = chunk_data[A]
        mat2 = chunk_data[B]

        num_components = mat1.shape[0]

        max_len = min(mat1.shape[1], mat2.shape[1])

        corrs = [[] for i in range(num_components)]
        lags = [[] for i in range(num_components)]

        for s in range(0, max_len - numOfPixelsForXCorrWindow, numOfPixelsForXCorrStep):
            e = s + numOfPixelsForXCorrWindow

            direction = "bi-directional"

            for i in range(num_components):
                max_corr, max_ind = get_max_xcorr_pearson_1d(mat1[i], mat2[i], max_lag_ms,
                                                    downsampleWindowInMs, s, e, direction, weighted_corr=False, lowess=False)
                corrs[i].append(max_corr)
                lags[i].append(max_ind)

        corrs_A_with_actress = [[] for i in range(num_components)]
        lags_A_with_actress = [[] for i in range(num_components)]

        corrs_B_with_actress = [[] for i in range(num_components)]
        lags_B_with_actress = [[] for i in range(num_components)]

        corrs_with_actress = [corrs_A_with_actress, corrs_B_with_actress]
        lags_with_actress = [lags_A_with_actress, lags_B_with_actress]

        for p, mat in zip([A, B], [mat1, mat2]):
            max_len = min(mat.shape[1], mat_actress.shape[1])
            for s in range(0, max_len - numOfPixelsForXCorrWindow, numOfPixelsForXCorrStep):
                e = s + numOfPixelsForXCorrWindow
                direction = "bi-directional"
                for i in range(num_components):
                    max_corr, max_ind = get_max_xcorr_pearson_1d(mat[i], mat_actress[i], max_lag_ms,
                                                        downsampleWindowInMs, s, e, direction, weighted_corr=False, lowess=False)
                    corrs_with_actress[p][i].append(max_corr)
                    lags_with_actress[p][i].append(max_ind)
            
        current_result = [current_session, key, action] + \
        corr_lag_measurments_n(corrs, lags) + \
        corr_lag_measurments_n(corrs_with_actress[A], lags_with_actress[A]) + \
        corr_lag_measurments_n(corrs_with_actress[B], lags_with_actress[B]) + \
        mean_comps_measurements + \
        [np.mean(mat_actress[i]) for i in range(num_components)]
        
        results.append(current_result)
    return results

def main_3():
    run_emg_analysis(main_corrolation_of_single_comps, rms_size_ms, downsampleWindowInMs, windowForCorrInMs, numOfPixelsForXCorrStep, wrap_in_default_headers(get_main_corrolation_of_single_comps_headers(True)), os.path.join(data_path, "results_2024", f"RMS_{rms_size_ms}_downsample_{downsampleWindowInMs}_window_{int(windowForCorrInMs/1000)}_maxlag_{int(max_lag_ms/1000)}.csv"))

def main_corr_with_actress_dwt():
    run_emg_analysis(corrolation_each_comp_with_actress, rms_size_ms, downsampleWindowInMs, windowForCorrInMs, numOfPixelsForXCorrStep, wrap_in_default_headers(corrolation_each_comp_with_actress_headers()), os.path.join(data_path, "results_2024", f"RMS_{rms_size_ms}_downsample_{downsampleWindowInMs}_window_{int(windowForCorrInMs/1000)}_maxlag_{int(max_lag_ms/1000)}_with_actress.csv"), with_actress=True)
