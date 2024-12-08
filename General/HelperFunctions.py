import os
from General.init import *
import gc
import json
import numpy as np
import re

def toFilePath(sessionFolder, fname):
    return os.path.join(sessionFolder, fname)

def getPathsFromSessionFolder(sessionFolder, correctionA = 0.4, correctionB = 0.4):
    files = [x for x in os.listdir(sessionFolder) if x.endswith(".edf")]
    fileA = ""
    fileB = ""
    if ( "_a_" in files[0].lower()):
        fileA = files[0]
        fileB = files[1]
    else:
        fileA = files[1]
        fileB = files[0]
    
    return [(toFilePath(sessionFolder, fileA), correctionA),(toFilePath(sessionFolder, fileB), correctionB)]

def getFirstCsvFile(path):
    for x in os.listdir(path):
        if x.endswith(".csv"):
            return toFilePath(path, x)

def isMyAnnotation(annotation):
    parts = [".ogg", ".png", "smile_", "angry_", "blink_"]
    s = str(annotation).lower()
    for part in parts:
        if ( part in s):
            return True
    return False

def cleanSpace():
    import cupy as xp
    gc.collect()
    xp.get_default_memory_pool().free_all_blocks()

def isSession(s):
    # Regular expression pattern to match the format ddmmyyyy_hhmm
    pattern = r'\d{8}_\d{4}'

    # Using fullmatch to ensure the entire string conforms to the pattern
    return bool(re.fullmatch(pattern, s))

def GetSessions(path):
    return [ s for s in os.listdir(path) if isSession(s) ]

def getPathsFromSessionFolder(fullFolder, edf_type="sd"):
    files = [x for x in os.listdir(fullFolder) if x.endswith(".edf") and edf_type in x.lower()]
    fileA = ""
    fileB = ""
    if ( "_a_" in files[0].lower()):
        fileA = files[0]
        fileB = files[1]
    else:
        fileA = files[1]
        fileB = files[0]
    
    p = os.path.join(fullFolder, "corrections.txt")
    with open(p, "r") as f:
        corrections = json.load(f)
    
    return [(toFilePath(fullFolder, fileA), corrections["A"]),(toFilePath(fullFolder, fileB), corrections["B"])]

def reshape_num_channels(mat, num_channels=16):
    return np.reshape(mat,(num_channels, len(mat)//num_channels) )

def particiapnt_to_char(p):
    if p == A:
        return "A"
    return "B"

def remove_rows_from_matrix(mat, rows_to_delete):
    rows_to_delete = np.array(rows_to_delete)
    if (rows_to_delete.shape[0] == 0):
        return mat
    return np.delete(mat, rows_to_delete, axis=0)

def replace_rows_with_zeros(mat, rows_to_delete):
    for row in rows_to_delete:
        mat[row] = np.zeros(mat.shape[1])
    return mat
