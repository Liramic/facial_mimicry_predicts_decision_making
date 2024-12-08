import pandas as pd
import os
import numpy as np


class UserChoice:
    def __init__(self, story1, story2, choiceA, choiceB, rtA, rtB, isOther) -> None:
        self.story1 = story1
        self.story2 = story2
        self.choiceA = choiceA
        self.choiceB = choiceB
        self.rtA = rtA
        self.rtB = rtB
        self.isOther = isOther
    
    def __str__(self):
        return f"story1 : {self.story1}, story2:{self.story2}, choiceA:{self.choiceA}, choiceB:{self.choiceB}, rtA:{self.rtA}, rtB:{self.rtB}, isOther:{self.isOther}"
    
    def to_list(self):
        return [self.story1, self.story2, self.choiceA, self.choiceB, self.rtA, self.rtB, self.isOther]
        
def GenerateUserChoice(df, i):
    s1 = str(df["StoryOrder1"][i])
    s2 = str(df["StoryOrder2"][i])
    choiceA = int(df["UserANumberChoice"][i])
    choiceB = int(df["UserBNumberChoice"][i])
    rtA = str(df["UserA_choice.rt"][i])
    rtB = str(df["UserB_choice.rt"][i])
    isOther = "other" in str(df["AudioInstruction"][i]).lower()
    return UserChoice(s1, s2, choiceA, choiceB, rtA, rtB,isOther)

def read_experiment_csv(data_path, session):
    userChoiceDict = dict()
    session_folder = os.path.join(data_path, session)
    fname = [x for x in os.listdir(session_folder) if x.endswith(".csv")][0]
    csvPath = os.path.join(session_folder, fname)
    df = pd.read_csv(csvPath)
    storyIndexes = np.where(df["trialId"].notnull())[0]
    for i in storyIndexes:
        uc = GenerateUserChoice(df, i)
        key = f"{uc.story1.split('.')[1].lower()}_{int(df['trialId'][i])}"
        userChoiceDict[key] = uc
    return userChoiceDict



#['story' , 'whoseReading', 'isChoiceA', 'isChoiceB', 'rtChoiceA', 'rtChoiceB', 'isOther', 'ChoseTogether']

def convert_row(df, i, whose_reading):
    s1 = str(df["StoryOrder1"][i])
    s2 = str(df["StoryOrder2"][i])
    choiceA = int(df["UserANumberChoice"][i])
    choiceB = int(df["UserBNumberChoice"][i])
    rtA = str(df["UserA_choice.rt"][i])
    rtB = str(df["UserB_choice.rt"][i])
    isOther = "other" in str(df["AudioInstruction"][i]).lower()
    
    s1_arr = [s1, whose_reading, int(choiceA == 1), int(choiceB == 1), rtA, rtB, int(isOther), int(choiceA == choiceB)]
    s2_arr = [s2, whose_reading, int(choiceA == 2), int(choiceB == 2), rtA, rtB, int(isOther), int(choiceA == choiceB)]

    return s1_arr, s2_arr


def read_experiment_csv_batch2(data_path, session):
    uc_list = []
    session_folder = os.path.join(data_path, session)
    fname = [x for x in os.listdir(session_folder) if x.endswith(".csv")][0]
    csvPath = os.path.join(session_folder, fname)
    df = pd.read_csv(csvPath)

    whose_reading_arr = [1,2,1,2]
    for i in range(len(df)):
        if pd.isna(df["Trigger0"][i]):
            continue
        if ( "rs" in str.lower(df["Trigger0"][i])):
            whose_reading = whose_reading_arr.pop(0)
        elif ("as" in str.lower(df["Trigger0"][i])):
            whose_reading = 0
        else:
            continue #smile\frown\blink
        s1_arr, s2_arr = convert_row(df, i, whose_reading)
        uc_list.append(s1_arr)
        uc_list.append(s2_arr)
    return uc_list
