"""
    Act_classification-with-CNN/utils.py

    This file contains some useful functions for other files.
    
"""

import os
import re
import json
import inspect
import numpy as np
import pandas as pd
from collections import defaultdict
import ast

def get_abs_path(file_dir, file_name):
    """
    Returns the absolute path of the file name
    
    :param str file_dir: Directory containing files
    :param str file_name: The name of the file.
    
    :return str abs_path: The absolute path of the file
    """
    abs_path = os.path.join(os.path.abspath(file_dir), file_name)
    return abs_path

def preprocess(string):
    """
    When creating a csv file from json files in a data_generator.py file, this function is used to preprocess the string.
    
    | It makes all the words in the sentence lower case and replace sentence symbols to space. 
    Then it replaces miss spelled words into correct words due to speech recognition errors and erases the proper nouns that are not found in word embedding among the words appearing in the dialogue. After that, it returns a copy of the string in which all chars have been stripped from the beginning and the end of the string.
    
    :param str string: Raw string
    :return str string: Preprocessed string
    """
    exception_words = {
        r"\btaquita\b": "taquitos",
        r"\bexpen\b": "expensive",
        r"\bmeditteranian\b": "mediterranean",
        r"\bdeosnt\b": "does not",
        r"\bbaskey\b": "basket",
        r"\bdontcare\b": "dont care",
        r"\bexpl\b": "explicit",
        r"\baddr\b": "address",
        r"\bconf\b": "confirm",
        r"\bnandos\b": "nando s", # careful
        r"\bopean\b": "european",
        r"\bexpe\b": "expensive",
        r"\bseouls\b": "seoul s", # careful
        r"\bunitelligible\b": "unintelligible",
        r"\bnosie\b": "noise",
        r"\bgoodb\b": "goodbye",
        r"\bscandin\b": "scandinavian",
        
        r"\bkymmoy\b": "",
        r"\bbennys\b": "",
        r"\beraina\b": "",
        r"\balimentum\b": "",
        r"\bpanahar\b": "",
        r"\bfitzbillies\b": "",
        r"\bdarrys\b": "",
        r"\bcocum\b": "",
        r"\bzizzi\b": "",
        r"\bpanasian\b": "",
        r"\bpipasha\b": "",
        r"\bLensfield\b": "",

    }
    
    string = string.strip().lower()
    string = re.sub(r"\'", " ", string)
    string = re.sub(r"[^a-z?]", " ", string)
    string = re.sub(r"\?"," ?",string)
    string = re.sub(r"\s+", " ", string)

    for fr, to in exception_words.items():
        string = re.sub(fr, to, string)
    string = re.sub(r"\s+", " ", string)
    
    return string.strip()


def load_data(file_name):
    """
    This function loads csv file as pandas.dataframe in train.py and eval.py.
    
    :param str file_name: File name to load
    :return dataframe df: Whole dataset as dataframe type
    .. code-block:: python
       :linenos:
       
       df = load_data('./csvs/train.csv')
    
    .. figure:: load_data_example.jpg
       :scale: 30%
       :align: center
       
       This example is train.csv file loaded into pandas.DataFrame.
       
    """
    file_path = os.path.abspath(file_name)
    if not file_name.endswith(".csv"):
        raise RuntimeError("file %s is of invalid file format" % file_path)
    
    df = pd.read_csv(file_path, index_col=False)
    return df

def convert_act_to_vector(df,act_data_file):
    """
    This function converts pandas act dataframe to one-hot vector in train.py and eval.py
    
    :param pandas.dataframe: Act dataframe 
    :param str act_data_file: Act csv file 
    :return list labels: Converted one-hot vectors from act as string type 
    
    .. code-block:: python
       :linenos:
 
        data = pd.read_csv('./csvs/train.csv')
        data['act'] = convert_act_to_vector(data['act'],'./csvs/act.csv')
        
    .. figure:: convert_act_to_vector_example.jpg
       :scale: 30%
       :align: center
      
       This example is some acts label to one hot vector label.
    """
    #act list from dataset
    actlist=df.values.tolist()
    
    #load act csv file and make it list type
    act_file_path = os.path.abspath(act_data_file)
    act_df=pd.read_csv(act_file_path)
    standardact=act_df['act'].values.tolist()
    
    #make one-hot label
    labels=[]
    for acts in actlist:
        label=np.zeros(len(standardact),dtype=np.float32)
        acts = ast.literal_eval(acts)

        for i in range(len(acts)):
            index=standardact.index(acts[i])
            label[index]=1.0
        label=list(label)
        labels.append(label)
        
    return labels

def convert_vector_to_act(act_array,act_data_file):
    """
    This function convert one-hot vector to act as string type
    
    :param numpy.array: one-hot vectors as numpy array type
    :param str act_data_file: Act csv file
    :returns list whole_act_list: converted acts list as string type from one-hot vectors
    """
    
    #load act csv file and make it list type
    act_file_path = os.path.abspath(act_data_file)
    act_df=pd.read_csv(act_file_path)
    standard_act=act_df['act'].values.tolist()
    
    whole_act_list=[]
    # convert one-hot vector to act as string type
    for i in range(len(act_array)):
        act_list=[]
        for j in range(len(act_array[i])):
                       if act_array[i][j]==1:
                          act_list.append(standard_act[j])
        whole_act_list.append(act_list)
                       
    return whole_act_list

