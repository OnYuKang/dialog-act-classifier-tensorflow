"""
    Act_classification-with-CNN/data_generator.py

    | This file makes DSTC2(Dialog State Tracking Challenge 2) json file to csv file.
    | DSTC2 dataset consist of call, each one in its own directory. 
    | Each call has 'log.json' file and 'label.json' file.
    * 'log.json' file, a Log Object which annotates the system's output 
    * 'label.json' file, a Label Object which annotates the user's true utterance and action.
    | The formats of these objects are fully specified in DSTC2&3 handbook 15p ~ 18p (Appendix A JSON Data Format)
    | Dialog State Tracking Challenge 2 & 3 handbook (http://camdial.org/~mh521/dstc/downloads/handbook.pdf)
 
"""

__version__ = '1.0'

import csv
import re
import os
import json
import numpy as np
from utils import *
 

class dataset_walker(object):
    """
    Access the file list(*.flist) in the config directory to get the file of train, dev, test dataset.
    | Pass the log file name and label file name to the call class

    :param list dataset: A list of dataset name.
    :param bool label: Boolean value for the existence of label file. The default value is False.
    :param str dataroot: A name of directory where data exists. The default value is None.
    
    :returns call: Dictinary type dataset contained each call information   
    """
    def __init__(self,dataset,label=False, dataroot=None):
        if "[" in dataset:
            self.datasets = json.loads(dataset)
        elif type(dataset) == type([]) :
            self.datasets = dataset
        else:
            self.datasets = [dataset]
            self.dataset = dataset
        try:
            self.install_root = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
        except:
            self.install_root = os.path.abspath(os.path.dirname(os.path.abspath("__file__")))

        self.dataset_session_lists = [os.path.join(self.install_root, 'config', dataset + '.flist') for dataset in self.datasets]
        self.label = label
        
        if dataroot == None:
            install_parent = os.path.dirname(self.install_root)
            self.dataroot = os.path.join(install_parent, 'dstc2-simpleacts')  
        else:
            self.dataroot = os.path.join(os.path.abspath(dataroot))

        # load dataset (list of calls)
        self.session_list = []
        for dataset_session_list in self.dataset_session_lists:
            f = open(dataset_session_list)
            for line in f:
                line = line.strip()

                if line in self.session_list:
                    raise RuntimeError("Call appears twice: %s" % line) 
                    
                self.session_list.append(line)
                
            f.close()   

    def __iter__(self):
        for session_id in self.session_list:
            session_id_list = session_id.split('/')
            session_dirname = os.path.join(self.dataroot, *session_id_list)
            log_filename = os.path.join(session_dirname, 'log.json')
            if self.label:
                label_filename = os.path.join(session_dirname, 'label.json')
                if not os.path.exists(label_filename):
                    raise RuntimeError("Cant score: cant open label file %s" % label_filename)
            else:
                label_filename = None
            call = Call(log_filename, label_filename)
            call.dirname = session_dirname

            yield call
        
        
    def __len__(self):
        return len(self.session_list)
            


class Call(object):
    """
    The DSTC2 dataset contains the system's call information in log.json and the user's call information in the label.json file. Therefore this class receives log file name and label file name and then accesses turns (a list of log-turn objects / a list of label-turn objects) in each file to combine information about each call.
    
    
    A list of log-turns, which gives the output of the system
    A list of label-turn object runs in parallel to the log-turn object and provides annotations of user's output on the turn level.
    
    
    The detailed format for Log object and Label object is described in DSTC2&3 handbook 15p ~ 18p (Appendix A JSON Data Format)

    :param str log_filename: A name of file containing log file
    :param str label_filename: A name of file containing label file

    :returns dict: Dictinary type dataset contained each call information  
    """
    def __init__(self, log_filename, label_filename):
        self.log_filename = log_filename
        self.label_filename = label_filename
        f = open(log_filename)
        self.log = json.load(f)
        f.close()
        if label_filename != None:
            f = open(label_filename)
            self.label = json.load(f)
            f.close()
        else:
            self.label = None
    
    def __iter__(self):
        if self.label_filename != None:
            for (log, label) in zip(self.log['turns'], self.label['turns']):
                yield {'log': log, 'label': label}
        else:
            for log in self.log['turns']:
                yield {'log': log, 'label': {}}
                
    def __len__(self):
        return len(self.log['turns'])
    

def make_csv(dataset, file_name, file_dir, postfix='.csv'):
    """
    It receives a dataset that combines log file and label file through dataset_walker.
    | This function extracts the necessary information from the dataset and transforms that into csv file format. The necessary information is call_index, trun_index, sentence, and annotated dialog-act.
    
    
    Note that the key names of the dictionary in which the user and system call information are stored are different. The format of the detailed dictionary is specified in DSTC2&3 handbook 15p ~ 18p (Appendix A JSON Data Format)
    
    
    In the current code, save it in the existing "csvs" folder. Otherwise, create the "csvs" folder and save the data in that folder.

    
    :param dict dataset: Dictionary type dataset containing information about call
    :param str file_name: A name of file to save
    :param str file_dir: A name of directory in which to save the file
    :param str postfix: An extension name of the file to be saved. The default value is ".csv".
    
    .. figure:: train_csv_example.jpg
       :align: center
       
       This shows the train.csv file created with this function as Excel.The first line indicates each column's name and the others are member of each column.
    
    """
    full_name = get_abs_path(file_dir, file_name + postfix)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    with open(full_name, 'w', encoding='utf8') as f:
        wr = csv.writer(f)
        header = [
            'call_idx',
            'turn_idx',
            'sentence',
            'act'
        ]
        wr.writerow(header)
        
        call_idx=0
        for call in dataset:
            turn_idx=0
            for turn in call:
                # system's turn            
                # system sentence
                sys_output = str(turn['log']['output']['transcript'])
                
                # system act list
                sys_act=[]
                for j in turn ['log']['output']['dialog-acts']:
                    sys_act.append(j['act'])
                sys_act=list(set(sys_act))
                
                #write one line to file
                wr.writerow([
                    call_idx,
                    turn_idx,
                    preprocess(sys_output),
                    sys_act
                ])
                turn_idx = turn_idx+1
                
                # user's turn  
                # user sentence
                utterance = str(turn['label']['transcription'])
                
                # user act list
                user_act=[]
                for i in turn['label']['semantics']['json'] :
                    user_act.append(i['act'])
                user_act=list(set(user_act))
                if user_act==[]:
                    user_act=['null']
                                 
                wr.writerow([
                    call_idx,
                    turn_idx,
                    preprocess(utterance),
                    user_act
                ])
                turn_idx=turn_idx+1       
            call_idx=call_idx+1
            
        print ("Finished writing %s" % file_name)             
           
def make_act_csv(file_name,file_dir, postfix='.csv'):
    """
    DSTC2 does not provide a file for the whole dialog act. 
    | Therefore this function creates a csv file for the DSTC2 whole dialog acts.
    | Note thatif you use other data (not DSTC2), you can use it directly in the model if you make acts to csv file format.
    
    :param str file_name: A name of file to save
    :param str file_dir: A name of directory in which to save the file
    :param str postfix: The extension name of the file to be saved. The default value is ".csv".
    
    .. figure:: act_csv_example.jpg
       :align: center
       
       This is simplified act.csv as Excel. The first line indicates column's name and the others are member of that column.
    """
    full_name = get_abs_path(file_dir, file_name + postfix)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    
    with open(full_name, 'w', encoding='utf8') as f:
    
        wr = csv.writer(f)
        wr.writerow(['act'])

        acts=['ack','affirm','bye','hello','negate','null', 
              'reqalts','restart','thankyou','confirm','deny','inform',
              'request','reqmore','repeat',
              'confirm-domain','expl-conf','impl-conf','welcomemsg','canthelp',
              'canthelp.exception','offer','select']


        for act in acts:
            wr.writerow([act])
        f.close()
        print ("Finished writing %s" % file_name)
    
if __name__ == '__main__':
    
    # act dataset : act
    make_act_csv("act",file_dir="csvs",postfix=".csv")
    
    # train dataset : train
    dataset = dataset_walker(["train"], True, "dstc2-simpleacts")
    make_csv(dataset, "train", file_dir="csvs", postfix=".csv")
    
    # dev dataset : train
    dataset = dataset_walker(["dev"], True, "dstc2-simpleacts")
    make_csv(dataset, "dev", file_dir="csvs", postfix=".csv")

    # test dataset : test
    dataset = dataset_walker(["test"], True, "dstc2-simpleacts")
    make_csv(dataset, "test", file_dir="csvs", postfix=".csv")
