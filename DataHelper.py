"""
    Act_classification-with-CNN/DataHelper.py


    This file manages the data to feed to model.
 
 
    To ease data management, this code uses the Pandas python package. 
    Pandas is a Python package providing fast, flexible, and expressive data structures designed to make working with relational or labeled data both easy and intuitive.
    The two primary data structures of pandas, Series (1-dimensional) and DataFrame (2-dimensional), handle the vast majority of typical use cases in many areas of engineering. 
    | In this DataHelper class, each data as pandas.DataFrame is given as a parameter.
       
       
    
"""

import numpy as np
import pandas as pd

class DataHelper:
    """
    This class receives whole dataset in dataframe format and returns each call. 
    Each call consists of sentences and corresponding dialog acts.
    When we train a model, sentences and dialog acts for each call are fed into model as input and desired output.
    
    :param dataframe df: Whole dataset in dataframe format
  
    """
    
    def __init__(self,df):
        self.df = df
        
    def get_num_calls(self):
        """
        This function counts number of calls in each whole dataset
        
        :return int: Total number of calls in whole data 
        """
        return self.df['call_idx'].iloc[self.df['call_idx'].idxmax()] + 1
    
    def get_shuffled_call_idx(self, call_indexs):
        """
        This function shuffles whole call indexs 
        
        :param int call_indexs: Whole call indexs in order
        :return np.array: Array of shuffled call indexs
        """
        arr=np.asarray([i for i in call_indexs])
        np.random.shuffle(arr)
        return arr

    def get_call_df(self, call_idx):
        """
        This function returns a dataframe of each call corresponding to each call index
        
        :param int call_idx: A call index
        :return pandas.dataframe: Dataframe corresponding to call index
        """
        return self.df[self.df['call_idx'] == call_idx] 

    def get_contents(self, shuffle=False) :
        """     
        This function manages the whole train or dev dataset and returns sentences and dialog acts for one call.
        
        
        Usually, when we train the model, we shuffle the order of the data. 
        In this case, the shuffle flag is set to True, it returns data in mixed order
        But when we validate the model, we do not shuffle the order of the data.
        In this case, the shuffle flag is set to False, and it returns data in order 
        
        
        if shuffle is True: Return shuffled data (for train data)
        elif shuffle is False: Return unshuffled data (for dev data) 
        
        Args:
            shuffle (bool): Whether to shuffle the order of the calls 

        Yields:
            pandas.series: Series of sentences in each call 
            numpy.ndarray: An array of dialog acts in each call
        """
        call_indexs = range(self.get_num_calls())

        if shuffle:
            call_indexs = self.get_shuffled_call_idx(call_indexs)
        
        for call in call_indexs:    
            base = pd.DataFrame()
            base = self.get_call_df(call)  
            
            sentence_arr =base['sentence']
            act_arr = np.asarray(base['act'])
            
            yield sentence_arr, act_arr
           
    def get_test_contents(self):
        """     
        This function manages the whole test dataset and returns sentences for one call.
        
        
        When we test the model, we just feed input sentence. 
        After we get the predictions then we compare it to desired output. 
        Therefore in this function, it just returns sentences of each call
        
        Yields:
            pandas.series: Series of sentences in each call 
        """
        call_indexs = range(self.get_num_calls())
        
        for call in call_indexs:
            base = pd.DataFrame()
            base = self.get_call_df(call) 

            sentence_arr = base['sentence']

            yield sentence_arr
