"""
    Act_classification-with-CNN/eval.py
    
    | Evaluate the trained model.     
    
    * Defines parameters for model restoring and model testing.
    * Loads test dataset and preprocess and restores trained model
    * After comparing the prediction of the trained model and the desired output, it measures the accuracy of exactly the same for all classes. 
    * For each act, it calculates recall, precision. f1 score and stores these at './checkpoint_dir/../act_accuracy.csv' 
    
     .. figure:: act_accuracy.jpg
       :align: center
       
       This is brief example of act_accuracy.csv 
       
    * To check prediction result, this file makes human readable format file at './checkpoint_dir/../prediction.csv'
    
     .. figure:: prediction.jpg
       :align: center
       
       This is brief example of prediction.csv 
       
       | If you want to see the wrong sample, you can set the answer column to the filter column and see a sample of zero labels.
"""

import tensorflow as tf  
import numpy as np
import os
import sys
import time
import datetime
from model import Model
from tensorflow.contrib import learn
import csv
from DataHelper import *
from utils import *
import csv
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Parameters
# ==================================================
# Data Parameters
tf.flags.DEFINE_string("test_data_file", "./csvs/test.csv", "Data source for the test data.")
tf.flags.DEFINE_string("act_data_file", "./csvs/act.csv", "Data source for the act data.")

# Eval Parameters

tf.flags.DEFINE_string("checkpoint_dir", "./runs/1508157599/checkpoints", "Checkpoint directory from training run")

tf.flags.DEFINE_integer("history_size1",2,"History size of first FNN layer (default: 2)")
tf.flags.DEFINE_integer("history_size2",1,"History size of second FNN layer (default: 1)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))
print("")

# load test data
test_data = load_data(FLAGS.test_data_file)

# map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir,"..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

x_raw = test_data['sentence'].values.tolist()
test_data['sentence']=list(vocab_processor.transform(test_data['sentence']))

#compute max sentence length for sentence padding
max_sentence_length = len(list(test_data['sentence'])[0])


#convert acts to one-hot vector
test_data['act'] = convert_act_to_vector(test_data['act'],FLAGS.act_data_file) 
y_test = test_data['act'].values.tolist()

print("\nEvaluating...\n")


# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        
        # Get the dropout operation from the graph bt name
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        
        # Tensors we want to evaluate
        prediction_op = graph.get_operation_by_name("output/predictions").outputs[0]
              
        test_dm = DataHelper(test_data)
        
        # Generate padding dataframe and padding to first sentence 
        padding = pd.DataFrame(columns = ['sentence'], index = range(FLAGS.history_size1))
        for i in range(FLAGS.history_size1 + FLAGS.history_size2):
            padding['sentence'][i]=np.array([0] * max_sentence_length)
        padding = padding['sentence']
        unshuffled_test_data = test_dm.get_test_contents()
        
        # Collect the predictions here
        all_predictions = []

        for x_test in unshuffled_test_data:   
            x_test = padding.append(x_test, ignore_index=True)  
            for i in range(len(x_test)-FLAGS.history_size1-1):
                x_test_batch = x_test[i:i+FLAGS.history_size1+FLAGS.history_size2+1].values.tolist()
               
                feed_dict = {
                    input_x : x_test_batch,
                    dropout_keep_prob: 1.0
                }
                prediction = sess.run([prediction_op], feed_dict) 
                prediction = np.squeeze(np.array(prediction))

                all_predictions.append(list(prediction))
                
# Print accuracy if y_test is defined
answer = []
if y_test is not None:
    correct_predictions = 0.0
    for i in range(len(all_predictions)):
        if all_predictions[i]==y_test[i]:
            correct_predictions +=1.0
            answer.append([1])
        else:
            answer.append([0])
    print("Total number of test examples: {}".format(len(y_test)))
    print("Correct prediction examples: {}".format(correct_predictions))
    print("System Accuracy: {:g}\n".format(correct_predictions/float(len(y_test))))


all_predictions = np.array(all_predictions)

# read acts list
with open('./csvs/act.csv','r') as f:
    reader = pd.read_csv(f)
    original_acts = reader.values.tolist()
    
# act_num : number of acts 
# data_num: number of sentences in test dataset
act_num = len(all_predictions[1])
data_num = len(all_predictions)

# label : number of true label for each act 
# prediction : number of system's prediction for each act 
label = np.sum(y_test,axis=0)
prediction = np.sum(all_predictions,axis=0)

with open(os.path.join(FLAGS.checkpoint_dir, "..",'act_accuracy.csv'), 'w') as summary:
    writer = csv.DictWriter(summary, fieldnames = ["act","number of label", "number of prediction","correct prediction","recall", "precision","f1score","remark"])
    writer.writeheader()    
     
    #correct_prediction : systems's correct prediction per one act
    #act_prediction : system's all prediction per one act
    for act_index in range(act_num):
        correct_prediction = 0
        act_prediction = 0
        remark = []
        for i in range(data_num):
            one_prediction = all_predictions[i][act_index]
            one_label = y_test[i][act_index]
            if one_prediction == 1 :
                act_prediction +=1
                if one_label ==1:
                    correct_prediction +=1

                    
        # recall
        if label[act_index]!=0 and correct_prediction !=0:
            recall = correct_prediction/label[act_index]

        else:
            if label[act_index] == 0 :
                print("{} : This act does not exist in label".format(original_acts[act_index]))
                remark.append(["This act does not exist in label"])
            if correct_prediction == 0 :
                print("{} : This act never matched the correct answer.".format(original_acts[act_index]))
                remark.append(["This act never matched the correct answer"])
            recall = 0   
                    
                    
        # precision
        if prediction[act_index]!= 0 and correct_prediction != 0:
            precision = correct_prediction/prediction[act_index]

        elif prediction[act_index] == 0:
            print("{} : This act is never detect by system".format(original_acts[act_index]))
            remark.append(["This act is never detect by system"])
            precision = 0
  
            
        # f1 score    
        if recall != 0 and precision != 0:
            f1score= 2 * recall * precision / (recall + precision)
 
        else:
            f1score = "NaN"
            
        if remark == []:
            remark = ""
            
        print("{0} correct prediction: {1},recall: {2}, precision : {3}, f1 score : {4}".format(original_acts[act_index],
                                                                                                correct_prediction,
                                                                                                recall,
                                                                                                precision,
                                                                                                f1score))
    
        writer.writerow({'act': original_acts[act_index][0],
                         'number of label':label[act_index],
                         'number of prediction':act_prediction,
                         'correct prediction':correct_prediction,
                         'recall':recall,
                         'precision':precision,
                         'f1score':f1score,
                        'remark':remark})  
     

# Save the predictions as human readable format
string_prediction = convert_vector_to_act(all_predictions,FLAGS.act_data_file)
string_label = convert_vector_to_act(y_test,FLAGS.act_data_file)
predictions_human_readable = np.column_stack((x_raw, string_prediction, string_label, answer))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))

with open(out_path, 'w') as f:
    csv.writer(f).writerows(np.column_stack((['sentence'],['prediction'],['label'],['answer'])))
    csv.writer(f).writerows(predictions_human_readable)
    
print("\nEvaluation finished!")
