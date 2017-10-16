"""
    Act_classification-with-CNN/train.py

    This file trains the Tensorflow model. 
   
    In this file, 
    
    * Defines parameters for model initialization and training.
    * Loads dataset for training/validation and preprocess the dataset.
    * Initializes the Tensorflow model.
    * If there is a pretrained word vector(GloVe), it is loaded and assigned to the word vector of the model.
    * Defines some operations to be displayed on the Tensorboard summary.
    * Store the model and vocaburary for each tf.FLAGS.checkpoint_every step.

"""
from DataHelper import *
from utils import *
import os, sys
import tensorflow as tf
import time
import datetime
from model import Model
from tensorflow.contrib import learn
import csv

# Specify GPU number
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

# Data loading params
tf.flags.DEFINE_string("train_data_file", "./csvs/train.csv", "Data source for the train data.")
tf.flags.DEFINE_string("dev_data_file", "./csvs/dev.csv", "Data source for the dev data.")
tf.flags.DEFINE_string("act_data_file", "./csvs/act.csv", "Data source for the act data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embed_size", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_integer("filter_size", 3, "Filter sizes (default: 3)")
tf.flags.DEFINE_integer("num_filters", 500, "Number of filters per filter size (default: 500)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer("history_size1",2,"History size of first FNN layer (default: 2)")
tf.flags.DEFINE_integer("history_size2",1,"History size of second FNN layer (default: 1)")


"""
    Misc Parameters
    * allow_soft_placement : TensorFlow automatically choose an existing and supported device to run the operations in case the specified one doesn't exist, 
    * log_device_placement : Finds the device to which the operation or tensor is assigned

"""

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Training parameters
tf.flags.DEFINE_integer("num_epochs", 150, "Number of training epochs (default: 150)")
tf.flags.DEFINE_integer("evaluate_every", 23355, "Evaluate model on dev set after this many steps (default: 23355, this step number is 1 epoch)")
tf.flags.DEFINE_integer("checkpoint_every", 233550, "Save model after this many steps (default: 233550, this step number is 10 epoch)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("num_classes",23,"Number of output act classes (default:23)")
tf.flags.DEFINE_integer("vocab_size",400000,"Vocabulary size (default:400000)")
tf.flags.DEFINE_string("glove","./glove/glove.txt","Using pretrained word embedding GloVe (default:True)")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

# parameter print
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value))
print("")



# Data Preparation
# ==================================================

# Load data
print("Loading data...")
whole_train_data = load_data(FLAGS.train_data_file)
whole_dev_data = load_data(FLAGS.dev_data_file)

# To remove NAN value in data
whole_train_data['sentence'].fillna("", inplace = True)
whole_dev_data['sentence'].fillna("", inplace = True)

# Build vocabulary and preprocess sentences
max_sentence_length = max(
    max([len(x.split(" ")) for x in whole_train_data['sentence']]),
    max([len(y.split(" ")) for y in whole_dev_data['sentence']])
)

# Preprocessing input sentences
vocab_processor = learn.preprocessing.VocabularyProcessor(max_sentence_length)
x_train_sen = list(vocab_processor.fit_transform(whole_train_data['sentence']))
x_dev_sen = list(vocab_processor.fit_transform(whole_dev_data['sentence']))

whole_train_data['sentence'] = x_train_sen
whole_dev_data['sentence'] = x_dev_sen

# Convert acts to one-hot vector
whole_train_data['act'] = convert_act_to_vector(whole_train_data['act'],FLAGS.act_data_file)
whole_dev_data['act'] = convert_act_to_vector(whole_dev_data['act'],FLAGS.act_data_file)

print("max_sentence_length: {:d}".format(max_sentence_length))
print("Vocabulary size : {:d}".format(len(vocab_processor.vocabulary_)))

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    
    with sess.as_default():
        model = Model(
            max_sentence_len=max_sentence_length,
            num_classes=FLAGS.num_classes,
            vocab_size=len(vocab_processor.vocabulary_),
            embed_size=FLAGS.embed_size,
            filter_size=FLAGS.filter_size,
            num_filters=FLAGS.num_filters,
            history_size1=FLAGS.history_size1,
            history_size2=FLAGS.history_size2
        )
        
        
        #define Training procedure
        global_step = tf.Variable(0, name="global_step",trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name='Adam')
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars,global_step=global_step)
        
        #define training accuracy measurements (accumulation value) 
        train_recall, train_recall_op =tf.metrics.recall(labels = model.input_y,
                                              predictions = model.prediction,
                                              name = "recall")
        train_precision, train_precision_op = tf.metrics.precision(labels = model.input_y,
                                                     predictions = model.prediction,
                                                     name = 'precision')
        train_f_score = tf.scalar_mul(2.0,tf.div(tf.multiply(train_precision,train_recall),
                                                    tf.add(train_precision,train_recall)))
        
        #define validation accuracy measurements (accumulation value)
        dev_recall, dev_recall_op =tf.metrics.recall(labels = model.input_y,
                                              predictions = model.prediction,
                                              name = "recall")
        dev_precision, dev_precision_op = tf.metrics.precision(labels = model.input_y,
                                                     predictions = model.prediction,
                                                     name = 'precision')
        dev_f_score = tf.scalar_mul(2.0,tf.div(tf.multiply(dev_precision,dev_recall),
                                                    tf.add(dev_precision,dev_recall)))
            
            
        #output directory for models and summries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))
        
        
        #summries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", model.loss)
        acc_summary = tf.summary.scalar("accuracy", model.accuracy)
        
        train_precision_summary = tf.summary.scalar("precision", train_precision)
        train_recall_summary = tf.summary.scalar("recall", train_recall)
        train_f_score_summary = tf.summary.scalar("f_score",train_f_score)
        
        dev_precision_summary = tf.summary.scalar("precision", dev_precision)
        dev_recall_summary = tf.summary.scalar("recall", dev_recall)
        dev_f_score_summary = tf.summary.scalar("f_score",dev_f_score)
        
        #Train summaries
        train_summary_op = tf.summary.merge([loss_summary,acc_summary, train_precision_summary, train_recall_summary, train_f_score_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        #Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary,dev_precision_summary, dev_recall_summary, dev_f_score_summary,])
        dev_summary_dir = os.path.join(out_dir, "summaries","dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        #Checkpoint directory, 
        checkpoint_dir = os.path.abspath(os.path.join(out_dir,"checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))
        
        
        #initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        
        # load pre-trained word vector
        if FLAGS.glove:
            print("Loading GloVe file... {}".format(FLAGS.glove))
            
            #initial matrix with random uniform	
            initW = np.random.uniform(-0.25,0.25,(len(vocab_processor.vocabulary_),FLAGS.embed_size))

            #load any vectors from the glove
            f = open(FLAGS.glove,"r")
            vocab =[]
            embed = []
            for line in f.readlines():
                row = line.strip().split(' ')
                idx = vocab_processor.vocabulary_.get(row[0])
                if idx != 0 :
                    initW[idx] = row[1:]
            f.close
            
            print("GloVe file has been loaded!\n")
            sess.run(model.W.assign(initW))
                   
        def train_step(x_text,y_text):
            """
            A single training step

            """
    
            feed_dict = {
                model.input_x : x_text,
                model.input_y: y_text,
                model.dropout_keep_prob: FLAGS.dropout_keep_prob
            }

            _,step, summaries, loss, accuracy, precision, recall = sess.run(
                [train_op, global_step, train_summary_op, model.loss, model.accuracy, train_precision_op, train_recall_op],
                feed_dict)

            if step % 1000 == 0:
                time_str = datetime.datetime.now().isoformat()
                print("train> {}: step {}, loss {:g}, acc {:g}, pre {:g}, recall {:g}".format(time_str, step, loss, accuracy, precision, recall))

            train_summary_writer.add_summary(summaries, step)

        
        def dev_step(validation_step, x_text, y_text, writer=None):
            """
            A single validation step

            """

            feed_dict = {
                model.input_x: x_text,
                model.input_y: y_text,
                model.dropout_keep_prob: 1.0
            }

            summaries, loss, accuracy, precision, recall = sess.run(
                [dev_summary_op, model.loss, model.accuracy, dev_precision_op, dev_recall_op],
                feed_dict)

            if validation_step % 1000 == 0:
                time_str = datetime.datetime.now().isoformat()
                print("dev> {}: step {}, loss {:g}, acc {:g}, pre {:g}, recall {:g}".format(time_str, validation_step, loss, accuracy, precision, recall))          

            if writer: 
                writer.add_summary(summaries, validation_step)

        train_dm = DataHelper(whole_train_data)
        dev_dm = DataHelper(whole_dev_data)
        
        #zero padding for the first sentence that does not have history sentences
        padding = pd.DataFrame(columns=['sentence'],index=range(FLAGS.history_size1))
        for i in range(FLAGS.history_size1 + FLAGS.history_size2):
            padding['sentence'][i]=np.array([0]*max_sentence_length)
        padding = padding['sentence']
        
        validation_step = 0
        #generate batches
        for epoch_i in range(FLAGS.num_epochs):
            train_data = train_dm.get_contents(shuffle=True)
            for x_batch, y_batch in train_data :        
                x_batch = padding.append(x_batch,ignore_index=True)
                
                #train step
                for i in range(len(x_batch)-FLAGS.history_size1-1):
                   
                    train_step(x_batch [i : i + FLAGS.history_size1 + FLAGS.history_size2 + 1].values.tolist(),
                               np.reshape(y_batch[i],(1,FLAGS.num_classes)))

                    current_step = tf.train.global_step(sess, global_step)

                    if current_step % FLAGS.evaluate_every ==0:
                        print("\nValidation:")

                        dev_data = dev_dm.get_contents()
                        for x_dev, y_dev in dev_data :
                            x_dev = padding.append(x_dev,ignore_index=True)

                            #dev_step
                            for i in range(len(x_dev)-FLAGS.history_size1-1):
                                validation_step += 1
                                dev_step(validation_step, x_dev[i : i + FLAGS.history_size1 + FLAGS.history_size2 + 1].values.tolist(),
                                         np.reshape(y_dev[i],(1,FLAGS.num_classes)),
                                        writer=dev_summary_writer)
                        print(" ")

                    if current_step %FLAGS.checkpoint_every ==0 :
                        path = saver.save(sess, checkpoint_prefix, global_step = current_step)
                        print("Saved model checkpoiont to {}\n".format(path))


