# Act-Classification-with-CNN

Requirements
============
* Python 3
* Tensorflow > 0.12
* Numpy
* Pandas

Training
=========
Print parameters:
<pre><code>
  python3 train.py --help
</code></pre>
<pre><code>
  optional arguments:

  -h, --help            show this help message and exit
  --train_data_file TRAIN_DATA_FILE
 	                    Data source for the train data.
  --dev_data_file DEV_DATA_FILE
                        Data source for the dev data.
  --act_data_file ACT_DATA_FILE
                        Data source for the act data.
  --embed_size EMBED_SIZE
                        Dimensionality of character embedding (default: 300)
   --filter_size FILTER_SIZE
                        Filter sizes (default: 3)
   --num_filters NUM_FILTERS
                        Number of filters per filter size (default: 500)
   --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability (default: 0.5)
   --history_size1 HISTORY_SIZE1
                        History size of first FNN layer (default: 2)
   --history_size2 HISTORY_SIZE2
                        History size of second FNN layer (default: 1)
   --allow_soft_placement [ALLOW_SOFT_PLACEMENT]
                        Allow device soft device placement
   --noallow_soft_placement
   --log_device_placement [LOG_DEVICE_PLACEMENT]
                        Log placement of ops on devices
   --nolog_device_placement
   --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 150)
   --evaluate_every EVALUATE_EVERY
                        Evaluate model on dev set after this many steps (default: 233550)
   --checkpoint_every CHECKPOINT_EVERY
                        Save model after this many steps (default: 233550)
   --num_checkpoints NUM_CHECKPOINTS
                        Number of checkpoints to store (default: 10)
   --num_classes NUM_CLASSES
                        Number of output act classes (default:23)
   --max_sentence_len MAX_SENTENCE_LEN
                        Max sentence length (default:30)
   --vocab_size VOCAB_SIZE
                        Vocabulary size (default:400000)
   --glove GLOVE        Using glove (default:True)
</code></pre>

Train:
<pre><code>
  python3 train.py
</code></pre>

Evaluating
==========
<pre><code>
  python3 eval.py --checkpoint_dir="./runs/150437919/checkpoints/" 
</code></pre>

References
==========
[Sequential Short-Text Classification with Reccurent and Convolutional Neural Networks](https://arxiv.org/abs/1603.03827)
