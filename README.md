# Dialog Act Classifier in Tensorflow

Tensorflow implementation of "Sequential Short-Text Classification with Recurrent and Convolutional Neural Networks"

## Requirements

* Python 3.x
* Tensorflow > 0.12
* Numpy
* Pandas

## Pre-execution instructions

### Datasets to download
Download following files in the program root dirctory (*.../Act-Classification-with-CNN)

* DSTC2 : [download](https://mi.eng.cam.ac.uk/~nm480/dstc2-clean-transcriptions.zip)
* GloVe : [download](https://github.com/nmrksic/counter-fitting/raw/master/word_vectors/glove.txt.zip)

### csv generation 
CSV files are generated from original DSTC2 dataset and used by the program.

#### Generation process
For each dataset [train | dev | test]

1.	Generate a integraed json file from multiple json files in DSTC2 dataset.
2.	generate a csv file using the integrated json file.

After csv files are generated, original DSTC2 dataset files and json files generated midway can be deleted.

## Basic Guideline
In the root directory:
1. Unzip downloaded DSTC2 dataset.

	   $ unzip dstc2-clean-transcriptions.zip
2. Unzip downloaded pretrained embeddings into data directory.
	   
	   $ unzip glove.txt.zip -d glove
3. Run data_generator.py.
  	 
	   $ python3 data_generator.py -a
4. (Optional) Delete unnecessary files.
    	   
	   $ rm -r dstc2-simpleacts flist _json

## Directory structure
After following the guideline, the directory structure will look like:

    Act-Classification-with-CNN
    ├── csvs
    │   ├── act.csv (generated csv file)
    │   ├── train.csv (generated csv file)
    │   ├── dev.csv (generated csv file)
    │   └── test.csv (generated csv file)
    ├── glove
    │   └── glove.txt (pretrained embedding)
    ├── data_generator.py
    ├── DataHelper.py
    ├── model.py
    ├── train.py
    ├── eval.py
    └── utils.py


## Training

Print parameters:

    $ python3 train.py --help

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

    $ python3 train.py


## Evaluating
   
    $ python3 eval.py --checkpoint_dir="./runs/150437919/checkpoints/" 
    
### Output CSV files
* act_accuracy.csv : This file contains  [ number of labels | number of predictions | correct prediction | recall | precision | f1score | remark ] for each act.
* prediction.csv : This file contains [ model's predictions | actual answers ] for each sentence. If you want to see only the wrong samples, set the answer column to the filter column and check the zero label sample.

## References
1.  Ji Young Lee et al., [Sequential Short-Text Classification with Reccurent and Convolutional Neural Networks](https://arxiv.org/abs/1603.03827), 2016
2.  DSTC2_handbook http://camdial.org/~mh521/dstc/downloads/handbook.pdf
3.	GloVe https://nlp.stanford.edu/pubs/glove.pdf
