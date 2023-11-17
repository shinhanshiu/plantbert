# Author: Shin-Han Shiu
# Date: 10/2/2023
# 
# Modified from
# - https://thepythoncode.com/article/pretraining-bert-huggingface-transformers-in-python
#
# Purpose:
#   Retrain BERT using the plant science history project corpus.
# 
# Install:
# ```bash
# conda create -n bert python=3.11.5
# conda activate bert
# 
# pip install datasets transformers sentencepiece ipywidgets pytest xformers \
#             session_info accelerate pyyaml
# ```
#
# 11/16/23: Training from checkpoint
#   https://stackoverflow.com/questions/75357653/how-to-resume-a-pytorch-training-of-a-deep-learning-model-while-training-stopped
# 11/12/23 Multi-GPU support
#   https://towardsdatascience.com/a-comprehensive-guide-of-distributed-data-parallel-ddp-2bb1d8b5edfb
#   Will not implemented it for now.
#

print("\n###################\nImporting packages\n")

import os, json, argparse, pickle, yaml
import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import BertTokenizerFast, BertConfig, BertForMaskedLM, \
                         DataCollatorForLanguageModeling, TrainingArguments, \
                         Trainer
from tokenizers import BertWordPieceTokenizer

################################################################################
# Functions
################################################################################

def get_args():
  """Get command-line arguments"""
  parser = argparse.ArgumentParser(
    description='Retrain BERT using the plant science history project corpus',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  parser.add_argument('-c', '--config_file', 
                      type=str,
                      help='Config file path',
                      default='./config.yaml')

  args = parser.parse_args()
  
  return args

def txt_to_split_dataset(corpus_file, test_size, rand_seed):
  '''Convert corpus to dataset, do train/test split, then save to disk'''

  # check if the split datasets already exist
  train_file   = data_dir / "train.txt"
  test_file    = data_dir / "test.txt"
  d_split_file = data_dir / "d_split.pickle"

  if train_file.is_file() and test_file.is_file() and d_split_file.is_file():
    print("  split datasets already exist, load it")
    # load the split datasets
    with open(d_split_file, "rb") as f:
      d_split = pickle.load(f)
  else:
    print("  read corpus file into dataframe")
    corpus             = pd.read_csv(corpus_file, sep="\t", compression="gzip")
    corpus_txt         = corpus[["Corpus", "Topic"]]
    corpus_txt.columns = ["text", "label"]
    dataset            = Dataset.from_pandas(corpus_txt)
    
    print("  split train/test")
    d_split = dataset.train_test_split(test_size=test_size, seed=rand_seed)

    print("  save train and test sets into text files")
    def dataset_to_text(dataset, output_filename):
      """Utility function to save dataset text to disk"""
      with open(output_filename, "w") as f:
        for t in dataset["text"]:
          print(t, file=f)

    dataset_to_text(d_split["train"], train_file)
    dataset_to_text(d_split["test"], test_file)

    with open(d_split_file, "wb") as f:
      pickle.dump(d_split, f)

  print(f"  train:{d_split['train'].num_rows}, "+\
        f"test:{d_split['test'].num_rows}\n")

  return d_split

def train_tokenizer(train_file, model_dir, config):
  """Train tokenizer"""

  # for checking if tokenizer has been trained
  vocab_file               = model_dir / "vocab.txt"
  tokenizer_config_file    = model_dir / "tokenizer_config.json"

  if vocab_file.is_file() and tokenizer_config_file.is_file():
    print("  tokenizer has been trained")
  else:
    print("  train tokenizer")
    # initialize the WordPiece tokenizer
    bwp_tokenizer = BertWordPieceTokenizer()

    # train the tokenizer
    bwp_tokenizer.train(files=[str(train_file)], 
                        vocab_size=config['tokenize']['vocab_size'], 
                        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", 
                                        "[MASK]", "<S>", "<T>"])

    # enable truncation up to the maximum 512 tokens
    max_length = config['tokenize']['max_length']
    bwp_tokenizer.enable_truncation(max_length=max_length)

    # save the tokenizer  
    bwp_tokenizer.save_model(str(model_dir))

    # dumping some of the tokenizer config to config file, 
    with open(os.path.join(model_dir, "tokenizer_config.json"), "w") as f:
      tokenizer_cfg = {
          "do_lower_case": True,          "unk_token": "[UNK]",
          "sep_token": "[SEP]",           "pad_token": "[PAD]",
          "cls_token": "[CLS]",           "mask_token": "[MASK]",
          "model_max_length": max_length, "max_len": max_length,
      }
      json.dump(tokenizer_cfg, f)

  # load trained tokenizer as BertTokenizerFast
  btz_tokenizer = BertTokenizerFast.from_pretrained(model_dir)
  print("  return: BertTokenizerFast\n")

  return btz_tokenizer

def tokenize(d_split, tokenizer, data_dir, config):
  '''Tokenize train and test datasets'''

  # for checking  if tokenized datasets have been saved
  train_tokenzied = data_dir / "train_dataset_tokenized"
  test_tokenzied  = data_dir / "test_dataset_tokenized"

  if train_tokenzied.is_dir() and test_tokenzied.is_dir():
    print("  tokenized datasets have been saved, load them")
    # load the tokenized datasets
    train_dataset = Dataset.load_from_disk(train_tokenzied)
    test_dataset  = Dataset.load_from_disk(test_tokenzied)
  
  else:
    # Tokenize train and test sets
    def encode(examples):
      """Local function to tokenize the sentences passed with truncation"""
      return tokenizer(examples["text"], 
                       truncation=True, 
                       padding="max_length", 
                       max_length=config['tokenize']['max_length'], 
                       return_special_tokens_mask=True)

    print("  tokenize in batches")
    train_dataset = d_split["train"].map(encode, batched=True)
    test_dataset  = d_split["test"].map(encode, batched=True)

    print("  save tokenized datasets")
    train_dataset.save_to_disk(train_tokenzied)
    test_dataset.save_to_disk(test_tokenzied)

  print(f"  train:{train_dataset.num_rows}, test:{test_dataset.num_rows}\n")

  return train_dataset, test_dataset

def pretrain_bert(train_dataset, test_dataset, tokenizer, model_dir, config):
  '''Retrain BERT
  Args:
    train_dataset (Dataset): tokenized training data split
    test_dataset (Dataset): tokenized testing data split
    tokenizer (BertTokenizerFast): trained tokenizer
    model_dir (Path): model directory
    config (ConfigParser): config parser object
  '''

  # BERT configuration
  bert_config = config['pretrain']

  print("  set dataset format")
  train_dataset.set_format(type="torch",columns=["input_ids", "attention_mask"])
  test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

  print("  initialize model")
  # initialize the model with the config
  vocab_size   = config['tokenize']['vocab_size']
  max_length   = config['tokenize']['max_length']

  model_config = BertConfig(vocab_size=vocab_size, 
                            max_position_embeddings=max_length)
  model = BertForMaskedLM(config=model_config)

  # initialize the data collator, randomly masking 20% (default is 15%) of the 
  # tokens for the Masked Language Modeling (MLM) task
  data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=bert_config['mlm'], 
    mlm_probability=bert_config['mlm_prob'],
  )

  # Training arguments:
  #   output dir to where save model checkpoint
  #   evaluate each `logging_steps` steps
  #   num of training epochs, feel free to tweak
  #   training batch size, depend on GPU memory
  #   accumulate gradients before weights update 
  #   evaluation batch size
  #   evaluate, log and save every 500 steps
  #   save SafeTensors instead of Tensors 
  #   best in terms of loss
  #   save 3 model weights to save space
  training_args = TrainingArguments(
                output_dir                  = model_dir,           
                evaluation_strategy         = bert_config['eval_by'],
                overwrite_output_dir        = bert_config['overwrite_out'],
                num_train_epochs            = bert_config['num_epochs'],
                per_device_train_batch_size = bert_config['train_batch_size'],
                gradient_accumulation_steps = bert_config['grad_acc_steps'],
                per_device_eval_batch_size  = bert_config['eval_batch_size'],
                logging_steps               = bert_config['log_steps'],
                save_steps                  = bert_config['save_steps'],
                save_safetensors            = bert_config['safetensors'],
                load_best_model_at_end      = bert_config['load_best'],  
                # save_total_limit          = bert_config['save_limit'],
  )

  # initialize the trainer and pass everything to it
  trainer = Trainer(model=model,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
  )

  # train the model
  trainer.train()

  return model

################################################################################
if __name__== '__main__':

  print("Get config")
  args        = get_args()
  #work_dir    = Path(args.work_dir)     # working dir
  #data_file   = Path(args.data_file)  # corpus file path
  config_file = Path(args.config_file)  # config file path

  print(f"  config_file: {config_file}\n")

  with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

  work_dir  = Path(config['env']['work_dir'])
  data_file = work_dir / config['env']['data_file']
  print(f"  work_dir: {work_dir}")
  print(f"  data_file: {data_file}")

  # file system and general config
  rand_seed   = config['env']['rand_seed']
  model_dir   = work_dir / config['env']['model_dir_name']
  data_dir    = work_dir / config['env']['data_dir_name']
  test_size   = config['env']['test_size']

  # Create directories
  model_dir.mkdir(parents=True, exist_ok=True)
  data_dir.mkdir(parents=True, exist_ok=True)

  print("###\nConvert corpus data file to dataset")
  d_split = txt_to_split_dataset(data_file, test_size, rand_seed)

  print("###\nTrain tokenizer")
  train_file = data_dir / "train.txt"
  tokenizer = train_tokenizer(train_file, model_dir, config)

  print("###\nTokenize dataset")
  train_tkn, test_tkn = tokenize(d_split, tokenizer, data_dir, config)

  print("###\nPretrain model")
  model = pretrain_bert(train_tkn, test_tkn, tokenizer, model_dir, config)

  print("###\nDone")
  print("  tokenizer saved to: ", model_dir)
  print("  model saved to: ", model_dir)
  print("  intermediate data in: ", data_dir)  
