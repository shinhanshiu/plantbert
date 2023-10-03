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

print("\n###################\nImporting packages\n")

import os, json, argparse, pickle, yaml, sys
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
  
  parser.add_argument('-c', '--config_path', 
                      type=str,
                      help='Config file path',
                      required=True)
  
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
    dataset_to_text(d_split["train"], train_file)
    dataset_to_text(d_split["test"], test_file)

    with open(d_split_file, "wb") as f:
      pickle.dump(d_split, f)

  print(f"  train:{d_split.train.num_rows}, test:{d_split['test'].num_rows}")

  return d_split

def dataset_to_text(dataset, output_filename="data.txt"):
  """Utility function to save dataset text to disk"""
  with open(output_filename, "w") as f:
    for t in dataset["text"]:
      print(t, file=f)

def tokenize(d_split, model_dir, config):
  """Train tokenizer and tokenize datasets"""

  # tokenization config
  config_t   = config['tokenization']
  vocab_size = int(config_t['vocab_size'])
  max_length = int(config_t['max_length'])

  # for checking if tokenizer has been trained
  vocab_file               = model_dir / "vocab.txt"
  tokenizer_config_file    = model_dir / "tokenizer_config.json"
  vocab_is_file            = vocab_file.is_file()
  tokenizer_config_is_file = tokenizer_config_file.is_file()

  if vocab_is_file and tokenizer_config_is_file:
    print("  tokenizer has been trained")
  else:
    print("  train tokenizer")
    # initialize the WordPiece tokenizer
    bwp_tokenizer = BertWordPieceTokenizer()

    # train the tokenizer
    bwp_tokenizer.train(files=["train.txt"], vocab_size=vocab_size, 
                        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", 
                                        "[MASK]", "<S>", "<T>"])

    # enable truncation up to the maximum 512 tokens
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

  # for checking  if tokenized datasets have been saved
  train_tokenzied_file = work_dir / "train_dataset_tokenized"
  test_tokenzied_file  = work_dir / "test_dataset_tokenized"
  train_tokenized_is_file = train_tokenzied_file.is_file()
  test_tokenized_is_file  = test_tokenzied_file.is_file()

  if train_tokenized_is_file and test_tokenized_is_file:
    print("  tokenized datasets have been saved, load them")
    # load the tokenized datasets
    train_dataset = Dataset.load_from_disk("train_dataset_tokenized")
    test_dataset  = Dataset.load_from_disk("test_dataset_tokenized")
  else:
    # Tokenize train and test set
    def encode(examples):
      """Local function to tokenize the sentences passed with truncation"""
      return btz_tokenizer(examples["text"], 
                            truncation=True, 
                            padding="max_length", 
                            max_length=max_length, 
                            return_special_tokens_mask=True)

    print("  tokenize in batches")
    train_dataset = d_split["train"].map(encode, batched=True)
    test_dataset  = d_split["test"].map(encode, batched=True)

    print("  save tokenized datasets")
    train_dataset.save_to_disk("train_dataset_tokenized")
    test_dataset.save_to_disk("test_dataset_tokenized")

  return train_dataset, test_dataset, btz_tokenizer

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
  bert_config = config['bert_retraining']

  # tokenizer configuration
  tokenizer_config = config['tokenization']

  print("  set dataset format")
  train_dataset.set_format(type="torch",columns=["input_ids", "attention_mask"])
  test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

  print("  initialize model")
  # initialize the model with the config
  vocab_size   = int(tokenizer_config['vocab_size'])
  max_pos_emb  = int(tokenizer_config['max_length'])
  model_config = BertConfig(vocab_size=vocab_size, 
                            max_position_embeddings=max_pos_emb)
  model        = BertForMaskedLM(config=model_config)

  # initialize the data collator, randomly masking 20% (default is 15%) of the 
  # tokens for the Masked Language Modeling (MLM) task
  mlm = bert_config['mlm']
  if mlm == "True":
    mlm = True
  else:
    mlm = False

  data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=mlm, 
    mlm_probability=float(bert_config['mlm_probability'])
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
    output_dir=model_dir,           
    evaluation_strategy=bert_config['evaluation_strategy'],
    overwrite_output_dir=bert_config['overwrite_output_dir'],
    num_train_epochs=bert_config['num_train_epochs'],
    per_device_train_batch_size=bert_config['per_device_train_batch_size'],
    gradient_accumulation_steps=bert_config['gradient_accumulation_steps'],
    per_device_eval_batch_size=bert_config['per_device_eval_batch_size'],
    logging_steps=bert_config['logging_steps'],
    save_steps=bert_config['save_steps'],
    save_safetensors=bert_config['save_safetensors'],
    # load_best_model_at_end=True,  
    # save_total_limit=3,
  )

  # initialize the trainer and pass everything to it
  trainer = Trainer(
      model=model,
      args=training_args,
      data_collator=data_collator,
      train_dataset=train_dataset,
      eval_dataset=test_dataset,
  )

  # train the model
  trainer.train()


################################################################################
if __name__== '__main__':

  print("Get config\n")
  args        = get_args()
  config_path = args.config_path
  with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

  # file system and general config
  rand_seed   = config['env']['rand_seed']
  work_dir    = Path(config['env']['work_dir'])
  model_dir   = work_dir / config['env']['model_dir_name']
  data_dir    = work_dir / config['env']['data_dir_name']
  corpus_file = data_dir / config['env']['corpus_name']
  test_size   = config['env']['test_size']

  # Create directories
  work_dir.mkdir(parents=True, exist_ok=True)
  model_dir.mkdir(parents=True, exist_ok=True)

  print("Convert corpus file to dataset")
  d_split = txt_to_split_dataset(corpus_file, test_size, rand_seed)

  sys.exit(0)

  print("Tokenize dataset")
  train_dataset, test_dataset, btz_tokenizer = tokenize(d_split, model_dir, 
                                                        config)

  print("Pretrain model")
  pretrain_bert(train_dataset, test_dataset, btz_tokenizer, model_dir, config)
