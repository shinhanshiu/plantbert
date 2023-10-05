# plantbert

## Goal

Pre-train a BERT model using plant sciecne corpus.

## Issues

### Tokenization

To train a tokenizer, the common practice is to consider wordpiece or similar approaches (e.g., byte-pair encoding) for reduce the size of vocab. But these approaches keeps frequent words intact and generate word pieces for less frequent words. Considering names for entities such as genes, proteins, and metabolites will be mostly rare in a corpus, they will almost always become word pieces which may (or may not) lead to issues with attention scores.

Possible solution:
- Use as is and see how bad it is.
- Use distinct words as tokens: But this will lead to a very large vocab. For the plant science corpus, there are >900,000 distinct words.
- Add genes, proteins, and metabolites manually into vocab after training tokenizers: But it is not trivial to collect all known gene names or other molecular entities from ALL species.
- Define vocab size with a threshold frequency: this need to be tested.

### Pre-training

Mask language modeling
- Random: default.
- Specific tokens: This would involve modifying `tf_mask_tokens()` in [DataCollatorForLanguageModeling](https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L607). Not sure if it is feasible.

### Resource consideration

Memory
- With 379k docs, vocab size=30_522, max_length=512, mlm_prob=0.2
- For RTX 3090 with 24Gb VRAM, can only get training batch size to 25 before problems arise.

Run time
- For 10 epochs, need 35 days on a NVDIA RTX 3070; 1.5 days on an RTX 3090.
