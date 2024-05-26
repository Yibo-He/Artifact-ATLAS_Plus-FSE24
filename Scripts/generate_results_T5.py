#!/usr/bin/env python
# coding: utf-8
#Installing libraries
#get_ipython().system('pip install transformers==3.0.2')
#get_ipython().system('pip install nlp==0.4.0')
#get_ipython().system('pip install pyarrow==0.16.0')
#get_ipython().system('pip install sentencepiece')

import torch
import nlp
from tqdm import tqdm
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
import pandas as pd

spm_path = '/workspace/T5/finetuned-model/Pytorch-Model/dl4se_vocab.model'
config_file = '/workspace/T5/finetuned-model/Pytorch-Model/config.json'
config = T5Config.from_json_file(config_file)
tokenizer = T5Tokenizer.from_pretrained(spm_path)

def add_eos_to_examples(example):
    example['input_text'] = 'generate small patch: %s </s>' % example['method'].lower()
    example['target_text'] = '%s </s>' % example['assertion'].lower()
    return example


def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'], pad_to_max_length=True, max_length=512, truncation=True)
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], pad_to_max_length=True, max_length=512, truncation=True)

    encodings = {
        'input_ids': input_encodings['input_ids'], 
        'attention_mask': input_encodings['attention_mask'],
        'target_ids': target_encodings['input_ids'],
        'target_attention_mask': target_encodings['attention_mask']
    }

    return encodings

valid_dataset = nlp.load_dataset('/workspace/T5/T5-learning-ICSE_2021-main/Code/run-on-test-set/AG/assertion_dataset_script.py', split=nlp.Split.TEST)

# map add_eos_to_examples function to the dataset example wise 
valid_dataset = valid_dataset.map(add_eos_to_examples, load_from_cache_file=False)

# map convert_to_features batch wise
valid_dataset = valid_dataset.map(convert_to_features, batched=True, load_from_cache_file=False)

columns = ['input_ids', 'target_ids', 'attention_mask','target_attention_mask']
valid_dataset.set_format(type='torch', columns=columns)

BATCH_SIZE = 16
dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE)

df = pd.read_csv('/workspace/T5/newtestdatasets/1/test.tsv',header=None,sep='\t')

references=[]

for item in df[1]:
  references.append(item.lower())

references[1]

CUDA = torch.device("cuda")

finetuned_model_path = '/workspace/T5/finetuned-model/Pytorch-Model/model.bin'

model = T5ForConditionalGeneration.from_pretrained(
        finetuned_model_path,
        config=config
        ).to(CUDA)
        
model.eval()

from tqdm import tqdm
import time

predictions = []

BEAM_SIZE = 1

torch.cuda.empty_cache()

start_time = time.time()
for batch in tqdm(dataloader):

      outs = model.generate(
                          input_ids=batch['input_ids'].to(CUDA),
                          attention_mask=batch['attention_mask'].to(CUDA),
                          num_beams=BEAM_SIZE, 
                          max_length=128,
                          num_return_sequences=BEAM_SIZE, 
                          early_stopping=True
                          )
    

    
      outs = [tokenizer.decode(ids, skip_special_tokens=True)  for ids in outs]
      predictions.extend(outs)

pred_refined = []
for pred in predictions:
    if len(pred)>=2:
      if pred[0]=='"':
          pred = pred[1:]
      if pred[-1]=='"':
          pred = pred[:-1]
    pred_refined.append(pred)

counter_pred = 0

mispred_list = []

sanity_check_list = []

idx = 0

len_prediction=(len(pred_refined))

for i in range(0, len_prediction, BEAM_SIZE):

    items_to_analyze = pred_refined[i:i+BEAM_SIZE]
    target_item = ''.join(references[idx].split(' '))
    

    for pred in items_to_analyze:
        pred_ref = ''.join(pred.split(' '))
        if pred_ref == target_item:
            counter_pred+=1
            sanity_check_list.append(pred_ref)
            break
        else:
          mispred_list.append(pred)
         
          
        
    idx += 1

print('% of perfect predictions: ',(counter_pred/len(references))*100 )
print(f'correct pred number:{counter_pred}')
print(f'time consume:{time.time()-start_time}s')

idx=0

with open('./mispredictions.txt', 'w') as f:
    for i in range( 0, len(mispred_list), BEAM_SIZE):
        
        items_to_analyze = mispred_list[i:i+BEAM_SIZE]

        f.write('\n************\n')
        f.write("tgt: %s\n" % references[idx])
        for (index,mispred) in enumerate(items_to_analyze):
          f.write('[%s]: %s\n' % (str(index),mispred) )
        f.write('\n************\n')

        idx+=1

with open('./predictions.txt', 'w') as f:
    for item in pred_refined:
        f.write("%s\n" % item)

