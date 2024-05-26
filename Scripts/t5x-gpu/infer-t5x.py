from collections.abc import Sequence
import enum
import functools
import inspect
import itertools
import logging
import os
import re
import time
from typing import Any, Callable, Iterator, Optional, Tuple, Union

import jax
from jax import random
from jax.experimental import multihost_utils
import numpy as np
import seqio
import tensorflow as tf
import tensorflow_datasets as tfds
import t5.data
import t5.data

import clu.data
from t5x.examples.t5 import network
import t5x
from t5x import models
from t5x import partitioning
from t5x import trainer as trainer_lib
from t5x import utils
from t5x.infer import _extract_tokens_and_aux_values
from t5x.infer import _Inferences
from t5x.interactive_model import InteractiveModel
from t5x.interactive_model import get_batches_from_seqio
from t5x.interactive_model import get_dataset_from_natural_text_examples
from t5x.interactive_model import get_gin_config_from_interactive_model
from t5x.interactive_model import T5XScriptType
from t5x.interactive_model import InferenceType

#os.environ['CUDA_VISIBLE_DEVICES'] = "-1" #4090
os.environ['CUDA_VISIBLE_DEVICES'] = "3" #4090

DEFAULT_SPM_PATH = "/opt/t5x/vocabs/cc_all.32000/sentencepiece.model"  # GCS
DEFAULT_EXTRA_IDS = 100
Data_dir = "/opt/t5x/T5-Data/4-TS-combined/"
output_dir = "/opt/t5x/models/finetuned/non-code-pretrained/TS-4/"
#checkpoint_path = os.path.join(output_dir, 'checkpoint_1013680')
checkpoint_path = os.path.join(output_dir, 'checkpoint_1300000')

# Define a model. The configuration below corresponds to the T5 1.1 Small model.
t5_config = network.T5Config(
    vocab_size=32128,
    dtype='bfloat16',
    emb_dim=512,
    num_heads=6,
    num_encoder_layers=8,
    num_decoder_layers=8,
    head_dim=64,
    mlp_dim=1024,
    mlp_activations=('gelu', 'linear'),
    dropout_rate=0.0,
    logits_via_embedding=False)
module = network.Transformer(config=t5_config)
model = t5x.models.EncoderDecoderModel(
    module=module,
    input_vocabulary=seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH, DEFAULT_EXTRA_IDS),
    output_vocabulary=seqio.SentencePieceVocabulary(DEFAULT_SPM_PATH, DEFAULT_EXTRA_IDS),
    optimizer_def=t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0))
# Define checkpoint arguments.
#checkpoint_path='gs://t5-data/pretrained_models/cbqa/small_ssm_nq/model.ckpt-1110000'
#checkpoint_path='/opt/t5x/models/pretrained/small_ssm_nq/model.ckpt-1110000'
checkpoint_path=checkpoint_path
dtype='bfloat16'
restore_mode='specific'
# Define a partitioner.
partitioner=partitioning.PjitPartitioner(num_partitions=1)
# Define additional, miscellaneous constructor arguments.
batch_size=4
task_feature_lengths = {'inputs': 512, 'targets': 512}
output_dir=output_dir
input_shapes = {
    'encoder_input_tokens': np.array([8, 38]),
    'decoder_target_tokens': np.array([8, 18]),
    'decoder_input_tokens': np.array([8, 18]),
    'decoder_loss_weights': np.array([8, 18])
}

init_random_seed = 42
random_seed = multihost_utils.broadcast_one_to_all(np.int32(init_random_seed))
utils.set_hardware_rng_ops()
rng = random.PRNGKey(random_seed)
init_rng, trainer_rng = random.split(rng, 2)

"""examples = [
    b'nq question: who has been appointed as the new chairman of sebi',
    b'nq question: who wrote the book lion the witch and the wardrobe',
    b'nq question: how many planes did japan lose at pearl harbor',
    b'nq question: who does the voice of mcgruff the dog',
    b'nq question: who sings the wheels in the sky keep on turning',
    b'nq question: who voices regina in glitter force doki doki',
    b'nq question: when did the us become allies with britain',
    b'nq question: who won the rugby 7 in las vegas'
]

for item in examples:
  print(type(item))
  print(item)"""

nq_tsv_path_assert_raw = {
    "train": os.path.join(Data_dir, "training.tsv"),
    "validation": os.path.join(Data_dir, "test.tsv"),
}
num_nq_examples_assert_raw = dict(train=138478, validation=17318)
def nq_dataset_assert_raw(split, shuffle_files=False):
  del shuffle_files

  # Load lines from the text file as examples.
  ds = tf.data.TextLineDataset(nq_tsv_path_assert_raw[split])
  ds = ds.map(
      functools.partial(tf.io.decode_csv, record_defaults=["string","string"],
                        field_delim="\t", use_quote_delim=False),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  ds = ds.map(lambda *ex: dict(zip(["method", "assert"], ex)))
  return ds

'''print("A few raw valid examples...")
for idx,ex in enumerate(tfds.as_numpy(nq_dataset_assert_raw("validation").take(5))):
  print(ex)
print()'''


def to_inputs_and_targets(ex):
  x_input = tf.strings.lower(ex['method'])
  y_label = tf.strings.lower(ex['assert'])
  inputs = tf.strings.join(['generate raw assert: ' + x_input], separator=' ')
  class_label = tf.strings.join([y_label], separator=' ')
  return {'input': inputs, 'target': class_label }

def atlas_preprocessing_input(ds):
  res = []
  for idx,ex in enumerate(tfds.as_numpy(ds)):
    res.append(to_inputs_and_targets(ex)['input'].numpy())
  return res
  #return [ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)]

def atlas_preprocessing_target(ds):
  res = []
  for idx,ex in enumerate(tfds.as_numpy(ds)):
    res.append(to_inputs_and_targets(ex)['target'].numpy())
  return res

examples = atlas_preprocessing_input(nq_dataset_assert_raw("validation"))
ground_truth = atlas_preprocessing_target(nq_dataset_assert_raw("validation"))

#examples = examples[0:500]

total = len(examples)
print(f"##########  test set size = {total} ##########")

for item in examples[0:5]:
  print(type(item))
  print(item)

output_features = {
        "inputs":
            seqio.Feature(
                vocabulary=model.input_vocabulary, add_eos=True),
        "targets":
            seqio.Feature(
                vocabulary=model.output_vocabulary, add_eos=True)
    }
features = dict(sorted(output_features.items()))

mode = InferenceType.PREDICT_WITH_AUX

dataset = get_dataset_from_natural_text_examples(
    examples,
    preprocessors=[
        seqio.preprocessors.tokenize,
        seqio.preprocessors.append_eos
    ],
    task_feature_lengths=task_feature_lengths,
    features=features)
feature_converter = model.FEATURE_CONVERTER_CLS(pack=False)
model_dataset = feature_converter(
    dataset, task_feature_lengths=task_feature_lengths)
# Zip task and model features.
infer_dataset = tf.data.Dataset.zip((dataset, model_dataset))
# Create batches and index them.
infer_dataset = infer_dataset.padded_batch(
    batch_size, drop_remainder=False).enumerate()
infer_dataset_iter: Iterator[Tuple[int, Any]] = iter(
    infer_dataset.prefetch(tf.data.experimental.AUTOTUNE))


# Define CheckpointCfg wrappers.
save_checkpoint_cfg = utils.SaveCheckpointConfig(
        dtype=dtype,
        keep=3, # The number of checkpoints to keep in the output_dir.
        save_dataset=False)
restore_checkpoint_cfg = utils.RestoreCheckpointConfig(
        dtype=dtype,
        mode=restore_mode,
        path=checkpoint_path)

train_state_initializer = utils.TrainStateInitializer(
        optimizer_def=model.optimizer_def,
        init_fn=model.get_initial_variables,
        input_shapes=input_shapes,
        input_types=None,
        partitioner=partitioner)

checkpoint_manager = utils.LegacyCheckpointManager(
        save_cfg=save_checkpoint_cfg,
        restore_cfg=restore_checkpoint_cfg,
        train_state_shape=train_state_initializer.global_train_state_shape,
        partitioner=partitioner,
        ds_iter=None,
        model_dir=output_dir)


def get_state(rng):
  return train_state_initializer.from_scratch(rng).state_dict()

# 1. Try to restore a model from a checkpoint.
train_state = checkpoint_manager.restore(
  [restore_checkpoint_cfg.path],
  restore_checkpoint_cfg,
  utils.get_fallback_state(restore_checkpoint_cfg, get_state, init_rng)
)

if mode == InferenceType.PREDICT_WITH_AUX:
  infer_step = model.predict_batch_with_aux
elif mode == InferenceType.SCORE:
  infer_step = model.score_batch
else:
  raise ValueError("Mode must be `predict_with_aux`, or `score`,"
                  f" but instead was {mode}.")
infer_fn = functools.partial(
  utils.get_infer_fn(
    infer_step=infer_step,
    batch_size=batch_size,
    train_state_axes=train_state_initializer.train_state_axes,
    partitioner=partitioner),
  train_state=train_state)

start_time = time.time()

# Main Loop over "batches".
all_inferences = []
all_aux_values = {}
for chunk, chunk_batch in infer_dataset_iter:
  # Load the dataset for the next chunk. We can't use `infer_dataset_iter`
  # directly since `infer_fn` needs to know the exact size of each chunk,
  # which may be smaller for the final one.
  chunk_dataset = tf.data.Dataset.from_tensor_slices(chunk_batch)
  chunk_dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)

  # Unzip chunk dataset in to pretokenized and model datasets.
  task_dataset = chunk_dataset.map(
      lambda p, m: p, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  model_dataset = chunk_dataset.map(
      lambda p, m: m, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # Get a chunk-specific RNG key.
  chunk_rng = jax.random.fold_in(jax.random.PRNGKey(0), chunk)

  inferences = _extract_tokens_and_aux_values(
      infer_fn(model_dataset.enumerate(), rng=chunk_rng))

  predictions, aux_values = inferences
  accumulated_inferences = []
  for idx, inputs in task_dataset.enumerate().as_numpy_iterator():
    prediction = predictions[idx]
    # Decode predictions if applicable.
    if mode == InferenceType.PREDICT_WITH_AUX:
      prediction = features["targets"].vocabulary.decode_tf(
          tf.constant(prediction)).numpy()
    accumulated_inferences.append((inputs, prediction))
  all_inferences += accumulated_inferences
  # Accumulate aux values over batches.
  if not all_aux_values:
    all_aux_values = aux_values
  else:
    for key, values in aux_values.items():
      all_aux_values[key] += values

cor = 0
for i in range(total):
  input, prediction = all_inferences[i]
  #print(f"pre:{prediction.decode(encoding='utf-8')};\n ground:{ground_truth[i].decode(encoding='utf-8')}")
  if prediction.decode(encoding="utf-8") == ground_truth[i].decode(encoding="utf-8"):
    cor += 1

print(f"acc = {cor/total*100:.3f}%; correct number = {cor}; total number = {total}")
print(f'time consume:{time.time()-start_time}s')

#print(all_inferences)
'''for input, prediction in all_inferences:
  print(f"Input: {input['inputs_pretokenized']}")
  print(f"Prediction: {prediction}\n")
  print(type(prediction))
  print()'''

with open(os.path.join(output_dir, "prediction.txt"), "w", encoding="utf-8") as file:
  for input, prediction in all_inferences:
    file.write(prediction.decode(encoding="utf-8") + "\n")

'''with open(os.path.join(output_dir, "ground_truth.txt"), "w", encoding="utf-8") as file:
  for item in ground_truth:
    file.write(item.decode(encoding="utf-8")+"\n")'''