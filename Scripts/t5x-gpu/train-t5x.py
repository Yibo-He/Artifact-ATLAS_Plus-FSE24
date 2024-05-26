from collections.abc import Sequence
import enum
import functools
import inspect
import itertools
import logging
import os
import re
from typing import Any, Callable, Iterator, Optional, Tuple, Union
import warnings
warnings.filterwarnings("ignore")

import jax
from jax import random
from jax.experimental import multihost_utils
import numpy as np
import seqio
import tensorflow as tf
import tensorflow_datasets as tfds
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

os.environ['CUDA_VISIBLE_DEVICES'] = "3" #4090

Data_dir = "/opt/t5x/T5-Data/0-ATLAS/"
iter_per_step = 1000
epcho = 300

output_dir = '/opt/t5x/models/finetuned/non-code-pretrained/ATLAS-0'
checkpoint_path = '/opt/t5x/models/pretrained/t5_1_1_small/checkpoint_1000000'

#output_dir = "/opt/t5x/models/finetuned/code-pretrained/TS-1"
#checkpoint_path = '/opt/t5x/models/pretrained/small_code_pretrained/model.ckpt-249900'

DEFAULT_SPM_PATH = "/opt/t5x/vocabs/cc_all.32000/sentencepiece.model"  # GCS
DEFAULT_EXTRA_IDS = 100

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
batch_size=8
task_feature_lengths = {'inputs': 512, 'targets': 512}
output_dir=output_dir
input_shapes = {
    'encoder_input_tokens': np.array([8, 512]),
    'decoder_target_tokens': np.array([8, 512]),
    'decoder_input_tokens': np.array([8, 512]),
    'decoder_loss_weights': np.array([8, 512])
}


init_random_seed = 42
random_seed = multihost_utils.broadcast_one_to_all(np.int32(init_random_seed))
utils.set_hardware_rng_ops()
rng = random.PRNGKey(random_seed)
init_rng, trainer_rng = random.split(rng, 2)

if partitioner._model_parallel_submesh:
  num_partitions = np.prod(partitioner._model_parallel_submesh)
else:
  num_partitions = partitioner._num_partitions
if jax.device_count() % num_partitions != 0:
  raise ValueError(
    "The number of devices available must be a multiple of the number of",
    f" partitions. There are {jax.device_count()} devices available, but",
    f" the number of partitions is set to {num_partitions}. Please",
    " provide a different number of partitions.")


# Define CheckpointCfg wrappers.
save_checkpoint_cfg = utils.SaveCheckpointConfig(
        #period=2000,
        dtype=dtype,
        keep=5, # The number of checkpoints to keep in the output_dir.
        save_dataset=False)
restore_checkpoint_cfg = utils.RestoreCheckpointConfig(
        dtype=dtype,
        mode=restore_mode,
        path=checkpoint_path)

# Define a train state initializer, which will help us get information about the
# TrainState shape.
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

# 2. If no checkpoint to restore, init from scratch.
if train_state is None:
  train_state = train_state_initializer.from_scratch(init_rng)

# Validate that we got an expected form of TrainState.
if isinstance(train_state, Sequence):
  raise ValueError(
    "Expected a single train state, but instead received a Sequence.")
train_state_axes = train_state_initializer.train_state_axes

# Log the variable shapes information and write to a file.
log_file = os.path.join(output_dir, "model-info.txt")
utils.log_model_info(log_file,
                     train_state_initializer.global_train_state_shape,
                     partitioner)

trainer = trainer_lib.Trainer(
  model=model,
  train_state=train_state,
  partitioner=partitioner,
  eval_names=[],
  summary_dir=output_dir,
  train_state_axes=train_state_axes,
  rng=trainer_rng,
  learning_rate_fn=utils.create_learning_rate_scheduler(),
  num_microbatches=None)

output_features = {
        "inputs":
            seqio.Feature(
                vocabulary=model.input_vocabulary, add_eos=True),
        "targets":
            seqio.Feature(
                vocabulary=model.output_vocabulary, add_eos=True)
    }
features = dict(sorted(output_features.items()))


"""examples = [
  {
      'target': b'Ajay Tyagi',
      'input':b'nq question: who has been appointed as the new chairman of sebi'
  },
  {
      'target': b'C. S. Lewis',
      'input': b'nq question: who wrote the book lion the witch and the wardrobe'},
  {
      'target': b'29',
      'input': b'nq question: how many planes did japan lose at pearl harbor'},
  {
      'target': b'Jack Keil',
      'input': b'nq question: who does the voice of mcgruff the dog'},
  {
      'target': b'Journey',
      'input': b'nq question: who sings the wheels in the sky keep on turning'},
  {
      'target': b'Kumiko Watanabe',
      'input': b'nq question: who voices regina in glitter force doki doki'},
  {
      'target': b'during World War II',
      'input': b'nq question: when did the us become allies with britain'},
  {
      'target': b'the United States',
      'input': b'nq question: who won the rugby 7 in las vegas'},
]
"""

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


def to_inputs_and_targets(ex):
  x_input = tf.strings.lower(ex['method'])
  y_label = tf.strings.lower(ex['assert'])
  inputs = tf.strings.join(['generate raw assert: ' + x_input], separator=' ')
  class_label = tf.strings.join([y_label], separator=' ')
  return {'input': inputs.numpy(), 'target': class_label.numpy() }

def atlas_preprocessing_raw(ds):
  res = []
  for idx,ex in enumerate(tfds.as_numpy(ds)):
    res.append(to_inputs_and_targets(ex))
  return res
  #return [ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)]

examples = atlas_preprocessing_raw(nq_dataset_assert_raw("train"))

print(f"##########  training set size = {len(examples)} ##########")

print("A few training examples...")
for i in range(3):
  print(examples[i])

print()

# Validate num examples.
if len(examples) < batch_size:
  raise ValueError(
    "At least one batch of data must be provided. Please decrease the "
    "batch_size or provide more examples.")
# Get a tf.Dataset.
train_dataset = get_dataset_from_natural_text_examples(
    examples=examples,
    preprocessors=[
        seqio.preprocessors.tokenize,
        seqio.preprocessors.append_eos
    ],
    task_feature_lengths=task_feature_lengths,
    features=features)


print("A few preprocessed training examples...")
for ex in tfds.as_numpy(train_dataset.take(3)):
  print(ex)
print()


# Convert and pad features.
feature_converter = model.FEATURE_CONVERTER_CLS(pack=False)
train_dataset = feature_converter(
        train_dataset, task_feature_lengths=task_feature_lengths)
train_dataset = train_dataset.padded_batch(batch_size, drop_remainder=True)
train_iter = clu.data.dataset_iterator.TfDatasetIterator(train_dataset, checkpoint=False)


def train_step(
    trainer: t5x.trainer.Trainer,
    train_state: t5x.train_state.TrainState,
    train_iter: clu.data.dataset_iterator.TfDatasetIterator,
    checkpoint_manager: utils.LegacyCheckpointManager,
    save_checkpoint_cfg: utils.SaveCheckpointConfig):
  # Validate that training can occur.
  if trainer.stop_training:
    logging.info("Stopping training early since `stop_training` is requested.")
    return

  # Take a training step.
  try:
    first_step = int(utils.get_local_data(train_state.step))
    train_summary = trainer.train(
      train_iter, iter_per_step, start_step=first_step)
  except trainer_lib.PreemptionError as e:
    logging.info("Saving emergency checkpoint.")
    checkpoint_manager.save(
      trainer.train_state,
      save_checkpoint_cfg.state_transformation_fns)
    logging.info("Saving emergency checkpoint done.")
    raise e

  # Save a checkpoint.
  logging.info("Saving checkpoint.")
  checkpoint_manager.save(
      trainer.train_state,
      save_checkpoint_cfg.state_transformation_fns)

  # Wait until computations are done before exiting
  multihost_utils.sync_global_devices("complete")
  return trainer.train_state, train_summary.result()

'''
print(f"Current Step: {train_state.step}")
train_state, train_summary = train_step(trainer, train_state, train_iter, checkpoint_manager, save_checkpoint_cfg)
print(f"Current Step: {train_state.step}")
print(f"Summary of Training: {train_summary}")
'''
init_step = train_state.step

for _ in range(epcho):
  # Reset the iterator, since we use the same batch for every step.
  train_iter = clu.data.dataset_iterator.TfDatasetIterator(train_dataset, checkpoint=False)
  train_state, train_summary = train_step(
      trainer,
      train_state,
      train_iter,
      checkpoint_manager,
      save_checkpoint_cfg
  )
  print(f"Current step: {train_state.step}; Total step: {init_step + epcho*iter_per_step}")

print(f"Summary of Training: {train_summary}")