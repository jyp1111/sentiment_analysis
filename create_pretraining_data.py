# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import random
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def write_instance_to_example_files(instances, tokenizer, max_seq_length):
  """Create TF example files from `TrainingInstance`s."""
  writers = list()

  for instance in instances:
    tokens, masked_lm_positions, masked_lm_labels = instance
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length

    masked_lm_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

    labels = [-100]*max_seq_length
    for i,v in enumerate(masked_lm_positions):
      labels[v] = masked_lm_ids[i]

    input_ids = torch.tensor(input_ids)
    input_mask = torch.tensor(input_mask)
    position_ids = torch.tensor(range(max_seq_length))
    labels = torch.tensor(labels)
    writers.append((input_ids,input_mask,position_ids,labels))
  return writers

def create_training_instances(input_file, tokenizer, max_seq_length, dupe_factor, masked_lm_prob, rng):
  """Create `TrainingInstance`s from raw text."""
  all_documents = list()
  with open(input_file, "r", encoding="utf-8") as reader:
    while True:
      line = reader.readline()
      if not line:
        break
      line = line.strip()
      tokens = tokenizer.tokenize(line)
      if tokens:
        all_documents.append(tokens)

  # Remove empty documents
  all_documents = [x for x in all_documents if x]
  rng.shuffle(all_documents)

  vocab_words = list(tokenizer.vocab.keys())
  instances = []
  for _ in range(dupe_factor):
    for document_index in range(len(all_documents)):
      instances.append(create_instances_from_document(all_documents, document_index, max_seq_length, masked_lm_prob, vocab_words, rng))
  rng.shuffle(instances)
  return instances


def create_instances_from_document(all_documents, document_index, max_seq_length, masked_lm_prob, vocab_words, rng):
  """Creates `TrainingInstance`s for a single document."""
  document = all_documents[document_index]
  # Account for [CLS], [SEP]
  max_num_tokens = max_seq_length - 2
  
  tokens = ["[CLS]"]+document[:max_num_tokens]+["[SEP]"]
  instance = create_masked_lm_predictions(tokens, masked_lm_prob, vocab_words, rng)

  return instance


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob, vocab_words, rng):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    cand_indexes.append(i)

  rng.shuffle(cand_indexes)

  output_tokens = list(tokens)

  num_to_predict = max(1, int(round(len(tokens) * masked_lm_prob)))

  masked_lms = []
  covered_indexes = set()
  for index in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    if index in covered_indexes:
      continue
    covered_indexes.add(index)

    masked_token = None
    # 80% of the time, replace with [MASK]
    if rng.random() < 0.8:
      masked_token = "[MASK]"
    else:
      # 10% of the time, keep original
      if rng.random() < 0.5:
        masked_token = tokens[index]
      # 10% of the time, replace with random word
      else:
        masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

    output_tokens[index] = masked_token

    masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)


class DataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, random_seed, max_seq_length, dupe_factor, masked_lm_prob, input_file):
        super().__init__()
        self.input_file = input_file
        self.tokenizer = tokenizer
        self.random_seed = random_seed
        self.max_seq_length = max_seq_length
        self.dupe_factor = dupe_factor
        self.masked_lm_prob = masked_lm_prob
        self.train_data = None
        self.val_data = None
    def setup(self, stage):
        # Assign train/val datasets for use in dataloaders
      if stage == "fit" or stage is None:
        rng = random.Random(self.random_seed)
        instances = create_training_instances(self.input_file, self.tokenizer, self.max_seq_length, self.dupe_factor, self.masked_lm_prob, rng)
        outputs =  write_instance_to_example_files(instances, self.tokenizer, self.max_seq_length)
        self.train_data, self.val_data = train_test_split(outputs, test_size = 0.1, random_state = self.random_seed)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=32, pin_memory =True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=32, pin_memory =True)

