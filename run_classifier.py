# coding=utf-8

import os
import argparse
import pandas as pd
import logging
from transformers import AutoTokenizer,AutoModelForSequenceClassification,AutoConfig 
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.functional as F

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    contexts, labels = examples
    features = []
    for (ex_index, context) in enumerate(contexts):
            # Account for [CLS] and [SEP] with "- 2"
        if len(context) > max_seq_length - 2:
            context = context[:max_seq_length - 2]

        tokens = ["[CLS]"]+context+["SEP"]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        position_ids = torch.tensor(range(max_seq_length))
        label = torch.tensor(labels[ex_index])
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        features.append((input_ids, input_mask, position_ids, label))
    return features

class DataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, random_seed, max_seq_length, data_dir, train_batch_size, eval_batch_size):
        super().__init__()
        self.tokenizer = tokenizer
        self.random_seed = random_seed
        self.max_seq_length = max_seq_length
        self.data_dir = data_dir
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        contexts, labels = list(), list()
        for line in lines:	
            context,label = line
            contexts.append(self.tokenizer.tokenize(context))
            labels.append(label)
        return contexts, labels
    def setup(self, stage):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            train_data = pd.read_csv(os.path.join(self.data_dir, "train.csv")).values
            examples = self._create_examples(train_data)
            features = convert_examples_to_features(examples,self.max_seq_length, self.tokenizer)
            self.train_data, self.val_data = train_test_split(features, test_size = 0.1, random_state = self.random_seed)
        if stage == "test" or stage is None:
            test_data = pd.read_csv(os.path.join(self.data_dir, "test.csv")).values
            examples = self._create_examples(test_data)
            features = convert_examples_to_features(examples,self.max_seq_length, self.tokenizer)
            self.test_data = features
            
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.train_batch_size, pin_memory =True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.train_batch_size, pin_memory =True)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.eval_batch_size)
    
class Classifier_model(pl.LightningModule):
    def __init__(self,model,optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        
    def forward(self,x):
        features = x
        input_ids, input_mask, position_ids, labels = features
        outputs = self.model(input_ids=input_ids, attention_mask=input_mask, position_ids=position_ids, labels=labels)
        return outputs

    def training_step(self, batch, batch_idx):
        loss = self(batch)[0]
        self.log("train_loss", loss, on_step=True, logger=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer

    def configure_callbacks(self):
        checkpoint = ModelCheckpoint(monitor="val_loss")
        return [checkpoint]

    def validation_step(self, batch, batch_idx):
        loss = self(batch)[0]
        return loss, len(batch[0])

    def validation_epoch_end(self, validation_step_outputs):
        total_loss = 0
        cnt = 0
        for loss,batch_size in validation_step_outputs:
            total_loss += loss*batch_size
            cnt +=  batch_size
        val_loss = total_loss/cnt
        self.log("val_loss",val_loss)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        _, _, _, labels = batch
        logits = self(batch)[1]
        preds = torch.argmax(F.softmax(logits), dim=1)
        correct_cnt = 0
        for i,pred in enumerate(preds):
            if labels[i] == pred:
                correct_cnt += 1
        return correct_cnt, len(preds)
    
    def test_epoch_end(self, outputs):
        total_correct,cnt = 0,0
        for correct_cnt,batch_size in outputs:
            total_correct += correct_cnt
            cnt += batch_size
        accuracy = total_correct/cnt
        self.log("test_acc", accuracy)
        

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default="./data",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    ## Other parameters
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--pretrained_model_path",
                        default=None,
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--discr",
                        default=False,
                        action='store_true',
                        help="Whether to do discriminative fine-tuning.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--layer_learning_rate',
                        type=float,
                        default=2e-5,
                        help="learning rate in each group")
    parser.add_argument('--layer_learning_rate_decay',
                        type=float,
                        default=0.95)                   
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

        
    config = AutoConfig.from_pretrained('bert-base-multilingual-cased')
    pretrained_model = AutoModelForSequenceClassification.from_config(config)
    if args.pretrained_model_path is not None:
        state_dict=torch.load(args.pretrained_model_path)["state_dict"]
        new_dict = {}
        for key in pretrained_model.bert.state_dict().keys():
            if "pooler" in key:
                new_dict[key] = pretrained_model.bert.state_dict()[key]
            else:
                new_dict[key] = state_dict["model.bert."+key]
        pretrained_model.bert.load_state_dict(new_dict)
    if args.init_checkpoint is not None:
        state_dict=torch.load(args.init_checkpoint)["state_dict"]
        new_dict = {}
        for key in pretrained_model.bert.state_dict().keys():
            new_dict[key] = state_dict["model.bert."+key]
        pretrained_model.bert.load_state_dict(new_dict)

    if args.discr:
        lr = args.layer_learning_rate
        groups = [(f'layer.{i}.', lr * pow(args.layer_learning_rate_decay, 11 - i)) for i in range(12)]
        group_all = [f'layer.{i}.' for i in range(12)]
        no_decay_optimizer_parameters = []
        decay_optimizer_parameters = []
        for g, l in groups:
            decay_optimizer_parameters.append(
                {
                    'params': [p for n, p in pretrained_model.named_parameters() if "bias" not in n and g in n],
                    'weight_decay_rate': 0.01, 'lr': l
                }
            )
            no_decay_optimizer_parameters.append(
                {
                    'params': [p for n, p in pretrained_model.named_parameters() if "bias" in n and g in n],
                    'weight_decay_rate': 0.0, 'lr': l
                }
            )

        group_all_parameters = [
            {'params': [p for n, p in pretrained_model.named_parameters() if ("bias" not in n) and not any(nd in n for nd in group_all)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in pretrained_model.named_parameters() if ("bias" in n) and not any(nd in n for nd in group_all)],
             'weight_decay_rate': 0.0},
        ]
        optimizer_parameters = no_decay_optimizer_parameters + decay_optimizer_parameters + group_all_parameters

    else:
        optimizer_parameters = [
            {'params': [p for n, p in pretrained_model.named_parameters() if "bias" not in n],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in pretrained_model.named_parameters() if "bias" in n],
             'weight_decay_rate': 0.0}
        ]		
    optimizer = torch.optim.Adam(optimizer_parameters, lr=args.learning_rate)
    random_seed = args.seed
    max_seq_length = args.max_seq_length
    data_dir = args.data_dir
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    trainer = pl.Trainer(max_epochs = args.num_train_epochs, accelerator="cpu")
    pl.seed_everything(random_seed)
    data = DataModule(tokenizer, random_seed, max_seq_length, data_dir, train_batch_size, eval_batch_size)
    model = Classifier_model(pretrained_model,optimizer)
    if args.do_train:
        trainer.fit(model,data,ckpt_path=args.init_checkpoint)
    elif args.do_eval:
        trainer.test(model,data,ckpt_path=args.init_checkpoint)

if __name__ == "__main__":
    main()
