# coding=utf-8

import argparse
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
import pytorch_lightning as pl
import torch
from create_pretraining_data import DataModule
from pytorch_lightning.callbacks import ModelCheckpoint

class Pretrain_model(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.config = AutoConfig.from_pretrained('bert-base-multilingual-cased')
    self.model = AutoModelForMaskedLM.from_config(self.config)
  def forward(self,x):
    input_ids, input_mask, position_ids, labels = x
    outputs = self.model(input_ids=input_ids, attention_mask=input_mask, position_ids=position_ids, labels=labels)
    return outputs
  
  def training_step(self, batch, batch_idx):
    loss = self(batch)[0]
    self.log("train_loss", loss, on_step=True, logger=True)
    return loss
  
  def configure_optimizers(self):
      return torch.optim.Adam(self.parameters(), lr=2e-5)
    
  def configure_callbacks(self):
      checkpoint = ModelCheckpoint(monitor="val_loss")
      return [checkpoint]
    
  def validation_step(self, batch, batch_idx):
    loss = self(batch)[0]
    return loss, len(batch)
  
  def validation_epoch_end(self, validation_step_outputs):
    total_loss = 0
    cnt = 0
    for loss,batch_size in validation_step_outputs:
      total_loss += loss*batch_size
      cnt +=  batch_size
    val_loss = total_loss/cnt
    self.log("val_loss",val_loss)
    return val_loss
  


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--init_checkpoint",
                      default=None,
                      type=str,
                      help="Initial checkpoint (usually from a pre-trained BERT model).")
  parser.add_argument("--input_file",
                      default="steam_data_1.txt",
                      type=str,
                      help="Input text file")
  parser.add_argument('--seed', 
                      type=int, 
                      default=42,
                      help="random seed for initialization")
  parser.add_argument("--max_seq_length",
                      default=128,
                      type=int,
                      help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")
  parser.add_argument('--dupe_factor', 
                      type=int, 
                      default=1,
                      help="The number of repeating create instances")
  parser.add_argument('--masked_lm_prob',
                      type=float,
                      default=0.15,
                      help="The probability of masked tokens in the token sequence")
  parser.add_argument("--num_train_epochs",
                      default=3.0,
                      type=float,
                      help="Total number of training epochs to perform.")
  args = parser.parse_args()
  
  tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
  random_seed = args.seed
  max_seq_length = args.max_seq_length
  dupe_factor = args.dupe_factor
  masked_lm_prob = args.masked_lm_prob
  input_file = args.input_file
  
  pl.seed_everything(random_seed)
  data = DataModule(tokenizer, random_seed, max_seq_length, dupe_factor, masked_lm_prob, input_file)
  trainer = pl.Trainer(max_epochs = args.num_train_epochs, accelerator="cpu")
  model = Pretrain_model()
  trainer.fit(model,data,ckpt_path=args.init_checkpoinit)


if __name__ == "__main__":
  main()
