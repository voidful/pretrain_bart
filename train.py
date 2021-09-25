import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import (
    AutoTokenizer
)
from model.bart import BARTDPTModel
from datasets import load_dataset
import pytorch_lightning as pl

model_config = "facebook/bart-base"
tokenizer_config = "facebook/mbart-large-50-one-to-many-mmt"
batch_size = 2

train_dataset = load_dataset("text", data_files={'data': './wiki_08_05_1215639.txt'}, split='data[:80%]')
test_dataset = load_dataset("text", data_files={'data': './wiki_08_05_1215639.txt'}, split='data[80%:]')
train_dataset = train_dataset.load_from_disk("bart_pretrain_data_train")
test_dataset = test_dataset.load_from_disk("bart_pretrain_data_test")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_config)
bart_dpt = BARTDPTModel(model_config, tokenizer_config, train_datalist=train_dataset,
                        eval_datalist=test_dataset, batch_size=batch_size)

es = EarlyStopping(monitor='dev_loss')
trainer = pl.Trainer(gpus=1, check_val_every_n_epoch=1, callbacks=[es, ModelCheckpoint(
    monitor='dev_loss', filename='{epoch}-{dev_loss:.2f}', save_last=True, )],
                     default_root_dir='./bart_dpt/', auto_lr_find=True)
trainer.tune(bart_dpt)
trainer.fit(bart_dpt)
