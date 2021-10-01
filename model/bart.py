"""PyTorch Lighting bart model. """

import torch
import torch.utils.checkpoint
from transformers import BartForConditionalGeneration, AutoTokenizer
from transformers.utils import logging
import pytorch_lightning as pl
import torch.utils.data as data
from functools import partial
from tqdm.auto import tqdm

logger = logging.get_logger(__name__)


def dataset_collate(batch, tokenizer):
    input_sent = [i['input_sent'] for i in batch]
    target_sent = [i['target_sent'] for i in batch]
    inputs = tokenizer(input_sent, return_tensors="pt", padding='max_length', truncation=True)
    labels = tokenizer(target_sent, return_tensors="pt", truncation=True,
                       padding='max_length', add_special_tokens=False)['input_ids']
    inputs['labels'] = labels
    return inputs


class DataModule(pl.LightningDataModule):
    def __init__(self, tokenizer_config, train_datalist, val_datalist, batch_size):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config)
        self.train_datalist = train_datalist
        self.val_datalist = val_datalist
        self.batch_size = batch_size

    def train_dataloader(self):
        loader = data.DataLoader(Dataset(self.train_datalist, self.tokenizer), batch_size=self.batch_size, shuffle=True,
                                 collate_fn=partial(dataset_collate, tokenizer=self.tokenizer))
        return loader

    def val_dataloader(self):
        loader = data.DataLoader(Dataset(self.val_datalist, self.tokenizer), batch_size=self.batch_size,
                                 shuffle=False,
                                 collate_fn=partial(dataset_collate, tokenizer=self.tokenizer))
        return loader


class Dataset(data.Dataset):
    def __init__(self, pool, tokenizer):
        self.input_pool = [i for i in tqdm(pool) if len(tokenizer.encode(i['input_sent'])) < 1024]

    def __getitem__(self, index):
        return self.input_pool[index]

    def __len__(self):
        return len(self.input_pool)


# Denoising Pre-Training
class BARTDPTModel(pl.LightningModule):

    def __init__(self, model_config=None, tokenizer_config=None, lr=3e-4, batch_size=10):
        super().__init__()
        if model_config is not None:
            # https://huggingface.co/transformers/model_doc/bart.html#transformers.BartForConditionalGeneration
            self.bart = BartForConditionalGeneration.from_pretrained(model_config)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config)
            self.bart.resize_token_embeddings(self.tokenizer.vocab_size)
            self.config = model_config
            self.lr = lr
            self.batch_size = batch_size

    def forward(self, input_text):
        outputs = self.bart(
            **self.tokenizer(input_text, return_tensors='pt')
        )
        return outputs

    def training_step(self, batch, batch_idx):
        loss = self.bart(
            **batch
        )[0]
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log('dev_loss', loss, prog_bar=True, on_step=False, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=(self.lr or self.learning_rate))
