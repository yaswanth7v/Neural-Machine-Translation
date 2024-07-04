import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import re
import numpy as np
import string

from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
import random

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from config import get_config
from BilingualDataset import BilingualDataset
from model import *


def train_model(model, src_tokenizer, tgt_tokenizer, train_dataloader, config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Assuming you are using a GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=config["eps"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=src_tokenizer.token_to_id('[PAD]'), label_smoothing=config["label_smoothing"]).to(device)
    num_epochs = config["num_epochs"]

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
        
            encoder_input = batch["encoder_input"].to(device)  # (B, seq_len)
            decoder_input = batch["decoder_input"].to(device)  # (B, seq_len)
            target = batch["label"].to(device)  # (B, seq_len)

            # Create masks
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            optimizer.zero_grad()

            # Run the tensors through the encoder, decoder, and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)  # (B, seq_len, d_model)
            proj_output = model.projection_layer(decoder_output)  # (B, seq_len, vocab_size)
            
            label = batch['label'].to(device)
            loss = loss_fn(proj_output.view(-1, tgt_tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()

        model_save_path = "model_fra-eng_1.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}")

    return model


def load_preprocess():

    data_path = 'fra.txt'
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    sents = text.strip().split('\n')
    sents = [i.split('\t') for i in sents]
    
    fra_eng = [[s.replace('\u202f', '') for s in sample] for sample in sents]
    fra_eng = np.array(fra_eng)
    fra_eng = fra_eng[:,[0,1]]

    fra_eng[:,0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in fra_eng[:,0]]
    fra_eng[:,1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in fra_eng[:,1]]
    
    for i in range(len(fra_eng)):
        fra_eng[i,0] = fra_eng[i,0].lower()
        fra_eng[i,1] = fra_eng[i,1].lower()
        fra_eng

    fra_eng[:,0] = [re.sub(r'[^\w\s]', '', s) for s in fra_eng[:,0]]
    fra_eng[:,1] = [re.sub(r'[^\w\s]', '', s) for s in fra_eng[:,1]]
    
    return fra_eng

def build_tokenizer(ds):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
    tokenizer.train_from_iterator(ds, trainer=trainer)
    return tokenizer

def get_tokenizers():
    return 

def main():

    dataset = load_preprocess()
    # Build tokenizers
    src_tokenizer = build_tokenizer(dataset[:, 0])
    tgt_tokenizer = build_tokenizer(dataset[:, 1])

    src_tokenizer.save("tokenizer_eng.json")
    tgt_tokenizer.save("tokenizer_fr.json")

    src_vocab_size = src_tokenizer.get_vocab_size()
    tgt_vocab_size = tgt_tokenizer.get_vocab_size()

    train_ds_size = int(0.9 * len(dataset))
    val_ds_size = len(dataset) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(dataset, [train_ds_size, val_ds_size])

    config = get_config()
    train_ds = BilingualDataset(train_ds_raw, src_tokenizer, tgt_tokenizer, config['src_lang'], config['tgt_lang'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, src_tokenizer, tgt_tokenizer, config['src_lang'], config['tgt_lang'], config['seq_len'])

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
   
    if len(val_ds_raw) > 0:
        val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    else:
        print("Validation dataset is empty. Check your dataset splitting logic.")

    model = build_transformer(src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size(), config['seq_len'], config['seq_len'], config['d_model'], config['N'], config['h'], config['dropout'], config['d_ff'])
    model = train_model(model, src_tokenizer, tgt_tokenizer, train_dataloader, config)
    

if __name__=="__main__":
    main()
    