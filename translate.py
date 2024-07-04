import torch
from tokenizers import Tokenizer
from config import get_config
from model import build_transformer

# Load tokenizers and configuration
src_tokenizer = Tokenizer.from_file("tokenizer_eng.json")
tgt_tokenizer = Tokenizer.from_file("tokenizer_fr.json")
config = get_config()

def translate(sentence: str, model, tokenizer_src, tokenizer_tgt, seq_len, device):
    model.eval()
    with torch.no_grad():
        # Encode the source sentence
        source_ids = tokenizer_src.encode(sentence).ids
        
        # Handle source sentence longer than seq_len - 2 (for [SOS] and [EOS])
        if len(source_ids) > seq_len - 2:
            source_ids = source_ids[:seq_len - 2]
        
        source_tensor = torch.tensor(
            [tokenizer_src.token_to_id('[SOS]')] + source_ids + [tokenizer_src.token_to_id('[EOS]')] +
            [tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source_ids) - 2),
            dtype=torch.int64
        ).unsqueeze(0).to(device)

        # Create the source mask
        source_mask = (source_tensor != tokenizer_src.token_to_id('[PAD]')).unsqueeze(1).unsqueeze(2).to(device)
        
        # Compute encoder output
        encoder_output = model.encode(source_tensor, source_mask)
        
        # Initialize the decoder input with the SOS token
        decoder_input = torch.tensor([[tokenizer_tgt.token_to_id('[SOS]')]], dtype=torch.int64).to(device)

        while decoder_input.size(1) < seq_len:
            # Create the decoder mask
            decoder_mask = torch.triu(torch.ones((decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.bool).to(device)
            
            # Decode
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
            
            # Project to the target vocabulary
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            
            # Append the next word to the decoder input
            decoder_input = torch.cat([decoder_input, next_word.unsqueeze(0)], dim=1)
            
            # Print the translated word
            print(tokenizer_tgt.decode([next_word.item()]), end=' ')
            
            # Break if the EOS token is predicted
            if next_word.item() == tokenizer_tgt.token_to_id('[EOS]'):
                break
        
        print()  # for newline

    # Convert IDs to tokens, skipping the initial [SOS] token
    return tokenizer_tgt.decode(decoder_input.squeeze().tolist()[1:])

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build the model
model = build_transformer(src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size(), config["seq_len"], config["seq_len"], config["d_model"], config["N"], config["h"], config["dropout"], config["d_ff"])

# Load the state dictionary
model_path = "model_fra-eng_1.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

n = 1
while n:
    print('1. Translate\n2. Stop\n')
    n = int(input("Enter choice: "))
    if n == 1:
        sentence = input('Enter sentence in English: ')
        translated_sentence = translate(sentence, model, src_tokenizer, tgt_tokenizer, config["seq_len"], device)
        print(f'French: {translated_sentence}')
    else:
        break
