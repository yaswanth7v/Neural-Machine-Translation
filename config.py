def get_config():
    return {
        "src_lang" : 'eng',
        "tgt_lang" : 'fr',
        "batch_size" : 8,
        "num_epochs" : 5,
        "lr" : 10**-4,
        "eps" : 1e-9,
        "label_smoothing" : 0.1,
        "seq_len" : 128,
        "d_model" : 512,
        "N" : 6, 
        "h" : 8, 
        "dropout" : 0.1,
        "d_ff" : 2048  
    }