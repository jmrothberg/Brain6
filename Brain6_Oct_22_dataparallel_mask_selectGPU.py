#JMR LLM transformers from scratch Character version April 4 2024 WORKING!
#JMR LLM changes to tokens and it learns 100s of times faster than character version
#JMR LLM May 2nd made it work with multiple GPUs using DataParallel, had failed earlier.
#JMR LLM Sept 21st added a custom dataset for chess. And new optimizer AdamW
#JMR LLM Sept 24th added a custom dataset tokenizing each move as one token.
#JMR LLM Sept 25th updated to save optimizer and scheduler and scaler
#JMR LLM Sept 26th added gradient clipping    
#JMR LLM Sept 28th save and load embeddings seperate from model, refactored code to put in functions, made class dataset parallel
#JMR LLM Oct 1st made to run on mac or linux and allow reloading of models
#JMR LLM Oct 3rd added max_norm to settings, also fixed DataParallel bug whem you wrap it twice!
#JMR LLM Oct 7th added game mask to attention mechanism
#JMR LLM Oct 8th  scaler.unscale_(optimizer) #this stopped the gradient explosion 
#JMR LLM Oct 10th first time continuing training with same learning rates and then same loss as when I started, and no norm errors
#JMR LLM Oct 12th added ability to change dropout
import os
import torch
import torch.nn as nn
import torch.optim as optim
import requests
import math
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import tkinter as tk
from tkinter import filedialog
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

# Determine the device
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using MPS")
else:
    device = torch.device('cpu')
    print("Using CPU")

# At the beginning of your script, after device selection:
if device.type == 'mps':
    torch.set_default_dtype(torch.float32)

# Define special tokens for game start and end
special_tokens = ['<STARTGAME>', '<EOFG>']


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        x = x + self.sa(self.ln1(x), mask=mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.n_head = n_head
        self.head_size = head_size
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, nh, T, T)
        if mask is not None:
            att = att.masked_fill(mask[:, :T, :T].unsqueeze(1) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)  # (B, nh, T, T)
        att = self.dropout(att)
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.proj(y)  # (B, T, C)
        return y

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, block_size, n_layer, dropout, pretrained_embeddings=None):
        super().__init__()
        dtype = torch.get_default_dtype() # Added to run on mac 
        if pretrained_embeddings is not None:
            self.token_embedding_table = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
            vocab_size, n_embd = pretrained_embeddings.shape
        else:
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd, dtype=dtype)
        self.position_embedding_table = nn.Embedding(block_size, n_embd, dtype=dtype)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size
        self.start_game_token = move_to_idx['<STARTGAME>']

    def create_game_mask(self, idx):
        # Create a mask that's 1 for tokens in the same game, 0 otherwise
        mask = torch.ones_like(idx, dtype=torch.float32)
        game_boundaries = (idx == self.start_game_token).float().cumsum(dim=1)
        mask = (game_boundaries.unsqueeze(1) == game_boundaries.unsqueeze(2)).float()
        return mask

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if idx.device.type == 'mps':
            idx = idx.to(torch.int64)  # Convert to int64 (long) only on MPS
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        game_mask = self.create_game_mask(idx)
        
        for block in self.blocks:
            x = block(x, mask=game_mask)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    

class GPT2TokenDataset(Dataset):
    def __init__(self, text, seq_length, tokenizer):
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.tokens = []
        tokenized_chunks = []
        chunk_size = 1024 # Adjust this value based on the tokenizer's limit
        text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        for chunk in text_chunks:
            tokens = tokenizer.encode(chunk) #GPT2 tokenizer
            tokenized_chunks.append(tokens)
        tokens = [token for chunk in tokenized_chunks for token in chunk]
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens) - self.seq_length

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx+self.seq_length])
        y = torch.tensor(self.tokens[idx+1:idx+self.seq_length+1])
        return x, y
        

class ChessChrDataset(Dataset):
    def __init__(self, text, seq_length, char_to_idx):
        #self.text = text
        self.seq_length = seq_length
        self.chars = []
        i = 0
        while i < len(text):
            if text[i:i+11] == '<STARTGAME>':
                self.chars.append(char_to_idx['<STARTGAME>'])
                i += 11
            elif text[i:i+6] == '<EOFG>':
                self.chars.append(char_to_idx['<EOFG>'])
                i += 6
            else:
                self.chars.append(char_to_idx[text[i]])
                i += 1
        self.current_idx = 0  # Initialize current_idx here

    def __len__(self):
        return len(self.chars) - self.seq_length

    def __getitem__(self, idx):
        x = torch.tensor(self.chars[idx:idx+self.seq_length])
        y = torch.tensor(self.chars[idx+1:idx+self.seq_length+1])
        return x, y


class ChessMovesDataset(Dataset):
    def __init__(self, text, seq_length, move_to_idx):
        self.seq_length = seq_length
        #self.move_to_idx = move_to_idx
        self.tokens = []
        
        i = 0
        while i < len(text):
            if text[i:i+11] == '<STARTGAME>':
                self.tokens.append(move_to_idx['<STARTGAME>'])
                i += 11
            elif text[i:i+6] == '<EOFG>':
                self.tokens.append(move_to_idx['<EOFG>'])
                i += 6
            elif text[i].isspace():
                # Skip spaces
                i += 1
            elif i + 4 <= len(text):
                move = text[i:i+4].upper()
                if move in move_to_idx:
                    self.tokens.append(move_to_idx[move])
                    i += 4
                else:
                    # Skip invalid characters
                    i += 1
            else:
                # Skip any remaining characters at the end
                i += 1

    def __len__(self):
        return len(self.tokens) - self.seq_length

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx+self.seq_length])
        y = torch.tensor(self.tokens[idx+1:idx+self.seq_length+1])
        return x, y


def create_char_to_idx():
 # Define the chess characters and special tokens
    chess_chars = "abcdefghKQRBNPkqrnp12345678=+#-O/x \n"
    special_tokens = ['<STARTGAME>', '<EOFG>']

    # Create char_to_idx dictionary
    char_to_idx = {char: idx for idx, char in enumerate(chess_chars)}

    # Add special tokens to char_to_idx
    for idx, token in enumerate(special_tokens, start=len(char_to_idx)):
        char_to_idx[token] = idx
    return char_to_idx


def create_idx_to_char():
    # Create idx_to_char dictionary
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return idx_to_char


def create_move_to_idx():
    # Create the basic move-to-index mapping
    move_to_idx = {f"{chr(97 + i % 8)}{8 - i // 8}{chr(97 + j % 8)}{8 - j // 8}".upper(): i * 63 + j for i in range(64) for j in range(64) if i != j}
    
    # Define special tokens
    special_tokens = ['<STARTGAME>', '<EOFG>', '<PAD>']

    # Add special tokens to move_to_idx
    for idx, token in enumerate(special_tokens, start=len(move_to_idx)):
        move_to_idx[token] = idx
    return move_to_idx


def create_idx_to_move():
    idx_to_move = {idx: move for move, idx in move_to_idx.items()}
    return idx_to_move


def get_input_with_default(prompt, default_value):
    value = input(f"{prompt} (default: {default_value}): ")
    return value if value else default_value


def load_text_file():
    text = ""
    print("Please select a text file.")  
    text_file = filedialog.askopenfilename()
    
    if text_file:
        try:
            with open(text_file, 'r', encoding='utf-8') as file:
                text = file.read()
            print(f"Text file selected: {text_file}")
            print(f"Loading text: {len(text)} characters") 
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Loading the Shakespeare dataset.")
            text = load_shakespeare_text()

    else:
        print("No text file selected. Loading the Shakespeare dataset.")
        text = load_shakespeare_text()
    return text


def load_shakespeare_text():
    # Create a directory to store the Shakespeare dataset
    if not os.path.exists('shakespeare'):
        os.makedirs('shakespeare')
    # Check if the Shakespeare dataset is already downloaded
    if not os.path.exists('shakespeare/shakespeare.txt'):
        # Download the Shakespeare dataset
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        response = requests.get(url)
        with open('shakespeare/shakespeare.txt', 'w', encoding='utf-8') as file:
            file.write(response.text)
    else:
        print("Shakespeare dataset already downloaded.")    
        # Read the Shakespeare dataset
        with open('shakespeare/shakespeare.txt', 'r', encoding='utf-8') as file:
            text = file.read()
            print(f"Loading Shakespeare text: {len(text)} characters")
    return text


def save_token_embeddings(model, filepath):
    if isinstance(model, nn.DataParallel):
        embeddings = model.module.token_embedding_table.weight.data
    else:
        embeddings = model.token_embedding_table.weight.data
    torch.save(embeddings, filepath)
    filename = os.path.basename(filepath)
    model_folder = os.path.dirname(filepath)
    print(f"Token embedding saved: {filename} in {model_folder}")


def load_token_embeddings(filepath):
    embeddings = torch.load(filepath)
    print(f"Token embeddings loaded from {filepath}")
    print(f"Loaded embedding shape: {embeddings.shape}")
    return embeddings


def create_file_dialog(title="Select File", filetypes=None):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return file_path


def save_model_all(model, all_text, n_embd, n_head, n_layer, dropout, block_size, epoch, batch_idx, optimizer, scheduler, scaler,loss):
    # Save the model and generated text
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define a base directory for your models
    BASE_DIR = "/data"  # Change this to your desired base directory

    # Create the model folder path
    model_folder = os.path.join(BASE_DIR, f"models_{n_embd}_{n_head}_{n_layer}_{dropout}_{block_size}_masked")

    model_filename = f"Epoch_{epoch+1}_Batch_{batch_idx+1}_Loss_{loss:.4f}_{timestamp}_masked.pth"

    # Check if there is a directory called model_folder
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Prepare the checkpoint
    if isinstance(model, nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    checkpoint = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'epoch': epoch,
        'batch_idx': batch_idx,
        'hyperparameters': {
            'vocab_size': vocab_size,
            'n_embd': n_embd,
            'n_head': n_head,
            'n_layer': n_layer,
            'dropout': dropout,
            'block_size': block_size,
        },
    }

    # Add the appropriate tokenizer based on the dataset type
    if move_to_token:
        checkpoint['tokenizer'] = move_to_idx
        checkpoint['dataset_type'] = 'chess_moves'
    elif use_chess:
        checkpoint['tokenizer'] = char_to_idx
        checkpoint['dataset_type'] = 'chess_characters'
    else:
        checkpoint['tokenizer'] = None  # For GPT-2 tokenizer
        checkpoint['dataset_type'] = 'text'

    # Save the model in the model folder
    print(f"\nModel folder: {model_folder}, {checkpoint['hyperparameters']}")
    torch.save(checkpoint, os.path.join(model_folder, model_filename))
    print(f"Model saved: {model_filename} in {model_folder}")

    # Save the generated text
    filenamealltext = f"all_text_{epoch+1}_{timestamp}.txt"
    with open(os.path.join(model_folder, filenamealltext), 'w', encoding='utf-8') as file:
        file.write(all_text)
    print(f"Text saved: {filenamealltext} in {model_folder}")

    # Save token embeddings separately
    embedding_filename = f"token_embeddings_{timestamp}.pt"
    save_token_embeddings(model, os.path.join(model_folder, embedding_filename))


def load_model_file(vocab_size, n_embd, n_head, block_size, n_layer, dropout):
    print("Please select a model file.")
    model_file = create_file_dialog(title="Select Model File", filetypes=[("PyTorch files", "*.pth")])
    if model_file:
        if device.type == 'mps':
            checkpoint = torch.load(model_file, map_location='cpu')  # Load to CPU first for MPS
        else:
            checkpoint = torch.load(model_file)  # Original loading for other devices
        
        hyperparameters = checkpoint['hyperparameters']
        vocab_size = hyperparameters['vocab_size']
        n_embd = hyperparameters['n_embd']
        n_head = hyperparameters['n_head']
        n_layer = hyperparameters['n_layer']
        dropout = hyperparameters['dropout']
        block_size = hyperparameters['block_size']

        # Get the dataset type and tokenizer
        dataset_type = checkpoint.get('dataset_type', 'text')  # Default to 'text' for backward compatibility
        tokenizer = checkpoint.get('tokenizer')

        print(f"Model file loaded: {model_file}, {hyperparameters}")
        print(f"Dataset type: {dataset_type}")

        # Create the model with the correct flag for pretrained embeddings
        model = TransformerModel(vocab_size, n_embd, n_head, block_size, n_layer, dropout)
        
        # Move the model to the device before loading the state dict
        model = model.to(device)
        
        # Load state dict
        state_dict = checkpoint['model_state_dict']
        
        # If using CUDA and multiple GPUs are available, wrap the model in DataParallel
        if torch.cuda.device_count() > 1 and device.type == 'cuda':
            model = nn.DataParallel(model)
            if not all(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {'module.' + k: v for k, v in state_dict.items()}
            print(f"Model wrapped in DataParallel. Using {torch.cuda.device_count()} GPUs")   
        else:
            if all(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                print("Not a DataParallel model, removed 'module.' prefix")

        model.load_state_dict(state_dict)
        print("Model state dict loaded successfully")

        # Check for optimizer, scheduler, and scaler states
        optimizer_state_dict = checkpoint.get('optimizer_state_dict')
        scheduler_state_dict = checkpoint.get('scheduler_state_dict')
        scaler_state_dict = checkpoint.get('scaler_state_dict')

        last_epoch = checkpoint.get('epoch', -1)
        last_batch_idx = checkpoint.get('batch_idx', -1)
        
        if optimizer_state_dict is None or len(optimizer_state_dict) == 0:
            print("No optimizer state found in checkpoint. Will initialize a new optimizer.")
        if scheduler_state_dict is None or len(scheduler_state_dict) == 0:
            print("No scheduler state found in checkpoint. Will initialize a new scheduler.")
        if scaler_state_dict is None or len(scaler_state_dict) == 0:
            print("No scaler state found in checkpoint. Will initialize a new scaler.")

        print(f"Model file loaded: {model_file}")
        print(f"Model hyperparameters: vocab_size={vocab_size}, n_embd={n_embd}, n_head={n_head}, block_size={block_size}, n_layer={n_layer}, dropout={dropout}")

        # Determine the dataset configuration
        use_chess = dataset_type in ['chess_moves', 'chess_characters']
        move_to_token = dataset_type == 'chess_moves'

        # Return additional optimizer hyperparameters
        return model, vocab_size, n_embd, n_head, block_size, n_layer, dropout, tokenizer, use_chess, move_to_token, dataset_type, optimizer_state_dict, scheduler_state_dict, scaler_state_dict, last_epoch, last_batch_idx
    else:
        print("No model file selected. Creating a new model...")
        return None


def specify_model_parameters(vocab_size, n_embd=128, n_head=8, block_size=256, n_layer=12, dropout=0.2, use_gpt2=False):
    pretrained_embeddings = None
    
    if use_gpt2:
        print("Using pre-built GPT-2 embeddings")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        pretrained_embeddings = tokenizer.get_input_embeddings().weight.clone().detach()
        n_embd = pretrained_embeddings.shape[1]
        vocab_size = pretrained_embeddings.shape[0]
        print(f"Using GPT-2 embeddings with n_embd={n_embd} and vocab_size={vocab_size}")
    else:
        use_pretrained_embeddings = input("Use custom pretrained token embeddings? (y/n): ").lower() == 'y'
        if use_pretrained_embeddings:
            embedding_file = filedialog.askopenfilename(title="Select Token Embedding File", filetypes=[("PyTorch files", "*.pt")])
            if embedding_file:
                pretrained_embeddings = load_token_embeddings(embedding_file)
                loaded_vocab_size, loaded_n_embd = pretrained_embeddings.shape
                if loaded_vocab_size != vocab_size:
                    print(f"Warning: Loaded embeddings vocab size {loaded_vocab_size} doesn't match tokenizer vocab size {vocab_size}.")
                    use_pretrained_embeddings = False
                else:
                    n_embd = loaded_n_embd
                    print(f"Using loaded embeddings with n_embd={n_embd}")

        if not use_pretrained_embeddings:
            n_head = int(get_input_with_default("Enter number of heads (n_head)", n_head))
            
            print(f"For n_head={n_head}, n_embd must be a multiple of {n_head}.")
            valid_embds = [i * n_head for i in range(max(16, n_head), 129, n_head)]
            valid_embds = [e for e in valid_embds if e >= 256]  # Ensure we start from at least 256
            print(f"Some valid options for n_embd are: {', '.join(str(e) for e in valid_embds[:10])}")
            
            while True:
                n_embd = int(get_input_with_default("Enter embedding dimensions (n_embd)", n_embd))
                if n_embd % n_head == 0:
                    break
                else:
                    print(f"Error: n_embd ({n_embd}) must be divisible by n_head ({n_head}). Please try again.")
    
    block_size = int(get_input_with_default("Enter seq_length/block_size (recommended 32-64 for chess)", block_size))
    n_layer = int(get_input_with_default("Enter n_layer", n_layer))
    dropout = float(get_input_with_default("Enter dropout", dropout))

    print(f"Final parameters: n_embd={n_embd}, n_head={n_head}, n_embd/n_head={n_embd//n_head}")

    model = TransformerModel(vocab_size, n_embd, n_head, block_size, n_layer, dropout, pretrained_embeddings)
    
    print("New model created.")
    print(f"Model hyperparameters: vocab_size={vocab_size}, n_embd={n_embd}, n_head={n_head}, block_size={block_size}, n_layer={n_layer}, dropout={dropout}")
    return model, vocab_size, n_embd, n_head, block_size, n_layer, dropout, use_pretrained_embeddings


def enter_batch_size(n_embd, n_head, block_size, n_layer, vocab_size, batch_size):
    # Constants
    bytes_per_float = 4  # Assuming we are using float32
    safety_factor = 0.9  # To avoid maxing out GPU memory

    # Adjust memory calculations based on device
    if device.type == 'cuda':
        gpu_memory = sum(torch.cuda.get_device_properties(i).total_memory for i in gpu_indices)
    elif device.type == 'mps':
        # For MPS, we don't have a direct way to get GPU memory, so we'll use a conservative estimate
        gpu_memory = 192 * 1024**3  # Assume 192GB as a conservative estimate
    else:
        gpu_memory = 64 * 1024**3  # Assume 64GB as a conservative estimate for CPU 
    
    # Memory calculations for model parameters
    token_embeddings = vocab_size * n_embd * bytes_per_float
    position_embeddings = block_size * n_embd * bytes_per_float
    attention_weights = 4 * n_layer * n_head * n_embd * n_embd * bytes_per_float
    feedforward_weights = 2 * n_layer * 4 * n_embd * n_embd * bytes_per_float
    layer_norms = 2 * n_layer * 2 * n_embd * bytes_per_float
    final_layer_norm = 2 * n_embd * bytes_per_float
    output_weights = n_embd * vocab_size * bytes_per_float

    total_model_params = (token_embeddings + position_embeddings + attention_weights + 
                          feedforward_weights + layer_norms + final_layer_norm + output_weights)

    # Account for optimizer memory (assuming Adam-like optimizer)
    optimizer_memory = total_model_params * 2  # For momentum and variance

    # Adjust gradient memory calculation
    gradient_memory = total_model_params

    # Memory required for activations and gradients per sequence in a batch
    activations_per_seq = block_size * n_embd * bytes_per_float * (4 * n_layer + 2)  # Increased factor for intermediate activations
    attention_activations = block_size * block_size * n_head * n_layer * bytes_per_float

    total_per_seq = activations_per_seq + attention_activations

    # Calculate maximum batch size
    available_memory = gpu_memory * safety_factor - (total_model_params + optimizer_memory + gradient_memory)
    max_batch_size = int(available_memory / total_per_seq)

    print(f"Estimated model parameters memory: {total_model_params / 1e9:.2f} GB")
    print(f"Estimated memory per sequence: {total_per_seq / 1e6:.2f} MB")
    print(f"Maximum feasible batch size within {gpu_memory / 1e9:.0f}GB GPU memory: {max_batch_size}")
    
    batch_size = int(get_input_with_default(f"Enter batch size (up to {max_batch_size}): ", batch_size))
    batch_size = min(batch_size, max_batch_size)  # Ensure batch size does not exceed maximum

    return batch_size


def test_progress(epoch, num_epochs, batch_idx, data_loader, loss, model, x, tokens_to_generate, all_text, tokenizer_or_idx_to_char_or_move):
    # This function tests the model's progress by generating text based on an input sequence
    
    #print("Test progress")
    print(f"\nEpoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item():.4f}")
    
    # Set the model to evaluation mode (disables dropout, etc.)
    model.eval()
    
    # Disable gradient calculation for efficiency during inference
    with torch.no_grad():
        # Take the last sequence from the batch as input
        input_seq = x[-1].unsqueeze(0)  # Add a batch dimension
        
        # Convert input sequence to readable format
        if isinstance(tokenizer_or_idx_to_char_or_move, dict):  # For dictionary-based tokenizer
            if len(tokenizer_or_idx_to_char_or_move) > 256:  # Assuming move-based tokenization has more than 256 entries
                input_seq_str = ' '.join([tokenizer_or_idx_to_char_or_move[idx.item()] for idx in input_seq[0]])
            else:  # Character-based tokenization
                input_seq_str = ''.join([tokenizer_or_idx_to_char_or_move[idx.item()] for idx in input_seq[0]])
        else:  # For GPT-2 tokenizer
            input_seq_str = tokenizer_or_idx_to_char_or_move.decode(input_seq[0])
        
        print("\nInput Sequence:")
        print(input_seq_str)
        
        generated_tokens = []
        num_tokens_to_generate = tokens_to_generate
        
        # Generate new tokens one by one
        for _ in range(num_tokens_to_generate):
            # Get model output for the current input sequence
            output, _ = model(input_seq)
            
            # Select the most likely next token
            pred_index = output[0, -1].argmax(dim=-1)
            
            # Add the predicted token to the list of generated tokens
            generated_tokens.append(pred_index.item())
            
            # Update the input sequence for the next iteration
            # Remove the first token and add the new predicted token
            input_seq = torch.cat((input_seq[:, 1:], pred_index.unsqueeze(0).unsqueeze(0)), dim=1)
        
        # Convert generated tokens to readable text
        if isinstance(tokenizer_or_idx_to_char_or_move, dict):  # For character-based tokenizer
            if len(tokenizer_or_idx_to_char_or_move) > 256:  # Assuming move-based tokenization has more than 256 entries   
                generated_text = ' '.join([tokenizer_or_idx_to_char_or_move[idx] for idx in generated_tokens])
            else:  # Character-based tokenization
                generated_text = ''.join([tokenizer_or_idx_to_char_or_move[idx] for idx in generated_tokens])
        else:  # For GPT-2 tokenizer
            generated_text = tokenizer_or_idx_to_char_or_move.decode(generated_tokens)
       
        print("\nGenerated Text:")
        print(generated_text)
       
        # Append the input and generated text to the overall text log
        all_text = all_text + ("\nInput Sequence:\n" +  input_seq_str + "\nGenerated Text:\n" + generated_text)
    
    # Set the model back to training mode
    model.train()
    
    return all_text

# Potential improvements:
# 1. Use a separate function for tokenization/detokenization to reduce code duplication
# 2. Consider using top-k or nucleus sampling for more diverse text generation
# 3. Add error handling for cases where the model or tokenizer might fail
# 4. Optionally limit the length of the input_seq_str and generated_text for very long sequences
# 5. Use tqdm for a progress bar during token generation
# 6. Add an option to save generated text to a file directly

def select_gpus():
    if torch.cuda.device_count() <= 1:
        print(f"Only {torch.cuda.device_count()} GPU available. Using it for computation.")
        return list(range(torch.cuda.device_count()))

    print(f"Available GPUs: {torch.cuda.device_count()}")
    while True:
        custom_gpus = input("Enter 2 to 4 GPU indices separated by commas (e.g., 0,1 or 0,1,2,3): ")
        gpu_indices = [int(idx.strip()) for idx in custom_gpus.split(',')]
        if 2 <= len(gpu_indices) <= 4 and all(0 <= idx < torch.cuda.device_count() for idx in gpu_indices):
            return gpu_indices
        else:
            print(f"Invalid input. Please specify 2 to 4 valid GPU indices (0 to {torch.cuda.device_count() - 1}).")

#This is main so you can also call the other functions
if __name__ == "__main__":

    text = ""
    model = None

    #Set the model parameters defaults
    n_embd = 128
    n_head = 8
    block_size = 256 # Sequence length
    n_layer = 12
    dropout = 0.2
    num_epochs = 15
    batch_size = 256

    learning_rate = 0.001
    weight_decay = 0.01
    embedding_learning_rate = 0.0005 # Learning rate for the embedding layer should be how much lower than the other layers
    embedding_weight_decay = 0.005

    max_norm = 1.0
    
    start_epoch = 0
    start_batch = 0

    use_pretrained_embeddings = False
    optimizer_state_dict = None
    scheduler_state_dict = None
    scaler_state_dict = None

    # Ask user to choose between text file, chess games, or whole chess games
    data_choice = input("Choose data source (1: Text file GPT2, 2: Chess games characters, 3: Chess games moves): ")

    if data_choice == "1":
        
        text = load_text_file()
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        vocab_size = tokenizer.vocab_size

        print("Vocab size:", vocab_size)
        print("Text dataset loaded. Total characters:", len(text))

        use_gpt2 = True
        use_chess = False
        move_to_token = False

    elif data_choice == "2":
        file_path = create_file_dialog(title="Select Chess Games File", filetypes=[("Text files", "*.txt")])

        if not file_path:
            print("No file selected. Exiting.")
            exit()

        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        games = text.split('\n\n')
        
        games = ['<STARTGAME>' +' ' + game.strip() + ' ' + '<EOFG>' for game in games if game.strip()]
        text = ' '.join(games)
        print(f"Chess dataset loaded. Total games: {len(games)}, Total characters: {len(text)}")

        char_to_idx = create_char_to_idx()
        idx_to_char = create_idx_to_char()
        
        vocab_size = len(char_to_idx)

        #print("Char to idx:", char_to_idx)
        print("Vocab size:", vocab_size)
        #print("Idx to char:", idx_to_char)
        
        use_gpt2 = False
        use_chess = True
        move_to_token = False

    elif data_choice == "3":
        file_path = create_file_dialog(title="Select Chess Games File", filetypes=[("Text files", "*.txt")])
        
        if not file_path:
            print("No file selected. Exiting.")
            exit()

        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        games = text.split('\n\n')

        games = ['<STARTGAME>' +' ' + game.strip() + ' ' + '<EOFG>' for game in games if game.strip()]
        text = '\n'.join(games)
        print(f"Chess dataset loaded. Total games: {len(games)}, Total characters: {len(text)}")

        move_to_idx = create_move_to_idx()
        idx_to_move = create_idx_to_move()

        vocab_size = len(move_to_idx)

        #print("Move to idx:", move_to_idx)
        print("Vocab size:", vocab_size)
        #print("Idx to move:", idx_to_move)  
       
        use_gpt2 = False
        use_chess = True
        move_to_token = True
    else:
        print("Invalid choice. Exiting.")
        exit()

    try:
        load_or_create = input("Load a model file or Create a new model? (l/c): ").lower()
        if load_or_create == 'l':
            loaded_model = load_model_file(vocab_size, n_embd, n_head, block_size, n_layer, dropout)
            if loaded_model:
                model, vocab_size, n_embd, n_head, block_size, n_layer, dropout, tokenizer, use_chess, move_to_token, dataset_type, optimizer_state_dict, scheduler_state_dict, scaler_state_dict, last_epoch, last_batch_idx = loaded_model
                
                start_epoch = last_epoch # + 1
                start_batch = last_batch_idx + 1

                print(f"Model loaded. Resuming from epoch {start_epoch}, batch {start_batch}")

                # Handle dropout change
                new_dropout = float(get_input_with_default(f"Enter new dropout value (current: {dropout})", dropout))
                
                if new_dropout != dropout:
                    dropout = new_dropout
                    
                    def update_dropout(module):
                        if isinstance(module, nn.Dropout):
                            module.p = dropout
                    
                    if isinstance(model, nn.DataParallel):
                        model.module.apply(update_dropout)
                    else:
                        model.apply(update_dropout)
                    
                    print(f"Model dropout updated to: {dropout}")
                else:
                    print(f"Keeping current dropout value: {dropout}")

            else:
                print("Failed to load model. Creating a new one.")
                raise ValueError("Model loading failed")
        else:
            raise ValueError("User chose to create a new model")


    except Exception as e:
        print(f"An error occurred or user chose to create a new model: {e}")
        print("Creating a new model...")
        model, vocab_size, n_embd, n_head, block_size, n_layer, dropout, use_pretrained_embeddings = specify_model_parameters(
            vocab_size=vocab_size, 
            n_embd=n_embd, 
            n_head=n_head, 
            block_size=block_size, 
            n_layer=n_layer, 
            dropout=dropout,
            use_gpt2=use_gpt2
        )
        
    # Move the model to the device
    model = model.to(device)
    print(f"Model moved to {device}")
    print(f"Model is on device: {next(model.parameters()).device}")
    
    ''' If using CUDA and multiple GPUs are available, wrap the model in DataParallel
    #only do this if not already wrapped in DataParallel
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        if not isinstance(model, nn.DataParallel):
            model = nn.DataParallel(model)
        print(f"Model wrapped in DataParallel. Using {torch.cuda.device_count()} GPUs")'''
   
    # After loading the model
    if torch.cuda.is_available():
        gpu_indices = select_gpus()
        if len(gpu_indices) > 0:
            print(f"Using GPUs: {gpu_indices}")
            device = torch.device('cuda')
            if isinstance(model, nn.DataParallel):
                # Update device_ids for existing DataParallel model
                model.device_ids = gpu_indices
            else:
                # Wrap in DataParallel if it's not already (shouldn't happen if always saved with DataParallel)
                model = nn.DataParallel(model, device_ids=gpu_indices)
            
            # Move model to CUDA
            model = model.to(device)
            #model = torch.compile(model)    
            print(f"Model using DataParallel on {len(gpu_indices)} GPU(s): {gpu_indices} and is compiled")
        else:
            print("No GPUs selected. Using MPS.")
            device = torch.device('mps')
            model = model.to(device)
    else:
        print("CUDA not available. Using MPS.")
        device = torch.device('mps')
        model = model.to(device)

    if optimizer_state_dict:
        print("Loading from saved optimizer state dict")
        # Extract the saved hyperparameters from the optimizer_state_dict
        optimizer_hyperparams = optimizer_state_dict['param_groups']
        
        # Check if there are separate parameter groups for embeddings
        if len(optimizer_hyperparams) > 1:
            print("Detected separate parameter groups for embeddings")
            embedding_params = optimizer_hyperparams[0]
            other_params = optimizer_hyperparams[1]
            
            embedding_learning_rate = float(get_input_with_default(f"Enter embedding learning rate (default: {embedding_params['lr']})", embedding_params['lr']))
            embedding_weight_decay = float(get_input_with_default(f"Enter embedding weight decay (default: {embedding_params['weight_decay']})", embedding_params['weight_decay']))
            
            learning_rate = float(get_input_with_default(f"Enter learning rate for other params (default: {other_params['lr']})", other_params['lr']))
            weight_decay = float(get_input_with_default(f"Enter weight decay for other params (default: {other_params['weight_decay']})", other_params['weight_decay']))
            
            if isinstance(model, nn.DataParallel):
                embedding_params = model.module.token_embedding_table.parameters()
                other_params = (p for n, p in model.module.named_parameters() if n != 'token_embedding_table.weight')
            else:
                embedding_params = model.token_embedding_table.parameters()
                other_params = (p for n, p in model.named_parameters() if n != 'token_embedding_table.weight')
            
            optimizer = optim.AdamW([
                {'params': embedding_params, 'lr': embedding_learning_rate, 'weight_decay': embedding_weight_decay},
                {'params': other_params, 'lr': learning_rate, 'weight_decay': weight_decay}
            ])
        else:
            learning_rate = float(get_input_with_default(f"Enter learning rate (default: {optimizer_hyperparams[0]['lr']})", optimizer_hyperparams[0]['lr']))
            weight_decay = float(get_input_with_default(f"Enter weight decay (default: {optimizer_hyperparams[0]['weight_decay']})", optimizer_hyperparams[0]['weight_decay']))
            
            optimizer = optim.AdamW([
                {'params': model.parameters(), 'lr': learning_rate, 'weight_decay': weight_decay}
            ])

            # Load the state dict
        optimizer.load_state_dict(optimizer_state_dict)

        # Update all four parameters in the loaded state
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[0]['lr'] = embedding_learning_rate
            optimizer.param_groups[0]['weight_decay'] = embedding_weight_decay
            optimizer.param_groups[1]['lr'] = learning_rate
            optimizer.param_groups[1]['weight_decay'] = weight_decay
        else:
            optimizer.param_groups[0]['lr'] = learning_rate
            optimizer.param_groups[0]['weight_decay'] = weight_decay
            
        #print out to verify
        print(f"Optimizer learning rate: {optimizer.param_groups[0]['lr']}")
        print(f"Optimizer weight decay: {optimizer.param_groups[0]['weight_decay']}")
        if len(optimizer.param_groups) > 1:         
            print(f"Optimizer learning rate: {optimizer.param_groups[1]['lr']}")
            print(f"Optimizer weight decay: {optimizer.param_groups[1]['weight_decay']}")   

    else:
        print("Creating a new optimizer")
        # Initialize a new optimizer with specified hyperparameters
        learning_rate = float(get_input_with_default("Enter learning rate (default: 0.001)", 0.001))
        weight_decay = float(get_input_with_default("Enter weight decay (default: 0.01)", 0.01))
        
        if use_pretrained_embeddings:
            # Adjusted code to handle DataParallel
            if isinstance(model, nn.DataParallel):
                embedding_params = model.module.token_embedding_table.parameters()
                other_params = (p for n, p in model.module.named_parameters() if n != 'token_embedding_table.weight')
            else:
                embedding_params = model.token_embedding_table.parameters()
                other_params = (p for n, p in model.named_parameters() if n != 'token_embedding_table.weight')

            embedding_learning_rate = float(get_input_with_default("Enter embedding learning rate (default: 0.0005)", 0.0005))
            embedding_weight_decay = float(get_input_with_default("Enter embedding weight decay (default: 0.005)", 0.005))

            optimizer = optim.AdamW([
                {'params': embedding_params, 'lr': embedding_learning_rate, 'weight_decay': embedding_weight_decay},
                {'params': other_params, 'lr': learning_rate, 'weight_decay': weight_decay}
            ])
        else:
            optimizer = optim.AdamW([
                {'params': model.parameters(), 'lr': learning_rate, 'weight_decay': weight_decay}
            ])

    max_norm = float(get_input_with_default("Enter max norm (default: 10.0)", 10.0))

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    if scheduler_state_dict and len(scheduler_state_dict) > 0:
        print("Loading from saved scheduler state dict")
        scheduler.load_state_dict(scheduler_state_dict)

    scaler = GradScaler()
    if scaler_state_dict and len(scaler_state_dict) > 0:
        print("Loading from saved scaler state dict")
        scaler.load_state_dict(scaler_state_dict)
   
    #Define the loss function and optimizer
    #criterion = nn.CrossEntropyLoss(ignore_index=move_to_idx['<PAD>'])
    criterion = nn.CrossEntropyLoss()
    
    batch_size = enter_batch_size(n_embd, n_head, block_size, n_layer, vocab_size, batch_size)
    start_epoch = int(get_input_with_default(f"Enter epoch (default: {start_epoch})", start_epoch))
    num_epochs = int(get_input_with_default("Enter number of epochs", num_epochs))
    tokens_to_generate = int(get_input_with_default("Enter number of tokens to generate", 128))
    inference_frequency = int(get_input_with_default("Enter inference and save frequency", 100))

    # Create the dataset and data loader
    if use_chess:
        if move_to_token:
            dataset = ChessMovesDataset(text, block_size, move_to_idx)
        else:
            dataset = ChessChrDataset(text, block_size, char_to_idx)
    else:
        dataset = GPT2TokenDataset(text, block_size, tokenizer)

    print(f"Number of tokens in the dataset: {len(dataset)}")

    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        print(f"Using {torch.cuda.device_count()} GPUs")
        num_workers = min(32, os.cpu_count())
    else:
        num_workers = 0

    # Print the number of model parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of model parameters: {num_params}")
    if isinstance(model, nn.DataParallel):
        print(f"Model's primary device: {next(model.module.parameters()).device}")
        print(f"Model distributed across devices: {model.device_ids}")
    else:
        print(f"Model is on device: {next(model.parameters()).device}")

    print(f"Using {num_workers} workers for data loading")

    model.to(device)

    # Add these debug prints
    print(f"Model type: {type(model)}")
    if isinstance(model, nn.DataParallel):
        print(f"DataParallel devices: {model.device_ids}")

    # Update the memory usage print section:
    if torch.cuda.is_available():
        for i in gpu_indices:
            print(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            print(f"GPU {i} memory reserved: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")

    # Training loop
    model.train()   
    for epoch in range(start_epoch, num_epochs):
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
        print("len(data_loader)",len(data_loader))
        print("epoch", epoch)
        all_text = ""
        for batch_idx, (x, y) in enumerate(data_loader):
           
            x, y = x.to(device), y.to(device)

            # Forward pass with mixed precision
            if device.type == 'cuda':
                with autocast('cuda'):
                    output, loss = model(x, targets=y)
                if isinstance(model, nn.DataParallel):
                    loss = loss.mean()
                # Backward pass and optimization with gradient scaling
                optimizer.zero_grad()
                scaler.scale(loss).backward()

                # Gradient clipping with logging
                #max_norm = 1.0
                scaler.unscale_(optimizer) #this stopped the gradient explosion 
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                if total_norm > max_norm:
                    print(f"Gradient clipping applied at batch {batch_idx} in epoch {epoch}. Total norm: {total_norm:.2f}. Loss: {loss:.4f}")

                scaler.step(optimizer)
                scaler.update()
            else:
                output, loss = model(x, targets=y)
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping with logging
                #max_norm = 1.0
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                if total_norm > max_norm:
                    print(f"Gradient clipping applied at batch {batch_idx} in epoch {epoch}. Total norm: {total_norm:.2f}. Loss: {loss:.4f}")

                optimizer.step()

            # Print progress and run test inference every 100 batches
            if (batch_idx + 1) % inference_frequency == 0:
                #print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item():.4f}")
                if not use_chess:
                    all_text = test_progress(epoch, num_epochs, batch_idx, data_loader, loss, model, x, tokens_to_generate, all_text, tokenizer)
                elif move_to_token:
                    all_text = test_progress(epoch, num_epochs, batch_idx, data_loader, loss, model, x, tokens_to_generate, all_text, idx_to_move)  
                else:
                    all_text = test_progress(epoch, num_epochs, batch_idx, data_loader, loss, model, x, tokens_to_generate, all_text, idx_to_char)

                # Save the model and generated text
                save_model_all(model, all_text, n_embd, n_head, n_layer, dropout, block_size, epoch, start_batch +batch_idx, optimizer, scheduler, scaler,loss)

            # Update the learning rate scheduler
        scheduler.step()
        start_batch = 0 #reset for next epoch

    # After training, save the token embeddings
    save_model_all(model, all_text, n_embd, n_head, n_layer, dropout, block_size, epoch, start_batch + batch_idx, optimizer, scheduler, scaler,loss)