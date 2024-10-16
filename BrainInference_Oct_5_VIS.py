#JMR Inference script for GPT-2 model or character-based model
#This script allows you to load a pre-trained model and generate responses
#It supports both token-based and character-based models
#Sep 19, 2024
#Sep 24, 2024 added chess moves tokenizer separate from character tokenizer
#Sept 26, 2024 made so works on mac and picks latest pth file
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import GPT2Tokenizer
import tkinter as tk
from tkinter import filedialog
import matplotlib
#matplotlib.use('MacOSX')  # Use the macOS backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

device = torch.device('cpu')
# Add these global variables at the top of the file
global_model = None
global_tokenizer = None
global_use_characters = None
global_use_chess_moves = None  # New variable for chess moves tokenizer
global_tokenizer_reverse = None

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

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        self.last_attention = att  # Store the attention matrix for visualization
        return y

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

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        self.last_attention = self.sa.last_attention  # Store the attention matrix for visualization
        x = x + self.ffwd(self.ln2(x))
        self.last_activation = x  # Store the activation for visualization
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, block_size, n_layer, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        self.intermediate_activations = []
        self.attention_weights = []  # Add this line to store attention weights
        for block in self.blocks:
            x = block(x)
            self.intermediate_activations.append(block.last_activation)
            self.attention_weights.append(block.sa.last_attention)  # Add this line to collect attention weights
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def load_model_file(vocab_size, n_embd, n_head, block_size, n_layer, dropout, frommain=False):
    use_characters = False
    use_chess_moves = False
    #check if running on a mac and if so, check for chess directory
    running_on_mac = os.name == 'posix' and os.uname().sysname == 'Darwin'
    try:
        if not frommain and running_on_mac:
            chess_directory = "/Users/jonathanrothberg/Chess"
            if not os.path.exists(chess_directory):
                print(f"Directory not found: {chess_directory}")
                return None, None, None, None, None, None, None, None, False, False

            pth_files = glob.glob(os.path.join(chess_directory, "*.pth"))
            if not pth_files:
                print(f"No .pth files found in {chess_directory}")
                return None, None, None, None, None, None, None, None, False, False

            model_file = max(pth_files, key=os.path.getctime)

        else:
            # Modified: Use file types parameter to filter for .pth files
            model_file = filedialog.askopenfilename(filetypes=[("PyTorch Model Files", "*.pth")])

        if model_file:
            checkpoint = torch.load(model_file, map_location="cpu")
            #checkpoint = torch.load(model_file)
            hyperparameters = checkpoint['hyperparameters']
            vocab_size = hyperparameters['vocab_size']
            n_embd = hyperparameters['n_embd']
            n_head = hyperparameters['n_head']
            n_layer = hyperparameters['n_layer']
            dropout = hyperparameters['dropout']
            block_size = hyperparameters['block_size']
            tokenizer = checkpoint.get('tokenizer')  # Changed: Get tokenizer from checkpoint
            dataset_type = checkpoint.get('dataset_type', 'text')  # Changed: Get dataset type, default to 'text'
            use_characters = dataset_type == 'chess_characters'  # Changed: Determine use_characters
            use_chess_moves = dataset_type == 'chess_moves'  # Changed: Determine use_chess_moves
            print(f"Model file loaded: {model_file}, {hyperparameters}")
            
            model = TransformerModel(vocab_size, n_embd, n_head, block_size, n_layer, dropout)
            
            state_dict = checkpoint['model_state_dict']
            if all(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {key[7:]: value for key, value in state_dict.items()}
            
            model.load_state_dict(state_dict)
            
            print(f"Model file loaded: {model_file}")
            print(f"Model hyperparameters: vocab_size={vocab_size}, n_embd={n_embd}, n_head={n_head}, block_size={block_size}, n_layer={n_layer}, dropout={dropout}")
            total_params = count_parameters(model)
            print(f"The model has {total_params:,} parameters")
            # Changed: Print tokenizer information based on dataset type
            if use_characters:
                print("Character-based tokenizer loaded:")
                #print(tokenizer)
                use_characters = True
            elif use_chess_moves:
                print("Chess moves tokenizer loaded:")
                #print(tokenizer)
                use_chess_moves = True
            else:
                print("Using GPT-2 tokenizer")
            
            return model, vocab_size, n_embd, n_head, block_size, n_layer, dropout, tokenizer, use_characters, use_chess_moves
        else:
            print("No model file selected. Exiting.")
            return None, None, None, None, None, None, None, None, False, False
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None, None, None, None, None, False, False


def generate_response(model, tokenizer, tokenizer_reverse, input_text, tokens_to_generate=10, use_characters=False, use_chess_moves=True, top_k=5):
    # Set the model to evaluation mode
    model.eval()
    # Move the model to the appropriate device (GPU or CPU)
    model.to(device)

    if use_characters or use_chess_moves:
        special_tokens = ['<STARTGAME>', '<EOFG>']
        tokens = []
        i = 0
        while i < len(input_text):
            found_special = False
            # Check for special tokens first
            for token in special_tokens:
                if input_text[i:].startswith(token):
                    tokens.append(tokenizer[token])
                    i += len(token)
                    found_special = True
                    break
            if not found_special:
                if use_chess_moves:
                    # Handle chess moves tokenization
                    move = input_text[i:i+4].strip().upper()  # Strip whitespace and convert to uppercase
                    if move in tokenizer:
                        tokens.append(tokenizer[move])
                        i += len(move)
                    else:
                        print(f"Warning: Move '{move}' not in vocabulary, skipping.")
                        i += 1
                else:
                    # Character-based tokenization
                    if input_text[i] in tokenizer:
                        tokens.append(tokenizer[input_text[i]])
                    else:
                        print(f"Warning: Character '{input_text[i]}' not in vocabulary, skipping.")
                    i += 1
            # Skip any whitespace between moves for chess_moves tokenization
            while use_chess_moves and i < len(input_text) and input_text[i].isspace():
                i += 1
        # Convert tokens to a PyTorch tensor and move to the appropriate device
        tokens = torch.tensor([tokens], dtype=torch.long).to(device)
    else:
        # Use GPT-2 tokenizer for non-character-based models
        tokens = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        generated_tokens = []
        for _ in range(tokens_to_generate):
            # Get model output
            output, _ = model(tokens)
            # Calculate probabilities for the next token
            pred_probs = F.softmax(output[0, -1], dim=-1)
            # Get the top k most likely next tokens
            top_k_probs, top_k_indices = torch.topk(pred_probs, k=top_k)
            
            # Add the top k indices to the generated tokens
            generated_tokens.append(top_k_indices.tolist())
            # Update input tokens for the next iteration
            tokens = torch.cat((tokens[:, 1:], top_k_indices[0].unsqueeze(0).unsqueeze(0)), dim=1)
    
    if use_characters:
        # Convert generated tokens back to text for character-based or chess moves models
        generated_text = [''.join([tokenizer_reverse[idx] for idx in token_list]) for token_list in zip(*generated_tokens)]
    elif use_chess_moves:
        # Decode generated tokens for chess moves models
        generated_text = [' '.join([tokenizer_reverse.get(idx, f'[UNK:{idx}]') for idx in token_list]) for token_list in zip(*generated_tokens)]
    else:
        # Decode generated tokens for GPT-2 based models
        generated_text = [tokenizer.decode(token_list) for token_list in zip(*generated_tokens)]
    
    return generated_text

# Comparison with test_progress function:
# 1. This function uses top-k sampling, while test_progress uses greedy sampling (always choosing the most likely token).
# 2. This function handles both character-based and GPT-2 tokenization, while test_progress assumes the tokenization method.
# 3. This function returns multiple possible continuations (top-k), while test_progress returns a single continuation.
# 4. This function doesn't print intermediate results or update a text log, focusing solely on text generation.

# Clarity comparison:
# - This function is more flexible but potentially more complex due to handling multiple tokenization methods and top-k sampling.
# - test_progress is simpler and more focused on demonstrating model progress during training.
# - Both functions could benefit from additional error handling and possibly breaking down into smaller, more focused functions.


def initialize_model():
    global global_model, global_tokenizer, global_tokenizer_reverse, global_use_characters, global_use_chess_moves  # Changed: Added global_use_chess_moves
    
    model, vocab_size, n_embd, n_head, block_size, n_layer, dropout, tokenizer, use_characters, use_chess_moves = load_model_file(None, None, None, None, None, None)  # Changed: Added use_chess_moves
    if model is None:
        print("Failed to load model.")
        return None
    
    global_model = model
    global_tokenizer = tokenizer
    global_tokenizer_reverse = {v: k for k, v in tokenizer.items()}
    global_use_characters = use_characters
    global_use_chess_moves = use_chess_moves  # Changed: Set global_use_chess_moves
    if not (use_characters or use_chess_moves):  # Changed: Load GPT-2 tokenizer only if not using characters or chess moves
        global_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    return model

# Add this new function for visualization
def visualize_chess_game_analysis(model, input_text, tokenizer, tokenizer_reverse, use_characters, use_chess_moves):
    model.eval()
    model.to(device)

    if use_characters or use_chess_moves:
        special_tokens = ['<STARTGAME>', '<EOFG>']
        tokens = []
        i = 0
        while i < len(input_text):
            found_special = False
            # Check for special tokens first
            for token in special_tokens:
                if input_text[i:].startswith(token):
                    tokens.append(tokenizer[token])
                    i += len(token)
                    found_special = True
                    break
            if not found_special:
                if use_chess_moves:
                    # Handle chess moves tokenization
                    move = input_text[i:i+4].strip().upper()  # Strip whitespace and convert to uppercase
                    if move in tokenizer:
                        tokens.append(tokenizer[move])
                        i += len(move)
                    else:
                        print(f"Warning: Move '{move}' not in vocabulary, skipping.")
                        i += 1
                else:
                    # Character-based tokenization
                    if input_text[i] in tokenizer:
                        tokens.append(tokenizer[input_text[i]])
                    else:
                        print(f"Warning: Character '{input_text[i]}' not in vocabulary, skipping.")
                    i += 1
            # Skip any whitespace between moves for chess_moves tokenization
            while use_chess_moves and i < len(input_text) and input_text[i].isspace():
                i += 1
        # Convert tokens to a PyTorch tensor and move to the appropriate device
        tokens = torch.tensor([tokens], dtype=torch.long).to(device)
    else:
        # Use GPT-2 tokenizer for non-character-based models
        tokens = tokenizer.encode(input_text, return_tensors='pt').to(device)

    if tokens.size(1) == 0:
        print("No valid tokens in input. Please check your input and tokenizer.")
        return

    # Forward pass
    with torch.no_grad():
        output, _ = model(tokens)

    # Generate and print the next moves
    next_moves = generate_response(
        model, tokenizer, tokenizer_reverse, input_text,
        tokens_to_generate=5, use_characters=use_characters, use_chess_moves=use_chess_moves
    )
    print("\nTop 5 predicted next moves:")
    for i, move in enumerate(next_moves, 1):
        print(f"{i}. {move}")

    # Analyze move transition probabilities
    transition_probs = F.softmax(output[0, -1], dim=-1)
    top_k = 10
    top_indices = torch.argsort(transition_probs, descending=True)[:top_k]
    top_probs = transition_probs[top_indices]

    print("\nTop 10 most likely next moves:")
    for i, (idx, prob) in enumerate(zip(top_indices, top_probs), 1):
        move = tokenizer_reverse[idx.item()] if idx.item() in tokenizer_reverse else f"[UNK:{idx.item()}]"
        print(f"{i}. {move}: {prob.item():.4f}")

    # Visualizations
    try:
        if tokens.size(1) > 1:
            # Visualize attention patterns
            visualize_attention_patterns(tokens, model.attention_weights, tokenizer_reverse)

            # Visualize token embeddings
            visualize_embeddings(model, tokenizer_reverse)

            # Visualize move transition probabilities
            plt.figure(figsize=(10, 6))
            plt.bar(range(1, top_k + 1), top_probs.cpu().numpy())
            plt.title("Top 10 Move Transition Probabilities")
            plt.xlabel("Move Rank")
            plt.ylabel("Probability")
            plt.xticks(range(1, top_k + 1))
            plt.tight_layout()
            plt.show()

        else:
            print("\nNot enough moves to visualize patterns.")
    except Exception as e:
        print(f"An error occurred during visualization: {e}")
        import traceback
        traceback.print_exc()

def visualize_attention_patterns(input_tokens, attention_weights, tokenizer_reverse):
    tokens = input_tokens.squeeze().tolist()
    token_labels = [tokenizer_reverse.get(token_id, f'[UNK:{token_id}]') for token_id in tokens]
    
    num_layers = len(attention_weights)
    num_heads = attention_weights[0].size(1)
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle("Attention Patterns Across Layers and Heads", fontsize=16)
    
    for i, (layer_idx, head_idx) in enumerate([(0, 0), (num_layers//2, num_heads//2), (num_layers-1, 0), (num_layers-1, num_heads-1)]):
        ax = axes[i//2, i%2]
        attn = attention_weights[layer_idx][0, head_idx].detach().cpu().numpy()
        sns.heatmap(attn, xticklabels=token_labels, yticklabels=token_labels, cmap='viridis', ax=ax)
        ax.set_title(f'Layer {layer_idx+1}, Head {head_idx+1}')
        ax.set_xlabel('Keys (Previous Moves)')
        ax.set_ylabel('Queries (Current Move)')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    plt.tight_layout()
    plt.show()

def visualize_embeddings(model, tokenizer_reverse, method='pca', num_points=200):
    embeddings = model.token_embedding_table.weight.detach().cpu().numpy()
    tokens = list(tokenizer_reverse.keys())
    labels = [tokenizer_reverse[token_id] for token_id in tokens]

    if len(tokens) > num_points:
        indices = np.random.choice(len(tokens), size=num_points, replace=False)
        embeddings = embeddings[indices]
        labels = [labels[i] for i in indices]

    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2)
    else:
        raise ValueError("Method should be 'pca' or 'tsne'.")

    embedding_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], marker='o', c=range(len(labels)), cmap='viridis')

    # Add labels for some interesting points
    for i, label in enumerate(labels):
        if label in ['<STARTGAME>', '<EOFG>'] or np.random.random() < 0.1:  # Label special tokens and 10% of other tokens
            plt.annotate(label, (embedding_2d[i, 0], embedding_2d[i, 1]), fontsize=9, alpha=0.7)

    plt.colorbar(scatter, label='Token Index')
    plt.title(f'Chess Move Embeddings Visualized using {method.upper()}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load a pre-trained model
    model, vocab_size, n_embd, n_head, block_size, n_layer, dropout, tokenizer, use_characters, use_chess_moves = load_model_file(None, None, None, None, None, None, frommain=True)
    if model is None:
        print("Failed to load model.")
        exit()      
    tokenizer_reverse = {v: k for k, v in tokenizer.items()}    

    if not (use_characters or use_chess_moves):
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    while True:
        input_text = input("Enter a chess game history (include <startgame> if needed, type 'quit' to exit): ")
        if input_text.lower() == 'quit':
            break
        
        # Generate response
        response = generate_response(model, tokenizer, tokenizer_reverse, input_text, use_characters=use_characters, use_chess_moves=use_chess_moves)
        print("Response:", response)
        
        # Add visualization
        visualize_chess_game_analysis(model, input_text, tokenizer, tokenizer_reverse, use_characters, use_chess_moves)