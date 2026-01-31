# Brain6
This is my own transformer based model to create small LLMs, started off doing the usual Shakespeare thing, but added ability to tokenize chess games

# Advanced Chess Game Transformer

## Overview

This project implements an advanced transformer-based language model specifically designed for chess game analysis and generation. It utilizes state-of-the-art deep learning techniques to process and generate chess game sequences, supporting both character-level and move-level tokenization.

## Key Features

- **Flexible Data Input**: Supports loading data from text files, chess game character sequences, or chess move sequences.
- **Multi-GPU Training**: Utilizes DataParallel for efficient training across multiple GPUs.
- **Mixed Precision Training**: Implements automatic mixed precision for faster training and reduced memory usage on compatible hardware.
- **Customizable Model Architecture**: Allows fine-tuning of model parameters including embedding size, number of heads, layers, and more.
- **Adaptive Learning Rate**: Uses CosineAnnealingLR scheduler for optimal learning rate adjustment.
- **Gradient Clipping**: Implements gradient clipping to prevent exploding gradients.
- **Checkpoint Management**: Supports saving and loading of model checkpoints, including optimizer and scheduler states.
- **Interactive Model Configuration**: Provides a user-friendly interface for configuring model parameters and training settings.

## Requirements

- Python 3.7+
- PyTorch 1.7+
- transformers library
- CUDA-compatible GPU (for GPU acceleration)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jmrothberg/Brain6.git
   cd Brain6
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script:


Follow the interactive prompts to:
1. Choose the data source (text file, chess games as characters, or chess games as moves).
2. Select whether to load an existing model or create a new one.
3. Configure model parameters and training settings.

## Model Architecture

The transformer model consists of:
- Token and positional embeddings
- Multi-head attention layers
- Feed-forward neural networks
- Layer normalization

Key components:
- `TransformerModel`: The main model class.
- `MultiHeadAttention`: Implements the multi-head attention mechanism.
- `FeedForward`: The position-wise feed-forward network.
- `TransformerBlock`: Combines attention and feed-forward layers.

## Data Processing

- `GPT2TokenDataset`: For processing general text using GPT-2 tokenizer.
- `ChessChrDataset`: For processing chess games as character sequences.
- `ChessMovesDataset`: For processing chess games as move sequences.

## Training Process

1. Data is loaded and tokenized based on the selected input method.
2. The model is initialized or loaded from a checkpoint.
3. Training proceeds with periodic evaluation and checkpoint saving.
4. Gradient scaling and clipping are applied to stabilize training.

## Inference

The `test_progress` function demonstrates the model's current capabilities by generating new sequences based on input prompts.

## Customization

Users can customize various aspects including:
- Model architecture (embedding size, number of heads, layers, etc.)
- Training parameters (learning rate, batch size, number of epochs, etc.)
- Data processing (tokenization method, sequence length)

## Performance Optimization

- Utilizes DataParallel for multi-GPU training.
- Implements automatic mixed precision for efficient GPU utilization.
- Adaptive learning rate scheduling for improved convergence.

## Limitations and Future Work

- Currently optimized for chess game analysis; may require modifications for other domains.
- Future updates may include more advanced sampling techniques for text generation.
- Potential for integration with chess engines for move validation and analysis.

## Contributing

Contributions to improve the model or extend its capabilities are welcome. Please submit pull requests or open issues for any bugs or feature requests.

## Author

**Jonathan M. Rothberg** - [@jmrothberg](https://github.com/jmrothberg)

## License

MIT License

## Acknowledgments

- This project builds upon the transformer architecture introduced in "Attention Is All You Need" by Vaswani et al.
- Inspired by the GPT (Generative Pre-trained Transformer) series of models.

