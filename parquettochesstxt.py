# JMR parquet to chess txt
# This script converts a Parquet file containing chess game data into a text file with moves separated by spaces.
# Works with my Brain6 to learn, and with my chess_Sept_23_LMM to play.
# https://huggingface.co/datasets/laion/strategic_game_chess

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

def convert_parquet_to_text(input_parquet, output_text):
    # Read the Parquet file
    df = pd.read_parquet(input_parquet)
    
    # Print column names
    print("Available columns:", df.columns.tolist())
    
    # Assuming 'Moves' is the correct column name
    move_column = 'Moves'
    
    # Open the output text file
    with open(output_text, 'w') as f:
        # Iterate through each row (game) in the DataFrame
        for index, row in df.iterrows():
            # Get the moves
            moves = row[move_column]
            
            # Process moves: convert to string, remove move numbers and join with spaces
            if isinstance(moves, np.ndarray):
                processed_moves = ' '.join(move.split('.')[-1].strip() for move in moves if isinstance(move, str))
            elif isinstance(moves, str):
                processed_moves = ' '.join(move.split('.')[-1].strip() for move in moves.split())
            else:
                print(f"Unexpected data type for moves in row {index}: {type(moves)}")
                continue
            
            # Write the processed moves to the file
            f.write(processed_moves)
            
            # Add two newlines after each game
            f.write('\n\n')
            
            # Debug: Print the first 5 entries
            if index < 5:
                print(f"Game {index + 1}:")
                print(f"Original moves: {moves[:100]}...")  # Print first 100 characters
                print(f"Processed moves: {processed_moves[:100]}...")
                print()

    # Print the first 5 rows of the DataFrame
    print("\nFirst 5 rows of the DataFrame:")
    print(df.head())

# Create a root window and hide it
root = tk.Tk()
root.withdraw()

# Open file dialog to select multiple input Parquet files
input_files = filedialog.askopenfilenames(title="Select Parquet files", filetypes=[("Parquet files", "*.parquet")])

# Open file dialog to specify output directory
output_directory = filedialog.askdirectory(title="Select output directory")

if input_files and output_directory:
    for input_file in input_files:
        output_file = os.path.join(output_directory, os.path.basename(input_file).replace('.parquet', '.txt'))
        convert_parquet_to_text(input_file, output_file)
        print(f"Conversion complete. Output saved to {output_file}")
else:
    print("Required file selection not completed. Conversion cancelled.")