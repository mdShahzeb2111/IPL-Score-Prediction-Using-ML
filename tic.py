import pandas as pd
import numpy as np

# Define the number of games
num_games = 28528

# Define the board size
board_size = 9

# Create an empty list to store the games
games = []

# Loop through each game
for i in range(num_games):
    # Create an empty board
    board = ['1'] * board_size
    
    # Randomly select a winner (X or O)
    winner = np.random.choice(['X', 'O'])
    
    # Randomly select a number of moves for the winner
    num_moves = np.random.randint(1, board_size)
    
    # Make the moves for the winner
    for j in range(num_moves):
        move = np.random.randint(0, board_size)
        while board[move] != '1':
            move = np.random.randint(0, board_size)
        board[move] = winner
    
    # Determine the outcome of the game
    if winner == 'X':
        outcome = 'win'
    elif winner == 'O':
        outcome = 'loss'
    else:
        outcome = 'draw'
    
    # Add the game to the list
    games.append([','.join(board), outcome])

# Create a Pandas dataframe from the list of games
df = pd.DataFrame(games, columns=['board', 'outcome'])

# Save the dataframe to a CSV file
df.to_csv('tictactoe.csv', index=False)