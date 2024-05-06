# Checkers Game
This Python code implements a checkers game using various artificial intelligence techniques. It includes functionalities for AI vs. AI gameplay.

# Features
Minimax Algorithm: The code utilizes the minimax algorithm with alpha-beta pruning for optimal move selection in the game. This algorithm allows the AI player to make informed decisions by simulating possible future game states and selecting the move that maximizes its chances of winning.
Monte Carlo Tree Search (MCTS): In addition to the minimax algorithm, the code implements the MCTS algorithm for decision-making. MCTS involves repeatedly simulating random games from the current game state and selecting moves based on the simulation results. This approach allows for a more exploratory style of gameplay, potentially uncovering novel strategies.
Object-Oriented Design: The code is organized using object-oriented programming principles, with classes such as Piece representing individual checker pieces, Board representing the game board and state, and Node representing game states for use in the MCTS algorithm.
# How to Use
Installation: Ensure you have Python installed on your system. 
Dependencies: The code requires the numpy library for array manipulation.
Running the Game: Execute the Python script. This will start a game session, which is a game between the two AI applying Minimax and MCTS algorithm. 

