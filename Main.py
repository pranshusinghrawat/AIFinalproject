import numpy as np
import random

# Constants for the game
NUM_COLS = 8 # Number of columns and rows (checkers board is 8x8)
WHITE = 1 # Numeric representation for white pieces
NOBODY = 0 # Numeric representation for empty spaces
BLACK = -1 # Numeric representation for black pieces
DIRECTIONS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]  # Movement directions for pieces

class Piece:
    """Class representing a single checker piece on the board."""
    def __init__(self, color, x, y):
        self.player = color    # Color of the piece, either WHITE or BLACK
        self.position = (x, y)  # Tuple representing the coordinates on the board
        self.king = False   # Boolean to determine if the piece is a king

    def make_king(self):
        """Upgrade the piece to a king."""
        self.king = True

class Board:
    """Class representing the checkers game board and game state."""
    def __init__(self):
        # Initialize an 8x8 board with zeros indicating empty spaces
        self.board = np.zeros((NUM_COLS, NUM_COLS), dtype=int)
        self.turn = WHITE # Start with white's turn
        self.game_over = False # Flag to check if the game has ended
        # Initialize white and black pieces in their standard starting positions
        self.white_pieces = [Piece(WHITE, i, j) for i in range(3) for j in range(NUM_COLS) if (i+j) % 2 == 1] 
        self.black_pieces = [Piece(BLACK, i, j) for i in range(5, 8) for j in range(NUM_COLS) if (i+j) % 2 == 1]
        # Place the pieces on the board
        for piece in self.white_pieces + self.black_pieces:
            self.board[piece.position[0], piece.position[1]] = piece.player

    def clone(self):
        """Create a deep copy of the board for simulation purposes."""
        new_board = Board()
        new_board.board = np.copy(self.board)
        new_board.turn = self.turn
        new_board.game_over = self.game_over
        new_board.white_pieces = [Piece(piece.player, piece.position[0], piece.position[1]) for piece in self.white_pieces]
        new_board.black_pieces = [Piece(piece.player, piece.position[0], piece.position[1]) for piece in self.black_pieces]
        return new_board

    def get_possible_moves(self):
        """Generate all legal moves for the current player, prioritizing captures."""
        moves = []
        captures = []
        pieces = self.white_pieces if self.turn == WHITE else self.black_pieces
        for piece in pieces:
            self._find_moves(piece, moves, captures)
        return captures if captures else moves  # Force captures if available

    def _find_moves(self, piece, moves, captures, visited_positions=None):
     """Recursive function to find all moves for a given piece."""
     if visited_positions is None:
        visited_positions = set()

        base_directions = DIRECTIONS[:2] if piece.player == WHITE else DIRECTIONS[2:]
        directions = base_directions + (DIRECTIONS if piece.king else [])
        for dx, dy in directions:
            step = 1
            while True:
                new_row = piece.position[0] + dx * step
                new_col = piece.position[1] + dy * step
                if not (0 <= new_row < NUM_COLS and 0 <= new_col < NUM_COLS):
                    break
                if self.board[new_row, new_col] == NOBODY:
                    if step == 1:
                        moves.append((piece, (new_row, new_col)))
                    else:
                        captures.append((piece, (new_row, new_col)))
                    if not piece.king:
                        break
                elif self.board[new_row, new_col] != piece.player:
                    if step == 1 and 0 <= new_row + dx < NUM_COLS and 0 <= new_col + dy < NUM_COLS and self.board[new_row + dx, new_col + dy] == NOBODY:
                        captures.append((piece, (new_row + dx, new_col + dy)))
                    break
                step += 1

    def make_move(self, piece, new_pos):
        """Move a piece to a new position, handling captures and kinging."""
        # Calculate the displacement
        dx = new_pos[0] - piece.position[0]
        dy = new_pos[1] - piece.position[1]
        if abs(dx) == 2 and abs(dy) == 2:  # This indicates a capture move
            mid_x = piece.position[0] + dx // 2
            mid_y = piece.position[1] + dy // 2
            self.board[mid_x, mid_y] = NOBODY  # Remove the captured piece
            if self.turn == WHITE:
                self.black_pieces = [p for p in self.black_pieces if (p.position[0] != mid_x or p.position[1] != mid_y)]
            else:
                self.white_pieces = [p for p in self.white_pieces if (p.position[0] != mid_x or p.position[1] != mid_y)]
        # Move the piece
        self.board[piece.position[0], piece.position[1]] = NOBODY
        self.board[new_pos[0], new_pos[1]] = piece.player
        piece.position = new_pos
        if (piece.player == WHITE and new_pos[0] == 0) or (piece.player == BLACK and new_pos[0] == NUM_COLS - 1):
            piece.make_king()

    def is_game_over(self):
        """Check if the game is over based on the availability of moves or remaining pieces."""
        if not self.get_possible_moves() and (not self.white_pieces or not self.black_pieces):
            self.game_over = True
        else:
            self.game_over = False
        return self.game_over

    def evaluate(self):
        """Evaluate the board state to determine the score for minimax calculations."""
        white_count = sum(1 for p in self.white_pieces if self.board[p.position[0], p.position[1]] == WHITE)
        black_count = sum(1 for p in self.black_pieces if self.board[p.position[0], p.position[1]] == BLACK)
        return white_count - black_count if self.turn == WHITE else black_count - white_count

class Node:
    """Node for use in the Monte Carlo Tree Search algorithm, representing game states."""
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = board.get_possible_moves()

    def UCB1(self):
        """Calculate the Upper Confidence Bound for tree node selection."""
        if self.visits == 0:
            return float('inf')  # to ensure unvisited nodes are prioritized
        return self.wins / self.visits + np.sqrt(2 * np.log(self.parent.visits) / self.visits)

    def is_fully_expanded(self):
        """Check if all potential moves have been explored for this node."""
        return len(self.untried_moves) == 0

    def add_child(self, move, board):
        """Add a new child node for a move that hasn't been tried yet."""
        child_node = Node(board=board, parent=self, move=move)
        self.untried_moves.remove(move)
        self.children.append(child_node)
        return child_node
    
def minimax(board, depth, alpha, beta, maximizing_player):
    """Implement the minimax algorithm with alpha-beta pruning for optimal move selection."""
    if depth == 0 or board.is_game_over():
        return board.evaluate(), None

    best_move = None
    if maximizing_player:
        max_eval = float('-inf')
        for move in board.get_possible_moves():
            new_board = board.clone()
            new_board.make_move(*move)
            new_board.turn = BLACK
            eval, _ = minimax(new_board, depth - 1, alpha, beta, False)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in board.get_possible_moves():
            new_board = board.clone()
            new_board.make_move(*move)
            new_board.turn = WHITE
            eval, _ = minimax(new_board, depth - 1, alpha, beta, True)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

def MCTS(board, iterations):
    """Monte Carlo Tree Search to find the best move based on random playouts."""
    root_node = Node(board)
    for _ in range(iterations):
        leaf_node = select(root_node)
        expansion_node = expand(leaf_node)
        simulation_result = simulate(expansion_node)
        backpropagate(expansion_node, simulation_result)

    best = best_child(root_node)
    return best.move if best else None  # Check if best is not None


def select(node):
    """Select a node to explore using the UCB1 algorithm."""
    current_node = node
    while not current_node.is_fully_expanded():
        if current_node.untried_moves:
            return expand(current_node)
        else:
            # Select the best child if no untried moves are available
            current_node = max(current_node.children, key=lambda x: x.UCB1()) if current_node.children else current_node
    return current_node

def expand(node):
    """Expand the selected node by adding a new child node for an untried move."""
    if node.untried_moves:
        move = random.choice(node.untried_moves)
        new_board = node.board.clone()
        new_board.make_move(*move)
        new_board.turn = BLACK if new_board.turn == WHITE else WHITE
        return node.add_child(move, new_board)
    else:
        # Return the node itself if there are no moves to expand (could be a terminal state)
        return node

def simulate(node):
    """Simulate a random game from the current node's state to determine a potential game outcome."""
    board_clone = node.board.clone()
    while not board_clone.is_game_over():
        possible_moves = board_clone.get_possible_moves()
        if not possible_moves:
            break
        move = random.choice(possible_moves)
        board_clone.make_move(*move)
        board_clone.turn = BLACK if board_clone.turn == WHITE else WHITE
    return 1 if (board_clone.evaluate() > 0 and board_clone.turn == BLACK) or (board_clone.evaluate() < 0 and board_clone.turn == WHITE) else 0

def backpropagate(node, result):
    """Backpropagate the simulation result up the tree, updating node statistics."""
    while node is not None:
        node.visits += 1
        if node.board.turn == BLACK:  # Assume win is good for the player who just played - hence, check turn *before* move
            node.wins += result
        else:
            node.wins += (1 - result)
        node = node.parent

def best_child(node):
    """Return the child node with the highest visit count, indicating the most promising move."""
    if not node.children:
        return None
    return max(node.children, key=lambda x: x.visits)

def play_game():
    """Main game loop to alternate between players using minimax and MCTS for decision making."""
    board = Board()
    while not board.is_game_over():
        if board.turn == WHITE:
            _, move = minimax(board, 3, float('-inf'), float('inf'), True)
            if move is None:
                print("No moves left for White. Game over.")
                break
            print("Minimax (White) Move:", move)
        else:
            move = MCTS(board, 100)
            if move is None:
                print("No moves left for Black. Game over.")
                break
            print("MCTS (Black) Move:", move)
        board.make_move(*move)
        board.turn = BLACK if board.turn == WHITE else WHITE
        print(board.board)
    print("Game Over. Winner:", "White" if board.evaluate() > 0 else "Black")

play_game()
