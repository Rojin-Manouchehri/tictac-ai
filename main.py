"""
Basic 3D Tic Tac Toe with Minimax and Alpha-Beta pruning, using a simple
heuristic to check for possible winning moves or blocking moves if no better
alternative exists.
"""


from shutil import move
from colorama import Back, Style, Fore
import numpy as np


class TicTacToe3D(object):
    """
    3D TTT logic and underlying game state object.
    Attributes:
        board (np.ndarray)3D array for board state.
        difficulty (int): Ply; number of moves to look ahead.
        depth_count (int): Used in conjunction with ply to control depth.
    Args:
        player (str): Player that makes the first move.
        player_1 (Optional[str]): player_1's character.
        player_2 (Optional[str]): player_2's character.
        ply (Optional[int]): Number of moves to look ahead.
    """

    def __init__(self, board=None, player=-1, player_1=-1, player_2=1, ply=3):
        if board is not None:
            assert type(board) == np.ndarray, "Board must be a numpy array"
            assert board.shape == (3, 3, 3), "Board must be 3x3x3"
            self.np_board = board
        else:
            self.np_board = self.create_board()
        self.map_seq_to_ind, self.map_ind_to_seq = self.create_map()
        self.allowed_moves = list(range(pow(3, 3)))
        self.difficulty = ply
        self.depth_count = 0
        if player == player_1:
            self.player_1_turn = True
        else:
            self.player_1_turn = False
        self.player_1 = player_1
        self.player_2 = player_2
        self.players = (self.player_1, self.player_2)

    def create_map(self):
        """Create a mapping between index of 3D array and list of sequence, and vice-versa.
        Args: None
        Returns:
            map_seq_to_ind (dict): Mapping between sequence and index.
            map_ind_to_seq (dict): Mapping between index and sequence.
        """
        a = np.hstack((np.zeros(9), np.ones(9), np.ones(9)*2))
        a = a.astype(int)
        b = np.hstack((np.zeros(3), np.ones(3), np.ones(3)*2))
        b = np.hstack((b, b, b))
        b = b.astype(int)
        c = np.array([0, 1, 2], dtype=int)
        c = np.tile(c, 9)
        mat = np.transpose(np.vstack((a, b, c)))
        ind = np.linspace(0, 26, 27).astype(int)
        map_seq_to_ind = {}
        map_ind_to_seq = {}
        for i in ind:
            map_seq_to_ind[i] = (mat[i][0], mat[i][1], mat[i][2])
            map_ind_to_seq[(mat[i][0], mat[i][1], mat[i][2])] = i
        return map_seq_to_ind, map_ind_to_seq

    def reset(self):
        """Reset the game state."""
        self.allowed_moves = list(range(pow(3, 3)))
        self.np_board = self.create_board()
        self.depth_count = 0

    @staticmethod
    def create_board():
        """Create the board with appropriate positions and the like
        Returns:
            np_board (numpy.ndarray):3D array with zeros for each position.
        """
        np_board = np.zeros((3, 3, 3), dtype=int)
        return np_board

  #  def setup():
   #     play_game(self)

    def play_game(self):
        self.np_board = np.array([[[ 0, 0, 0],
                              [ 0, 1, 0],
                              [ 0, 0,-1]],

                             [[ 0, 0, 0],
                              [ 1, 0, 0],
                              [ 0, 0, 0]],

                             [[ 0, 1, 0],
                              [ 0,-1,-1],
                              [ 0, 1, 0]]])
        self.player_1 = -1
        self.player_2 = 1
        current_player = self.player_1  # Start with Player 1 (or AI 1)
        while True:
            # Check for a winner or tie
            if current_player == self.player_1:
                # Call minimax for Player 1 (AI 1)

                best_score = float('-inf')
                best_move = None
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            print("--------------------------------------------------------------------------------------------------------------")
                            if self.np_board[i, j, k] == 0:  # Check if the spot is available
                                self.np_board[i, j, k] = self.player_1
                                winner = self.check_win()
                                if winner != 0:
                                    return self.np_board, winner  # Return the game result
                                if self.is_board_full():
                                    return 0
                                score = self.minimax(0, False, float('-inf'), float('inf'))
                                print("owo")
                                self.np_board[i, j, k] = 0  # Undo the move
                                if score > best_score:
                                    best_score = score
                                    best_move = (i, j, k)
                self.np_board[best_move[0], best_move[1], best_move[2]] = self.player_1
            else:
                # Call minimax for Player 2 (AI 2)
                best_score = float('-inf')
                best_move = None
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            if self.np_board[i, j, k] == 0:  # Check if the spot is available
                                self.np_board[i, j, k] = self.player_2
                                winner = self.check_win()
                                if winner != 0:
                                    return self.np_board, winner  # Return the game result
                                if self.is_board_full():
                                    return 0
                                score = self.minimax(0, False, float('-inf'), float('inf'))
                                self.np_board[i, j, k] = 0  # Undo the move
                                if score > best_score:
                                    best_score = score
                                    best_move = (i, j, k)
                self.np_board[best_move[0], best_move[1], best_move[2]] = self.player_2


            # Switch players for the next turn
            current_player = self.player_1 if current_player == self.player_2 else self.player_2


    #    def play_game(self):
 #       game_ended = False  # Flag to track if the game has ended
#
 #       while not game_ended:
            # Print the current game state here if needed
  #          print("Current game state:")
   #         print(self.np_board)
    #        print("Player 1's turn" if self.player_1_turn else "Player 2's turn")
#
 #           if self.player_1_turn:
   #             # Player 1's turn
  #              self.best_move(self.player_1)
    #        else:
     #           # Player 2's turn
      #          self.best_move(self.player_2)
#
 #           # Check for a win or tie and end the game if necessary
  #          winner = self.check_win()
   #         if winner == 1:
    #            print("Player 1 wins!")
     #           game_ended = True  # Set the game_ended flag to True
      #      elif winner == -1:
       #         print("Player 2 wins!")
        #        game_ended = True  # Set the game_ended flag to True
         #   elif self.is_board_full(self.np_board):
          #      print("It's a tie!")
           #     game_ended = True  # Set the game_ended flag to True
#
 #           # Switch turns
  #          self.player_1_turn = not self.player_1_turn
#
 #       return self.np_board, winner  # Return the game result

    #def make_move(self, current_player):
     #   print("Entering make_move function")  # Add this line
      #  best_score = float('-inf') if self.player_1_turn else float('inf')
       ##move = None
        #for i in range(3):
        # 3   for j in range(3):
          #      for k in range(3):
           #         if self.np_board[i, j, k] == 0:  # Check if the spot is available
            #            print("current_player: ",current_player)
             #           print(i, j, k)
              #          self.np_board[i, j, k] = current_player
               #         print(f"Maximizing: Setting board[{i}][{j}][{k}] to {self.player_1}")
                #        score = self.minimax(self.np_board, 0, not self.player_1_turn, float('-inf'), float('inf'))
                 #       self.np_board[i, j, k] = 0  # Undo the move
                  #      if (self.player_1_turn and score > best_score) or (not self.player_1_turn and score < best_score):
                   #         best_score = score
                    #        print("Maximizing: Beta cut-off")
                     #       move = (i, j, k)
        #print("Best score for the move:", best_score)  # Print the score here
        #self.np_board[move[0], move[1], move[2]] = current_player
        #self.allowed_moves.remove(self.map_ind_to_seq[(move[0], move[1], move[2])])
        #self.player_1_turn = not self.player_1_turn  # Switch the player's turn


    def best_move(self, current_player):
        print("Entering best_move function")  # Add this line
        best_score = float('-inf')  # Initialize best_score
        move = None  # Initialize move

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if self.np_board[i, j, k] == 0:  # Check if the spot is available
                        print("Checking move:", i, j, k)  # Add this line for debugging
                        self.np_board[i, j, k] = current_player
                        score = self.minimax(0, False, float('-inf'), float('inf'))
                        self.np_board[i, j, k] = 0  # Undo the move
                        if score > best_score:
                           best_score = score
                           move = (i, j, k)

        print("Best score for the move (best_move):", best_score)  # Print the score here
        print("Selected move:", move)  # Add this line for debugging
        self.np_board[move[0], move[1], move[2]] = current_player
        self.allowed_moves.remove(self.map_ind_to_seq[(move[0], move[1], move[2])])
        self.player_1_turn = not self.player_1_turn  # Switch the player's turn

    def minimax(self, depth, is_maximizing, alpha, beta):
        # Check for a winner or a tie and return the corresponding score
        winner = self.check_win()
        prune = False
        if winner == 1:
            return 1
        elif winner == -1:
            return -1
        elif self.is_board_full():
            print("WAAAAAAAAAAAAAAAAAAAAAaaa")
            return 0
        if is_maximizing:
            best_score = float('-inf')
            # print(self.np_board)
            for i in range(3):
                for j in range(3):
                    for k in range(3):

                        if self.np_board[i, j, k] == 0:  # Check if the spot is available

                            self.np_board[i, j, k] = self.player_1
                            score = self.minimax(depth + 1, False, alpha, beta)
                            self.np_board[i, j, k] = 0  # Undo the move
                            best_score = max(score, best_score)
                            alpha = max(alpha, best_score)
                            if beta <= alpha or 1 <= alpha:
                                prune = True
                                break  # Beta cut-off
                    if prune:
                        break
                if prune:
                    break
            # print(best_score)
            return best_score
        else:
            best_score = float('inf')
            # print(self.np_board)
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        if self.np_board[i, j, k] == 0:  # Check if the spot is available
                            self.np_board[i, j, k] = self.player_2
                            score = self.minimax(depth + 1, True, alpha, beta)
                            self.np_board[i, j, k] = 0  # Undo the move
                            best_score = min(score, best_score)
                            beta = min(beta, best_score)
                            if beta <= -1 or beta <= alpha:
                                prune = True
                                break  # Alpha cut-off
                    if prune:
                        break
                if prune:
                    break
            # print(best_score)
            return best_score

    def check_win(self):
        # checking rows, columns, and depths for the winner
        for i in range(3):
            for j in range(3):
                sum_row = sum(self.np_board[i, j, k] for k in range(3))
                sum_col = sum(self.np_board[i, k, j] for k in range(3))
                sum_depth = sum(self.np_board[k, i, j] for k in range(3))

                if any(s == 3 for s in [sum_row, sum_col, sum_depth]):
                    return 1
                if any(s == -3 for s in [sum_row, sum_col, sum_depth]):
                    return -1

        # checking 2D diagonals for the winner
        for i in range(3):
            sum_diag_2d = [sum(self.np_board[i, j, j] for j in range(3)),
                           sum(self.np_board[i, j, 2 - j] for j in range(3)),
                           sum(self.np_board[j, i, j] for j in range(3)),
                           sum(self.np_board[j, i, 2 - j] for j in range(3)),
                           sum(self.np_board[j, j, i] for j in range(3)),
                           sum(self.np_board[2 - j, j, i] for j in range(3))]

            for s in sum_diag_2d:
                if s == 3:
                    return 1
                if s == -3:
                    return -1

        # check 3D diagonals
        sum_3D_diag = [sum(self.np_board[i, i, i] for i in range(3)),
                       sum(self.np_board[i, i, 2 - i] for i in range(3)),
                       sum(self.np_board[i, 2 - i, i] for i in range(3)),
                       sum(self.np_board[2 - i, i, i] for i in range(3))]

        for s in sum_3D_diag:
            if s == 3:
                return 1
            if s == -3:
                return -1

        return 0

    def is_board_full(self):
        return not any(self.np_board[i, j, k] == 0 for i in range(3) for j in range(3) for k in range(3))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '--player', dest='player', help='Player that plays first, 1 or -1',
        type=int, default=-1, choices=[1, -1]
    )
    parser.add_argument(
        '--ply', dest='ply', help='Number of moves to look ahead',
        type=int, default=6
    )
    args = parser.parse_args()
    brd, winner = TicTacToe3D(player=args.player, ply=args.ply).play_game()
    print("final board: \n{}".format(brd))
    print("winner: player {}".format(winner))

