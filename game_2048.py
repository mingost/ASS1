import pygame
import numpy as np
import random
from enum import Enum

class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Game2048:
    def __init__(self, size=4):
        self.size = size
        self.reset()
        
    def reset(self):
        """Reset the game board"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.game_over = False
        self.add_new_tile()
        self.add_new_tile()
        return self.get_state()
    
    def add_new_tile(self):
        """Add a new tile (2 or 4) to a random empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            row, col = random.choice(empty_cells)
            self.board[row, col] = 2 if random.random() < 0.9 else 4
    
    def move(self, direction):
        """Execute a move in the given direction"""
        old_board = self.board.copy()
        moved = False
        
        if direction == Direction.UP:
            moved = self._move_up()
        elif direction == Direction.DOWN:
            moved = self._move_down()
        elif direction == Direction.LEFT:
            moved = self._move_left()
        elif direction == Direction.RIGHT:
            moved = self._move_right()
        
        if moved:
            self.add_new_tile()
            reward = np.sum(self.board) - np.sum(old_board)
            # Check game over after adding new tile
            self.game_over = self.is_game_over()
        else:
            reward = -10  # Penalty for invalid move
            # Also check game over on invalid move
            self.game_over = self.is_game_over()
        
        return self.get_state(), reward, self.game_over
    
    def _move_left(self):
        """Move all tiles to the left"""
        moved = False
        for i in range(self.size):
            row = self.board[i, :]
            new_row, row_moved = self._merge_line(row)
            self.board[i, :] = new_row
            moved = moved or row_moved
        return moved
    
    def _move_right(self):
        """Move all tiles to the right"""
        moved = False
        for i in range(self.size):
            row = self.board[i, ::-1]
            new_row, row_moved = self._merge_line(row)
            self.board[i, :] = new_row[::-1]
            moved = moved or row_moved
        return moved
    
    def _move_up(self):
        """Move all tiles up"""
        moved = False
        for j in range(self.size):
            col = self.board[:, j]
            new_col, col_moved = self._merge_line(col)
            self.board[:, j] = new_col
            moved = moved or col_moved
        return moved
    
    def _move_down(self):
        """Move all tiles down"""
        moved = False
        for j in range(self.size):
            col = self.board[::-1, j]
            new_col, col_moved = self._merge_line(col)
            self.board[:, j] = new_col[::-1]
            moved = moved or col_moved
        return moved
    
    def _merge_line(self, line):
        """Merge a single line (row or column)"""
        non_zero = line[line != 0]
        merged = []
        skip = False
        moved = False
        
        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged.append(non_zero[i] * 2)
                self.score += non_zero[i] * 2
                skip = True
                moved = True
            else:
                merged.append(non_zero[i])
        
        merged += [0] * (self.size - len(merged))
        new_line = np.array(merged)
        
        if not moved and not np.array_equal(line, new_line):
            moved = True
            
        return new_line, moved
    
    def is_game_over(self):
        """Check if the game is over"""
        if np.any(self.board == 0):
            return False
        
        for i in range(self.size):
            for j in range(self.size):
                if j < self.size - 1 and self.board[i, j] == self.board[i, j + 1]:
                    return False
                if i < self.size - 1 and self.board[i, j] == self.board[i + 1, j]:
                    return False
        return True
    
    def get_state(self):
        """Get the current state as a flattened array"""
        return self.board.flatten()
    
    def get_max_tile(self):
        """Get the maximum tile value"""
        return np.max(self.board)
