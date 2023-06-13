# %%

from utils import *

assert board_label_to_row_col("A0") == (0, 0)
assert board_label_to_row_col("A1") == (0, 1)
assert board_label_to_row_col("B1") == (1, 1)
assert board_label_to_row_col("H7") == (7, 7)

tokens = t.arange(1, 61)
board_index = TOKENS_TO_BOARD[tokens]
assert board_index.min() == 0, board_index
assert board_index.max() == 63, board_index
tokens_2 = BOARD_TO_TOKENS[board_index]
assert (tokens == tokens_2).all(), [tokens, tokens_2]

# %%
print("All tests passed!")
