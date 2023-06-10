# %%
import os

# os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import copy
import dataclasses
import itertools
import random
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from circuitsvis.attention import attention_patterns
import einops
import numpy as np
import plotly.express as px
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformer_lens
import transformer_lens.utils as utils
import wandb
from IPython.display import HTML, display
from ipywidgets import interact
from jaxtyping import Bool, Float, Int, jaxtyped
from neel_plotly import line, scatter
from rich import print as rprint
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformer_lens import (ActivationCache, FactoredMatrix, HookedTransformer,
                              HookedTransformerConfig)
from transformer_lens.hook_points import HookedRootModule, HookPoint

from plotly_utils import imshow

# %%

OTHELLO_ROOT = (Path(__file__).parent / "othello_world").resolve()
OTHELLO_MECHINT_ROOT = (OTHELLO_ROOT / "mechanistic_interpretability").resolve()

if not OTHELLO_ROOT.exists():
    os.system("git clone https://github.com/likenneth/othello_world")

from othello_world.mechanistic_interpretability.mech_interp_othello_utils import (
    OthelloBoardState, int_to_label, plot_board, plot_board_log_probs, plot_single_board,
    string_to_label, to_int, to_string)

# %%


def get_othello_gpt(device: str) -> Tuple[HookedTransformerConfig, HookedTransformer]:
    cfg = HookedTransformerConfig(
        n_layers=8,
        d_model=512,
        d_head=64,
        n_heads=8,
        d_mlp=2048,
        d_vocab=61,
        n_ctx=59,
        act_fn="gelu",
        normalization_type="LNPre",
        device=device,
    )
    model = HookedTransformer(cfg).to(device)

    sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens",
                                     "synthetic_model.pth")
    # champion_ship_sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "championship_model.pth")
    model.load_state_dict(sd)

    return cfg, model.to(device)


def load_sample_games(max_games: int = 100_000,
) -> Tuple[Int[Tensor, "games=100000 moves=60"], Int[Tensor, "games=100000 moves=60"]]:
    """
    Load a sample of games from the dataset.

    Returns:
        A tuple of `(full_games_tokens, full_games_board_index)`.
        - full_games_tokens is a tensor of shape `(num_games, length_of_game)` containing the board state at
          each move, as an integer from 0 to 60.  Suitable for input into the model.
          0 corresponds to 'pass' and is not used.
        - full_games_board_index is a tensor of shape `(num_games, length_of_game)` containing the board state at
          each move, as an integer from 0 to 63 with the middle squares skipped out.


    """
    full_games_tokens = t.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_int_small.npy"),
                                 dtype=t.long)
    full_games_board_index = t.tensor(np.load(OTHELLO_MECHINT_ROOT / "board_seqs_string_small.npy"),
                                      dtype=t.long)

    assert all([middle_sq not in full_games_board_index for middle_sq in [27, 28, 35, 36]])
    assert full_games_tokens.max() == 60
    assert full_games_tokens.min() == 1

    num_games, length_of_game = full_games_tokens.shape
    print("Number of games:", num_games)
    print("Length of game:", length_of_game)

    # We only want to use a subset of the games
    # We'll use the first `max_games` games
    full_games_tokens = full_games_tokens[:max_games]
    full_games_board_index = full_games_board_index[:max_games]

    return full_games_tokens, full_games_board_index


# %%
tokens_to_board = [i for i in range(64) if i not in [27, 28, 35, 36]]
"""Map from token index to board index"""


def to_board_label(board_index: int) -> str:
    """Convert an index into a board label, e.g. `E2`. 0 ≤ i < 64"""
    letter = "ABCDEFGH"[board_index // 8]
    return f"{letter}{board_index%8}"


# Get our list of board labels
board_labels = list(map(to_board_label, tokens_to_board))
"""Map from token index to board label, e.g. `E2`. 0 ≤ i < 60"""
full_board_labels = list(map(to_board_label, range(64)))
"""Map from token index to board label, e.g. `E2`. 0 ≤ i < 64"""


# %%
def logits_to_board(logits: Float[Tensor, "... 61"]) -> Float[Tensor, "... rows=8 cols=8"]:
    """Convert a set of logits into a board state, with each cell being a log prob of that cell being played."""
    log_probs = logits.log_softmax(-1)
    # Remove the "pass" move (the zeroth vocab item)
    log_probs = log_probs[..., 1:]
    assert log_probs.shape[-1] == 60

    extra_shape = log_probs.shape[:-1]
    # Set all cells to -13 by default, for a very negative log prob - this means the middle cells don't show up as mattering
    temp_board_state = (t.zeros((*extra_shape, 64), dtype=t.float32, device=logits.device) - 13.0)
    temp_board_state[..., tokens_to_board] = log_probs
    return temp_board_state.reshape(*extra_shape, 8, 8)


# %%
def plot_square_as_board(state: Float[Tensor, "... rows=8 cols=8"],
                         diverging_scale: bool = True,
                         **kwargs):
    """Takes a square input (8 by 8) and plot it as a board. Can do a stack of boards via facet_col=0"""
    kwargs = {
        "y": list("ABCDEFGH"),
        "x": [str(i) for i in range(8)],
        "color_continuous_scale": "RdBu" if diverging_scale else "Blues",
        "color_continuous_midpoint": 0.0 if diverging_scale else None,
        "aspect": "equal",
        **kwargs,
    }
    imshow(state, **kwargs)


# %%
def one_hot(list_of_ints: List[int], num_classes=64):
    """Encode a list of ints into a one-hot vector of length `num_classes`"""
    out = t.zeros((num_classes, ), dtype=t.float32)
    out[list_of_ints] = 1.0
    return out


# %%


def alternate_states(
    state_stack: Float[Tensor, "games moves rows=8 cols=8"]
) -> Float[Tensor, "games moves rows=8 cols=8"]:
    """
    Convert the states to be in terms of my (+1) and their (-1), rather than black and white.
    """
    alternating = t.tensor([-1 if i % 2 == 0 else 1 for i in range(state_stack.shape[1])])
    return state_stack * alternating[:, None, None]


def move_sequence_to_state(
    moves_board_index: Int[Tensor, "batch moves"],
    mode: Literal["valid", "alternate", "normal"] = "normal",
) -> Float[Tensor, "batch moves rows=8 cols=8"]:
    """Convert sequences of moves into a sequence of board states.
    Moves are encoded as integers from 0 to 63.

    If `mode="valid"`, then the board state is a one-hot encoding of the valid moves.
    If `mode="alternate"`, then the board state encoded as mine (+1) and their (-1) pieces.

    Output shape: `(batch, moves, rows=8, cols=8)
    """
    assert len(moves_board_index.shape) == 2

    states = t.zeros((*moves_board_index.shape, 8, 8), dtype=t.float32)
    for b, moves in enumerate(moves_board_index):
        board = OthelloBoardState()
        for m, move in enumerate(moves):
            board.umpire(move.item())
            if mode == "valid":
                states[b, m] = one_hot(board.get_valid_moves()).reshape(8, 8)
            else:
                states[b, m] = t.tensor(board.state)

    if mode == "alternate":
        states = alternate_states(states)
    return states


# %%
def state_stack_to_one_hot(
    state_stack: Float[Tensor, "games moves rows=8 cols=8"]
) -> Int[Tensor, "games moves rows=8 cols=8 options=3"]:
    """
    Creates a tensor of shape (games, moves, rows=8, cols=8, options=3), where the [g, m, r, c, :]-th entry
    is a one-hot encoded vector for the state of game g at move m, at row r and column c. In other words, this
    vector equals (1, 0, 0) when the state is empty, (0, 1, 0) when the state is "their", and (0, 0, 1) when the
    state is "my".
    This works if the input has values for mine/theirs. If it has values for black/white,
    this will return a tensor where (1, 0, 0) is empty, (0, 1, 0) is white, and (0, 0, 1) is black.
    """
    one_hot = t.zeros(
        state_stack.shape[0],  # num games
        state_stack.shape[1],  # num moves
        8,
        8,
        3,  # the options: empty, white, or black
        device=state_stack.device,
        dtype=t.int,
    )
    one_hot[..., 0] = state_stack == 0
    one_hot[..., 1] = state_stack == -1
    one_hot[..., 2] = state_stack == 1

    return one_hot
