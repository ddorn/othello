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
TOKENS_TO_BOARD = t.tensor([i for i in range(64) if i not in [27, 28, 35, 36]])

def tokens_to_board(tokens):
    """Map from token index (0 <= t < 60) to board index (0 < b < 64)"""
    return TOKENS_TO_BOARD[tokens]



def to_board_label(board_index: int) -> str:
    """Convert an index into a board label, e.g. `E2`. 0 ≤ i < 64"""
    letter = "ABCDEFGH"[board_index // 8]
    return f"{letter}{board_index%8}"


# Get our list of board labels
board_labels = list(map(to_board_label, TOKENS_TO_BOARD))
"""Map from token index to board label, e.g. `E2`. 0 ≤ i < 60"""
full_board_labels = list(map(to_board_label, range(64)))
"""Map from token index to board label, e.g. `E2`. 0 ≤ i < 64"""


# %%
def logits_to_board(logits: Float[Tensor, "... 61"],
                    mode: Literal['log_prob', 'prob', 'logits'],
                    ) -> Float[Tensor, "... rows=8 cols=8"]:
    """Convert a set of logits into a board state, with each cell being a log prob of that cell being played."""
    if mode == "log_prob":
        x = logits.log_softmax(-1)
    elif mode == "prob":
        x = logits.softmax(-1)
    elif mode == "logits":
        x = logits
    # Remove the "pass" move (the zeroth vocab item)
    x = x[..., 1:]
    assert x.shape[-1] == 60

    extra_shape = x.shape[:-1]
    temp_board_state = t.zeros((*extra_shape, 64), dtype=t.float32, device=logits.device)
    temp_board_state[..., TOKENS_TO_BOARD] = x
    # Set all cells to -13 by default, for a very negative log prob - this means the middle cells don't show up as mattering
    if mode == "log_prob":
        temp_board_state[..., [27, 28, 35, 36]] = -13
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


def board_to_tensor(board: str) -> Int[Tensor, "row=8 cols=8"]:
    """Convert a string of 'x', 'o', '.' to a tensor.

    The output is a tensor of shape (8, 8) with 0 for blank, +1 for mine (x), -1 for theirs (o)
    """

    lines = board.strip().split("\n")
    assert len(lines) == 8
    lines = [line.strip() for line in lines]
    assert all(len(line) == 8 for line in lines)
    data = [
        [0 if c == "." else 1 if c == "x" else -1 for c in line]
        for line in lines
    ]
    return t.tensor(data, dtype=t.int8)
# %%

@t.inference_mode()
def get_loss(
    model: HookedTransformer,
    games_token: Int[Tensor, "batch game_len rows cols"],
    games_board_index: Int[Tensor, "batch game_len"],
    move_start: int = 5,
    move_end: int = -5,
) -> Float[Tensor, "2"]:
    """Get the loss of the model on the given games.

    Args:
        model (HookedTransformer): the model to evaluate
        games_token (Int[Tensor, "batch game_len rows cols"]): the tokenized games, integers between 0 and 60
        games_board_index (Int[Tensor, "batch game_len"]): the board index of the games, integers between 0 and 64
        move_start (int, optional): The first move to consider. Defaults to 5.
        move_end (int, optional): The last move to consider. Defaults to -5.

    Returns:
        Float[Tensor, "2"]: the loss and accuracy of the model on the given games
    """
    # This is the input to our model
    assert isinstance(games_token, Int[Tensor, "batch full_game_len=60"])

    valid_moves = move_sequence_to_state(games_board_index, 'valid')
    valid_moves = valid_moves[:, move_start:move_end].to(device)
    # print("valid moves:", valid_moves.shape)
    assert isinstance(valid_moves, Float[Tensor, f"batch game_len rows=8 cols=8"])

    logits = model(games_token[:, :move_end])[:, move_start:]
    # print("model output:", logits.shape)
    logits_as_board = logits_to_board(logits)
    # print("logit as board:", logits_as_board.shape)

    # Flatten the last 2 dimensions to have a 64-dim vector instead of 8x8
    logits_as_board = einops.rearrange(
        logits_as_board,
        "batch move row col -> batch move (row col)",
    )
    valid_moves = einops.rearrange(
        valid_moves,
        "batch move row col -> batch move (row col)",
    )
    log_probs = logits_as_board.log_softmax(dim=-1)

    loss = log_probs * valid_moves
    # print("loss:", loss.shape)
    loss = -loss.sum(dim=-1).mean()

    # Compute accuracy
    nb_valid_moves = valid_moves.sum(dim=-1, keepdim=True)
    predicted = logits_as_board.softmax(dim=-1) > 1 / (2 * nb_valid_moves)
    correct = predicted == valid_moves
    accuracy = correct.float().mean()

    # Return one tensor
    return t.stack([loss, accuracy]).cpu()


@t.inference_mode()
def probe_accuracy(
    model: HookedTransformer,
    probe: Float[Tensor, "d_model rows cols options"],
    game_tokens: Int[Tensor, "batch game_len"],
    game_board_index: Int[Tensor, "batch game_len"],
    pos_start: int = 5,
    pos_end: int = -5,
    layer: int = 6,
    per_option: bool = False,
) -> Float[Tensor, "rows cols *options"]:
    """
    Compute and plot the accuracy of the probe at each position of the board.
    """
    pos_end = pos_end % game_tokens.shape[1]  # making sure it's positive

    states = move_sequence_to_state(game_board_index)
    flipped_states = alternate_states(states)[:, pos_start:pos_end]

    act_name = utils.get_act_name("resid_post", layer)
    with t.inference_mode():
        _, cache = model.run_with_cache(
            game_tokens[:, :pos_end].to(model.cfg.device),  # We ignore the last moves here
            names_filter=lambda name: name == act_name,
        )

    resid = cache[act_name][:, pos_start:]  # We ignore the first moves here
    probe_out = einops.einsum(
        resid, probe, "game move d_model, d_model row col options -> game move row col options")
    probe_out_value = probe_out.argmax(dim=-1).cpu()

    if not per_option:
        probe_out_value[probe_out_value == 2] = -1

        correct_answers = probe_out_value == flipped_states
        accuracy = einops.reduce(correct_answers.float(), "game move row col -> row col", "mean")

        display(plot_square_as_board(
            1 - accuracy,
            title="Error Rate of Linear Probe",
        ))
    else:
        correct_blank = (probe_out_value == 0) == (flipped_states == 0)
        correct_mine = (probe_out_value == 2) == (flipped_states == 1)
        correct_theirs = (probe_out_value == 1) == (flipped_states == -1)

        correct = t.stack([correct_blank, correct_mine, correct_theirs], dim=-1)
        accuracy = einops.reduce(correct.float(), "game move row col option -> row col option",
                                 "mean")
        print(accuracy.shape)
        display(
            plot_square_as_board(
                1 - accuracy,
                title=f"Error Rate of probes",
                facet_col=2,
                facet_labels=["Blank", "Mine", "Theirs"],
            ))

    return accuracy


# %%

def plot_similarities(vectors: Float[Tensor, '*n_vectors dim'], **kwargs):
    """Plot the dot product between each pair of vectors"""
    vectors = vectors.flatten(end_dim=-2)
    sim = einops.einsum(vectors, vectors, "vec_1 dim, vec_2 dim -> vec_1 vec_2")
    imshow(sim, **kwargs)

def plot_similarities_2(v1: Float[Tensor, '*n_vectors rows cols'],
                        v2: Float[Tensor, '*n_vectors rows cols'],
                        name: str = "vectors"):
    """Plot the dot product between each pair of vectors"""
    if v1.ndim > 2:
        v1 = v1.flatten(end_dim=-3)
    if v2.ndim > 2:
        v2 = v2.flatten(end_dim=-3)
    sim = einops.einsum(
        v1 / t.norm(v1, dim=0),
        v2 / t.norm(v2, dim=0),
        "d_model rows cols, d_model rows cols -> rows cols",
    )
    plot_square_as_board(sim, title=f"Cosine similarity between {name}")
