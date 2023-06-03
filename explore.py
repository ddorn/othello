# %%
import os

from pytorch_lightning.utilities.types import STEP_OUTPUT

os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np
import einops
import wandb
from ipywidgets import interact
import plotly.express as px
from pathlib import Path
import itertools
import random
from IPython.display import display
import wandb
from jaxtyping import Float, Int, Bool, jaxtyped
from typing import Any, List, Literal, Union, Optional, Tuple, Callable, Dict
from functools import partial
import copy
import dataclasses
import datasets
from IPython.display import HTML
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)
from tqdm.notebook import tqdm
from dataclasses import dataclass
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import pytorch_lightning as pl
from rich import print as rprint
import pandas as pd
from plotly_utils import imshow
from neel_plotly import scatter, line

device = "cuda" if t.cuda.is_available() else "cpu"

t.set_grad_enabled(False)

# %%

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
model = HookedTransformer(cfg)
# %%
sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "synthetic_model.pth")
# champion_ship_sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "championship_model.pth")
model.load_state_dict(sd)

# %%

OTHELLO_ROOT = (Path(__file__).parent / "othello_world").resolve()
OTHELLO_MECHINT_ROOT = (OTHELLO_ROOT / "mechanistic_interpretability").resolve()

if not OTHELLO_ROOT.exists():
    os.system("git clone https://github.com/likenneth/othello_world")
    os.

# %%
from othello_world.mechanistic_interpretability.mech_interp_othello_utils import (
    plot_board,
    plot_single_board,
    plot_board_log_probs,
    to_string,
    to_int,
    int_to_label,
    string_to_label,
    OthelloBoardState,
)

# %%
full_games_tokens: Int[Tensor, "games=100000 moves=60"] = t.tensor(np.load(
    OTHELLO_MECHINT_ROOT / "board_seqs_int_small.npy"),
                                                                   dtype=t.long)
"""A tensor of shape `(num_games, length_of_game)` containing the board state at
each move, as an integer from 0 to 60.  Suitable for input into the model.
0 corresponds to 'pass' and is not used."""

# Load board data as "strings" (i.e. 0 to 63 with middle squares skipped out)
full_games_board_index: Int[Tensor, "games=100000 moves=60"] = t.tensor(np.load(
    OTHELLO_MECHINT_ROOT / "board_seqs_string_small.npy"),
                                                                        dtype=t.long)
"""A tensor of shape `(num_games, length_of_game)` containing the board state at
each move, as an integer from 0 to 63 with the middle squares skipped out.
Suitable for ???."""

assert all([middle_sq not in full_games_board_index for middle_sq in [27, 28, 35, 36]])
assert full_games_tokens.max() == 60
assert full_games_tokens.min() == 1

num_games, length_of_game = full_games_tokens.shape
print("Number of games:", num_games)
print("Length of game:", length_of_game)

# %%
# Define possible indices (excluding the four center squares)
tokens_to_board = [i for i in range(64) if i not in [27, 28, 35, 36]]


def to_board_label(i: int) -> str:
    """Convert an index into a board label, e.g. `E2`. 0 ≤ i < 64"""
    letter = "ABCDEFGH"[i // 8]
    return f"{letter}{i%8}"


# Get our list of board labels
board_labels = list(map(to_board_label, tokens_to_board))
full_board_labels = list(map(to_board_label, range(64)))


# %%
def logits_to_board(logits: Float[Tensor, "... 61"]) -> Float[Tensor, "... rows=8 cols=8"]:
    """Convert a set of logits into a board state, with each cell being a log prob of that cell being played."""
    log_probs = logits.log_softmax(-1)
    # Remove the "pass" move (the zeroth vocab item)
    log_probs = log_probs[..., 1:]
    assert log_probs.shape[-1] == 60

    extra_shape = log_probs.shape[:-1]
    # Set all cells to -13 by default, for a very negative log prob - this means the middle cells don't show up as mattering
    temp_board_state = (t.zeros((*extra_shape, 64), dtype=t.float32, device=device) - 13.0)
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

num_games = 100
focus_games_tokens = full_games_tokens[:num_games]
focus_games_board_index = full_games_board_index[:num_games]


# %%
def one_hot(list_of_ints: List[int], num_classes=64):
    """Encode a list of ints into a one-hot vector of length `num_classes`"""
    out = t.zeros((num_classes, ), dtype=t.float32)
    out[list_of_ints] = 1.0
    return out


def move_sequence_to_state(
    moves_board_index: Int[Tensor, "batch moves"],
    only_valid: Literal[True] = False,
) -> Float[Tensor, "batch moves rows=8 cols=8"]:
    """Convert sequences of moves into a sequence of board states.
    Moves are encoded as integers from 0 to 63.
    If `only_valid` is True, then the board state is a one-hot encoding of the valid moves
    otherwise, white=-1, empty=0, black=1.
    """
    assert len(moves_board_index.shape) == 2

    states = t.zeros((*moves_board_index.shape, 8, 8), dtype=t.float32)
    for b, moves in enumerate(moves_board_index):
        board = OthelloBoardState()
        for m, move in enumerate(moves):
            board.umpire(move.item())
            if only_valid:
                states[b, m] = one_hot(board.get_valid_moves()).reshape(8, 8)
            else:
                states[b, m] = t.tensor(board.state)
    return states


focus_states = move_sequence_to_state(focus_games_board_index)
focus_valid_moves = move_sequence_to_state(focus_games_board_index, only_valid=True)

print("focus states:", focus_states.shape)
print("focus_valid_moves", focus_valid_moves.shape)

# %%
imshow(
    focus_states[0, :16],
    facet_col=0,
    facet_col_wrap=8,
    facet_labels=[f"Move {i}" for i in range(1, 17)],
    title="First 16 moves of first game",
    color_continuous_scale="Greys",
)

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


# %%

# We first convert the board states to be in terms of my (+1) and their (-1), rather than black and white
alternating = t.tensor([-1 if i % 2 == 0 else 1 for i in range(focus_games_tokens.shape[1])])
flipped_focus_states = focus_states * alternating[None, :, None, None]

# We now convert to one-hot encoded vectors
focus_states_flipped_one_hot = state_stack_to_one_hot(flipped_focus_states)

# Take the argmax (i.e. the index of option empty/their/mine)
focus_states_flipped_value = focus_states_flipped_one_hot.argmax(dim=-1).to(device)

# %%
# Zero ablation of every head and MLP
# We try to see how much the overall loss of the model increases when ablating each component
# First, we evalute the loss only on the focus games


def get_loss(
    model: HookedTransformer,
    games_token: Int[Tensor, "batch game_len rows cols"],
    games_board_index: Int[Tensor, "batch game_len"],
    move_start: int = 5,
    move_end: int = -5,
):
    # This is the input to our model
    assert isinstance(games_token, Int[Tensor, f"batch full_game_len=60"])

    valid_moves = move_sequence_to_state(games_board_index, only_valid=True)
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


loss, accuracy = get_loss(model, focus_games_tokens, focus_games_board_index)
# %%
# Now, we find the loss but after ablating each head


def zero_ablation_hook(activation: Tensor, hook: HookPoint, head: Optional[int] = None):
    if head is not None:
        assert activation.shape[2] == model.cfg.n_heads
        activation[:, :, head] = 0
    else:
        activation.fill_(0)


def zero_ablation(
    model: HookedTransformer,
    metrics: Callable[[HookedTransformer], Union[float, Float[Tensor, "n_metrics"]]],
    individual_heads: bool = False,
    substract_base: bool = False,
) -> Float[Tensor, "n_metrics n_layers n_components"]:
    """
    Compute the given metrics after ablating each multihead and MLP of the model.
    If `individual_heads` is True, then we ablate each head individually.
    If `substract_base` is True, then we substract the base metrics from the ablated metrics.
    n_components is 2 if `individual_heads` is False, and `model.cfg.n_heads + 2` otherwise.

    The order of the components is:
    - first `model.cfg.n_heads` are the individual heads (if `individual_heads` is True)
    - then the whole attention layer
    - then the MLP
    """
    base_metrics = metrics(model)
    assert isinstance(base_metrics, (Float[Tensor, "n_metrics"], float))
    n_metrics = 1 if isinstance(base_metrics, float) else base_metrics.shape[0]

    comp_per_layer = model.cfg.n_heads + 2 if individual_heads else 2
    record = t.zeros((n_metrics, model.cfg.n_layers, comp_per_layer))

    # Compute all ablations that we want to do
    hooks = {}
    for layer in range(model.cfg.n_layers):
        hooks[layer, -1] = (utils.get_act_name("mlp_out", layer), zero_ablation_hook)
        hooks[layer, -2] = (utils.get_act_name("attn_out", layer), zero_ablation_hook)

        if individual_heads:
            for head in range(model.cfg.n_heads):
                hooks[layer, head] = (
                    utils.get_act_name("v", layer),
                    partial(zero_ablation_hook, head=head),
                )

    for (layer, component), hook in tqdm(hooks.items()):
        with model.hooks(fwd_hooks=[hook]):
            record[:, layer, component] = metrics(model)

    if substract_base:
        if isinstance(base_metrics, float):
            record -= base_metrics
        else:
            record -= base_metrics[:, None, None]

    return record


# %%
individual_heads = True
get_metrics = lambda model: get_loss(model, focus_games_tokens, focus_games_board_index, 0, -1)
metrics = zero_ablation(model, get_metrics, individual_heads)
# %%
base_metrics = get_metrics(model)
# %%
# Plotting the results
x = [f"Head {i}" for i in range(model.cfg.n_heads)] + ["MLP"]
y = [f"Layer {i}" for i in range(model.cfg.n_layers)]
if not individual_heads:
    x = x[-2:]

imshow(
    metrics[0] - base_metrics[0],
    title="Loss after zeroing each component",
    x=x,
    y=y,
)
imshow(
    metrics[1] - base_metrics[1],
    title="Accuracy after zeroing each component",
    x=x,
    y=y,
)

# %% Are Attention heads even useful?

# Abblate all attention after the first layer
for start_layer in range(model.cfg.n_layers):

    def filter(name: str):
        if not name.startswith("blocks."):
            # 'hook_embed' or 'hook_pos_embed' or 'ln_final.hook_scale' or 'ln_final.hook_normalized'
            return False

        layer = int(name.split(".")[1])

        return layer >= start_layer and "attn_out" in name

    with model.hooks(fwd_hooks=[(filter, zero_ablation_hook)]):
        metrics = get_metrics(model)
        print(f"Layer {start_layer} ablation:", metrics)

# %% Find dataset example of when the model makes mistakes

t.cuda.empty_cache()
print("GPU memory:", t.cuda.memory_allocated() / 1e9, "GB")
# %%
# Find dataset examples where the model makes mistakes
# %%
