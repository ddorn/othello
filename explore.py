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

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part6_othellogpt"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from plotly_utils import imshow
from neel_plotly import scatter, line

# import part6_othellogpt.tests as tests

device = t.device("cuda" if t.cuda.is_available() else "cpu")

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
os.chdir(section_dir)

OTHELLO_ROOT = (section_dir / "othello_world").resolve()
OTHELLO_MECHINT_ROOT = (OTHELLO_ROOT / "mechanistic_interpretability").resolve()

if not OTHELLO_ROOT.exists():
    #!git clone https://github.com/likenneth/othello_world
    pass

sys.path.append(str(OTHELLO_MECHINT_ROOT))
# %%
from mech_interp_othello_utils import (
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

# ---------- START of not updated

PROBE_STUFF = False
if PROBE_STUFF:
    focus_logits, focus_cache = model.run_with_cache(focus_games_tokens[:, :-1].to(device))
    focus_logits.shape

    full_linear_probe: Float[Tensor, "mode=3 d_model rows=8 cols=8 options=3"] = t.load(
        OTHELLO_MECHINT_ROOT / "main_linear_probe.pth", map_location=device)

    rows = 8
    cols = 8
    options = 3
    assert full_linear_probe.shape == (3, cfg.d_model, rows, cols, options)

    black_to_play_index = 0
    white_to_play_index = 1
    blank_index = 0
    their_index = 1
    my_index = 2

    # Creating values for linear probe (converting the "black/white to play" notation into "me/them to play")
    linear_probe = t.zeros(cfg.d_model, rows, cols, options, device=device)
    linear_probe[..., blank_index] = 0.5 * (full_linear_probe[black_to_play_index, ..., 0] +
                                            full_linear_probe[white_to_play_index, ..., 0])
    linear_probe[..., their_index] = 0.5 * (full_linear_probe[black_to_play_index, ..., 1] +
                                            full_linear_probe[white_to_play_index, ..., 2])
    linear_probe[..., my_index] = 0.5 * (full_linear_probe[black_to_play_index, ..., 2] +
                                         full_linear_probe[white_to_play_index, ..., 1])

    layer = 6
    game_index = 9
    move = 41

    def plot_probe_outputs(layer, game_index, move, **kwargs):
        residual_stream = focus_cache["resid_post", layer][game_index, move]
        # print("residual_stream", residual_stream.shape)
        probe_out = einops.einsum(
            residual_stream,
            linear_probe,
            "d_model, d_model row col options -> row col options",
        )
        probabilities = probe_out.softmax(dim=-1)
        plot_square_as_board(
            probabilities,
            facet_col=2,
            facet_labels=["P(Empty)", "P(Their's)", "P(Mine)"],
            **kwargs,
        )

    plot_probe_outputs(
        layer,
        game_index,
        move,
        title="Example probe outputs after move 29 (black to play)",
    )

    plot_single_board(int_to_label(focus_games_tokens[game_index, :move + 1]))


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

if PROBE_STUFF:
    probe_out = einops.einsum(
        focus_cache["resid_post", 6],
        linear_probe,
        "game move d_model, d_model row col options -> game move row col options",
    )

    probe_out_value = probe_out.argmax(dim=-1)
    # %%
    correct_middle_odd_answers = (
        probe_out_value.cpu() == focus_states_flipped_value[:, :-1])[:, 5:-5:2]
    accuracies_odd = einops.reduce(correct_middle_odd_answers.float(),
                                   "game move row col -> row col", "mean")

    correct_middle_answers = (probe_out_value == focus_states_flipped_value[:, :-1])[:, 5:-5]
    accuracies = einops.reduce(correct_middle_answers.float(), "game move row col -> row col",
                               "mean")

    plot_square_as_board(
        1 - t.stack([accuracies_odd, accuracies], dim=0),
        title="Average Error Rate of Linear Probe",
        facet_col=0,
        facet_labels=["Black to Play moves", "All Moves"],
        zmax=0.25,
        zmin=-0.25,
    )
    # %%

    # define the `cosine_similarities` tensor, to be plotted
    two_probes = full_linear_probe[:2, ..., 0]
    print(two_probes.shape)
    # %%
    two_probes = two_probes / two_probes.norm(dim=1, keepdim=True)
    cosine_similarities = einops.einsum(
        two_probes,
        two_probes,
        "probe_1 d_model row_1 col_1, probe_2 d_model row_2 col_2 -> probe_1 probe_2 row_1 col_1 row_2 col_2",
    )

    cosine_similarities = einops.rearrange(
        cosine_similarities,
        "probe_1 probe_2 row_1 col_1 row_2 col_2 -> (probe_1 row_1 col_1) (probe_2 row_2 col_2)",
    )
    print(cosine_similarities.shape)
    # %%
    odd_BminusW_probe = full_linear_probe[0, ..., 1] - full_linear_probe[0, ..., 2]
    even_BminusW_probe = full_linear_probe[1, ..., 1] - full_linear_probe[1, ..., 2]
    both_probs = einops.rearrange(
        t.stack([odd_BminusW_probe, even_BminusW_probe], dim=0),
        "modes d_model rows cols -> (modes rows cols) d_model",
    )
    both_probs /= both_probs.norm(dim=-1, keepdim=True)
    cosine_similarities = einops.einsum(
        both_probs,
        both_probs,
        "square_y d_model, square_x d_model -> square_y square_x",
    )

    # %%
    imshow(
        cosine_similarities,
        title="Cosine Sim of B-W Linear Probe Directions by Cell",
        x=[f"{L} (O)" for L in full_board_labels] + [f"{L} (E)" for L in full_board_labels],
        y=[f"{L} (O)" for L in full_board_labels] + [f"{L} (E)" for L in full_board_labels],
    )
    # %%
    blank_probe = (linear_probe[..., blank_index] -
                   (linear_probe[..., my_index] + linear_probe[..., their_index]) / 2)
    my_probe = linear_probe[..., my_index] - linear_probe[..., their_index]

    # tests.test_my_probes(blank_probe, my_probe, linear_probe)
    # %%
    pos = 20
    game_index = 0

    # Plot board state
    moves = focus_games_board_index[game_index, :pos + 1]
    plot_single_board(moves)

    # Plot corresponding model predictions
    state = t.zeros((8, 8), dtype=t.float32, device=device) - 13.0
    state.flatten()[tokens_to_board] = focus_logits[game_index, pos].log_softmax(dim=-1)[1:]
    plot_square_as_board(state, zmax=0, diverging_scale=False, title="Log probs")

    # %%
    moves = focus_games_board_index[game_index, :pos + 1]
    cell_r = 5
    cell_c = 4
    print(f"Flipping the color of cell {'ABCDEFGH'[cell_r]}{cell_c}")

    board = OthelloBoardState()
    board.update(moves.tolist())
    board_state = board.state.copy()
    valid_moves = board.get_valid_moves()
    flipped_board = copy.deepcopy(board)
    flipped_board.state[cell_r, cell_c] *= -1
    flipped_valid_moves = flipped_board.get_valid_moves()

    newly_legal = [string_to_label(move) for move in flipped_valid_moves if move not in valid_moves]
    newly_illegal = [
        string_to_label(move) for move in valid_moves if move not in flipped_valid_moves
    ]
    print("newly_legal", newly_legal)
    print("newly_illegal", newly_illegal)

    # %%
    def apply_scale(
        resid: Float[Tensor, "batch=1 seq d_model"],
        flip_dir: Float[Tensor, "d_model"],
        scale: int,
        pos: int,
    ):
        """
        Returns a version of the residual stream, modified by the amount `scale` in the
        direction `flip_dir` at the sequence position `pos`, in the way described above.
        """
        v = flip_dir / flip_dir.norm()
        alpha = einops.einsum(resid[0, pos], v, "d_model, d_model -> ")
        resid[:, pos] -= (alpha + scale * alpha) * v
        return resid

    # tests.test_apply_scale(apply_scale)

    # %%
    flip_dir = my_probe[:, cell_r, cell_c]

    big_flipped_states_list = []
    layer = 4
    scales = [0, 1, 2, 4, 8, 16]

    # Iterate through scales, generate a new facet plot for each possible scale
    for scale in scales:
        # Hook function which will perform flipping in the "F4 flip direction"
        def flip_hook(resid: Float[Tensor, "batch=1 seq d_model"], hook: HookPoint):
            return apply_scale(resid, flip_dir, scale, pos)

        # Calculate the logits for the board state, with the `flip_hook` intervention
        # (note that we only need to use :pos+1 as input, because of causal attention)
        flipped_logits: Tensor = model.run_with_hooks(
            focus_games_tokens[game_index:game_index + 1, :pos + 1],
            fwd_hooks=[
                (utils.get_act_name("resid_post", layer), flip_hook),
            ],
        ).log_softmax(dim=-1)[0, pos]

        flip_state = t.zeros((64, ), dtype=t.float32, device=device) - 10.0
        flip_state[tokens_to_board] = flipped_logits.log_softmax(dim=-1)[1:]
        big_flipped_states_list.append(flip_state)

    flip_state_big = t.stack(big_flipped_states_list)
    state_big = einops.repeat(state.flatten(), "d -> b d", b=6)
    color = t.zeros((len(scales), 64)) + 0.2
    for s in newly_legal:
        color[:, to_string(s)] = 1
    for s in newly_illegal:
        color[:, to_string(s)] = -1

    scatter(
        y=state_big,
        x=flip_state_big,
        title=f"Original vs Flipped {string_to_label(8*cell_r+cell_c)} at Layer {layer}",
        # labels={"x": "Flipped", "y": "Original"},
        xaxis="Flipped",
        yaxis="Original",
        hover=[f"{r}{c}" for r in "ABCDEFGH" for c in range(8)],
        facet_col=0,
        facet_labels=[f"Translate by {i}x" for i in scales],
        color=color,
        color_name="Newly Legal",
        color_continuous_scale="Geyser",
    )
    # %%
    game_index = 1
    move = 20
    layer = 6

    plot_single_board(focus_games_board_index[game_index, :move + 1])
    plot_probe_outputs(layer, game_index, move)

    # %%
    def plot_contributions(contributions, component: str):
        imshow(
            contributions,
            facet_col=0,
            y=list("ABCDEFGH"),
            facet_labels=[f"Layer {i}" for i in range(7)],
            title=f"{component} Layer Contributions to my vs their (Game {game_index} Move {move})",
            aspect="equal",
            width=1400,
            height=350,
        )

    def calculate_attn_and_mlp_probe_score_contributions(
        focus_cache: ActivationCache,
        my_probe: Float[Tensor, "d_model rows cols"],
        layer: int,
        game_index: int,
        move: int,
    ) -> Tuple[Float[Tensor, "layers rows cols"], Float[Tensor, "layers rows cols"]]:
        attn = t.stack(
            [focus_cache["attn_out", layr][game_index, move] for layr in range(layer + 1)])
        mlp = t.stack([focus_cache["mlp_out", layr][game_index, move] for layr in range(layer + 1)])

        attn_contributions = einops.einsum(attn, my_probe,
                                           "layers d_model, d_model rows cols -> layers rows cols")
        mlp_contributions = einops.einsum(mlp, my_probe,
                                          "layers d_model, d_model rows cols -> layers rows cols")

        return attn_contributions, mlp_contributions

    (
        attn_contributions,
        mlp_contributions,
    ) = calculate_attn_and_mlp_probe_score_contributions(focus_cache, my_probe, layer, game_index,
                                                         move)

    plot_contributions(attn_contributions, "Attention")
    plot_contributions(mlp_contributions, "MLP")

    # %%
    def calculate_accumulated_probe_score(
        focus_cache: ActivationCache,
        my_probe: Float[Tensor, "d_model rows cols"],
        layer: int,
        game_index: int,
        move: int,
    ) -> Float[Tensor, "rows cols"]:
        return einops.einsum(
            focus_cache["resid_post", layer][game_index, move],
            my_probe,
            "d_model, d_model rows cols -> rows cols",
        )

    overall_contribution = calculate_accumulated_probe_score(focus_cache, my_probe, layer,
                                                             game_index, move)

    imshow(
        overall_contribution,
        title=
        f"Overall Probe Score after Layer {layer} for<br>my vs their (Game {game_index} Move {move})",
    )
    # %%
    (
        attn_contributions,
        mlp_contributions,
    ) = calculate_attn_and_mlp_probe_score_contributions(focus_cache, blank_probe, layer,
                                                         game_index, move)

    plot_contributions(attn_contributions, "Attention")
    plot_contributions(mlp_contributions, "MLP")

    overall_contribution = calculate_accumulated_probe_score(focus_cache, blank_probe, layer,
                                                             game_index, move)

    imshow(
        overall_contribution,
        title=
        f"Overall Probe Score after Layer {layer} for<br>my vs their (Game {game_index} Move {move})",
    )
    # %%
    # Scale the probes down to be unit norm per cell
    blank_probe_normalised = blank_probe / blank_probe.norm(dim=0, keepdim=True)
    my_probe_normalised = my_probe / my_probe.norm(dim=0, keepdim=True)
    # Set the center blank probes to 0, since they're never blank so the probe is meaningless
    blank_probe_normalised[:, [3, 3, 4, 4], [3, 4, 3, 4]] = 0.0

    # %%
    def get_w_in(
        model: HookedTransformer,
        layer: int,
        neuron: int,
        normalize: bool = False,
    ) -> Float[Tensor, "d_model"]:
        """
        Returns the input weights for the neuron in the list, at each square on the board.

        If normalize is True, the weights are normalized to unit norm.
        """
        w_in = model.W_in[layer, :, neuron].detach().clone()
        if normalize:
            w_in /= w_in.norm(dim=0, keepdim=True)
        return w_in

    def get_w_out(
        model: HookedTransformer,
        layer: int,
        neuron: int,
        normalize: bool = False,
    ) -> Float[Tensor, "d_model"]:
        """
        Returns the input weights for the neuron in the list, at each square on the board.
        """
        w_out = model.W_out[layer, neuron, :].detach().clone()
        if normalize:
            w_out /= w_out.norm(dim=0, keepdim=True)
        return w_out

    def calculate_neuron_input_weights(
        model: HookedTransformer,
        probe: Float[Tensor, "d_model row col"],
        layer: int,
        neuron: int,
    ) -> Float[Tensor, "rows cols"]:
        """
        Returns tensor of the input weights for each neuron in the list, at each square on the board,
        projected along the corresponding probe directions.

        Assume probe directions are normalized. You should also normalize the model weights.
        """
        w_in = get_w_in(model, layer, neuron, normalize=True)
        return einops.einsum(w_in, probe, "d_model, d_model row col -> row col")

    def calculate_neuron_output_weights(
        model: HookedTransformer,
        probe: Float[Tensor, "d_model row col"],
        layer: int,
        neuron: int,
    ) -> Float[Tensor, "rows cols"]:
        """
        Returns tensor of the output weights for each neuron in the list, at each square on the board,
        projected along the corresponding probe directions.

        Assume probe directions are normalized. You should also normalize the model weights.
        """

        w_out = get_w_out(model, layer, neuron, normalize=True)
        return einops.einsum(w_out, probe, "d_model, d_model row col -> row col")

    # tests.test_calculate_neuron_input_weights(calculate_neuron_input_weights, model)
    # tests.test_calculate_neuron_output_weights(calculate_neuron_output_weights, model)

    # %%
    layer = 5
    neuron = 1427

    w_in_L5N1393_blank = calculate_neuron_input_weights(model, blank_probe_normalised, layer,
                                                        neuron)
    w_in_L5N1393_my = calculate_neuron_input_weights(model, my_probe_normalised, layer, neuron)

    imshow(
        t.stack([w_in_L5N1393_blank, w_in_L5N1393_my]),
        facet_col=0,
        y=[i for i in "ABCDEFGH"],
        title=f"Input weights in terms of the probe for neuron L{layer}N{neuron}",
        facet_labels=["Blank In", "My In"],
        width=750,
    )

    w_in_L5N1393 = get_w_in(model, layer, neuron, normalize=True)
    w_out_L5N1393 = get_w_out(model, layer, neuron, normalize=True)

    U, S, Vh = t.svd(
        t.cat(
            [my_probe.reshape(cfg.d_model, 64),
             blank_probe.reshape(cfg.d_model, 64)],
            dim=1,
        ))

    # Remove the final four dimensions of U, as the 4 center cells are never blank and so the blank probe is meaningless there
    probe_space_basis = U[:, :-4]

    print(
        "Fraction of input weights in probe basis:",
        (w_in_L5N1393 @ probe_space_basis).norm().item()**2,
    )
    print(
        "Fraction of output weights in probe basis:",
        (w_out_L5N1393 @ probe_space_basis).norm().item()**2,
    )

    # %%
    def kurtosis(x: Tensor, reduced_axes, fisher=True):
        """
        Computes the kurtosis of a tensor over specified dimensions.
        """
        return ((
            (x - x.mean(dim=reduced_axes, keepdim=True)) / x.std(dim=reduced_axes, keepdim=True))**
                4).mean(dim=reduced_axes, keepdim=False) - fisher * 3

    # %%
    layer = 4
    top_layer_neurons = einops.reduce(
        focus_cache["post", layer][:, 3:-3],
        "game move neuron -> neuron",
        reduction=kurtosis,
    ).argsort(descending=True)[:10]
    # top_layer_neurons = focus_cache["post", layer][:, 3:-3].std(dim=[0, 1]).argsort(descending=True)[:10]
    # w_in = model.W_in[layer]
    # top_layer_neurons = (model.W_in[layer].T @ probe_space_basis).norm(dim=1).argsort(descending=True)[:10]
    heatmaps_blank = []
    heatmaps_my = []

    for neuron in top_layer_neurons:
        neuron = neuron.item()
        heatmaps_blank.append(
            calculate_neuron_output_weights(model, blank_probe_normalised, layer, neuron))
        heatmaps_my.append(
            calculate_neuron_output_weights(model, my_probe_normalised, layer, neuron))

    imshow(
        t.stack(heatmaps_blank),
        facet_col=0,
        y=[i for i in "ABCDEFGH"],
        title=
        f"Cosine sim of Output weights and the 'blank color' probe for top layer {layer} neurons",
        facet_labels=[f"L{layer}N{n.item()}" for n in top_layer_neurons],
        width=1600,
        height=300,
    )

    imshow(
        t.stack(heatmaps_my),
        facet_col=0,
        y=[i for i in "ABCDEFGH"],
        title=f"Cosine sim of Output weights and the 'my color' probe for top layer {layer} neurons",
        facet_labels=[f"L{layer}N{n.item()}" for n in top_layer_neurons],
        width=1600,
        height=300,
    )

    # %%
    game_index = 4
    move = 20

    plot_single_board(
        focus_games_board_index[game_index, :move + 1],
        title="Original Game (black plays E0)",
    )
    plot_single_board(
        focus_games_board_index[game_index, :move].tolist() + [16],
        title="Corrupted Game (blank plays C0)",
    )
    # %%
    clean_input = focus_games_tokens[game_index, :move + 1].clone()
    corrupted_input = focus_games_tokens[game_index, :move + 1].clone()
    corrupted_input[-1] = to_int("C0")
    print("Clean:     ", ", ".join(int_to_label(corrupted_input)))
    print("Corrupted: ", ", ".join(int_to_label(clean_input)))
    # %%
    clean_logits, clean_cache = model.run_with_cache(clean_input)
    corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_input)

    clean_log_probs = clean_logits.log_softmax(dim=-1)
    corrupted_log_probs = corrupted_logits.log_softmax(dim=-1)
    # %%
    f0_index = to_int("F0")
    clean_f0_log_prob = clean_log_probs[0, -1, f0_index]
    corrupted_f0_log_prob = corrupted_log_probs[0, -1, f0_index]

    print("Clean log prob", clean_f0_log_prob.item())
    print("Corrupted log prob", corrupted_f0_log_prob.item(), "\n")

    def patching_metric(patched_logits: Float[Tensor, "batch=1 seq=21 d_vocab=61"]):
        """
        Function of patched logits, calibrated so that it equals 0 when performance is
        same as on corrupted input, and 1 when performance is same as on clean input.

        Should be linear function of the logits for the F0 token at the final move.
        """
        diff = clean_f0_log_prob - patched_logits[0, -1].log_softmax(dim=-1)[f0_index]
        return 1 - diff / (clean_f0_log_prob - corrupted_f0_log_prob)

    # tests.test_patching_metric(patching_metric, clean_log_probs, corrupted_log_probs)

    # %%
    def patch_final_move_output(
        activation: Float[Tensor, "batch seq d_model"],
        hook: HookPoint,
        clean_cache: ActivationCache,
    ) -> Float[Tensor, "batch seq d_model"]:
        """
        Hook function which patches activations at the final sequence position.

        Note, we only need to patch in the final sequence position, because the
        prior moves in the clean and corrupted input are identical (and this is
        an autoregressive model).
        """
        # cache shape: (batch, seq, d_model)

        activation[:, -1] = clean_cache[hook.name][:, -1]

    def get_act_patch_resid_pre(
        model: HookedTransformer,
        corrupted_input: Float[Tensor, "batch pos"],
        clean_cache: ActivationCache,
        patching_metric: Callable[[Float[Tensor, "batch seq d_model"]], Float[Tensor, ""]],
    ) -> Float[Tensor, "2 n_layers"]:
        """
        Returns an array of results, corresponding to the results of patching at
        each (attn_out, mlp_out) for all layers in the model.
        """

        hook = partial(patch_final_move_output, clean_cache=clean_cache)

        out = t.zeros(2, model.cfg.n_layers)
        for layer in range(model.cfg.n_layers):
            logits = model.run_with_hooks(
                corrupted_input,
                fwd_hooks=[
                    (utils.get_act_name("attn_out", layer), hook),
                ],
            )
            out[0, layer] = patching_metric(logits)
            logits = model.run_with_hooks(
                corrupted_input,
                fwd_hooks=[
                    (utils.get_act_name("mlp_out", layer), hook),
                ],
            )
            out[1, layer] = patching_metric(logits)
        return out

    # %%
    patching_results = get_act_patch_resid_pre(model, corrupted_input, clean_cache, patching_metric)
    line(
        patching_results,
        title="Layer Output Patching Effect on F0 Log Prob",
        line_labels=["attn", "mlp"],
        width=750,
    )

    # %% Training the probe

    imshow(
        focus_states[0, :16],
        facet_col=0,
        facet_col_wrap=8,
        facet_labels=[f"Move {i}" for i in range(1, 17)],
        title="First 16 moves of first game",
        color_continuous_scale="Greys",
    )

    # %%
    @dataclass
    class ProbeTrainingArgs:
        # Which layer, and which positions in a game sequence to probe
        layer: int = 6
        pos_start: int = 5
        pos_end: int = model.cfg.n_ctx - 5
        length: int = pos_end - pos_start
        alternating: Tensor = t.tensor([1 if i % 2 == 0 else -1 for i in range(length)],
                                       device=device)

        # Game state (options are blank/mine/theirs)
        options: int = 3
        rows: int = 8
        cols: int = 8

        # Standard training hyperparams
        max_epochs: int = 8
        num_games: int = 50000

        # Hyperparams for optimizer
        batch_size: int = 256
        lr: float = 1e-4
        betas: Tuple[float, float] = (0.9, 0.99)
        wd: float = 0.01

        # Misc.
        probe_name: str = "main_linear_probe"

        # The first mode is blank or not, the second mode is black moves only or white moves only GIVEN that it is not blank
        modes = 3

        # Code to get randomly initialized probe
        def setup_linear_probe(self, model: HookedTransformer):
            linear_probe = t.randn(
                self.modes,
                model.cfg.d_model,
                self.rows,
                self.cols,
                self.options,
                requires_grad=False,
                device=device,
            ) / np.sqrt(model.cfg.d_model)
            linear_probe.requires_grad = True
            return linear_probe

    def seq_to_state_stack(str_moves):
        board = OthelloBoardState()
        states = []
        for move in str_moves:
            board.umpire(move)
            states.append(np.copy(board.state))
        states = np.stack(states, axis=0)
        return states

    class LitLinearProbe(pl.LightningModule):

        def __init__(self, model: HookedTransformer, args: ProbeTrainingArgs):
            super().__init__()
            self.model = model
            self.args = args
            self.linear_probe: Float[
                Tensor, "mode d_model rows cols options"] = args.setup_linear_probe(model)
            pl.seed_everything(42, workers=True)

        def training_step(self, batch: Int[Tensor, "game_idx"], batch_idx: int) -> t.Tensor:
            games_int = full_games_tokens[batch.cpu()]
            games_str = full_games_board_index[batch.cpu()]
            state_stack = t.stack(
                [t.tensor(seq_to_state_stack(game_str.tolist())) for game_str in games_str])
            state_stack = state_stack[:, self.args.pos_start:self.args.pos_end, :, :]
            state_stack_one_hot = state_stack_to_one_hot(state_stack).to(device)
            batch_size = self.args.batch_size
            game_len = self.args.length

            # games_int = tensor of game sequences, each of length 60
            # This is the input to our model
            assert isinstance(games_int, Int[Tensor, f"batch={batch_size} full_game_len=60"])

            # state_stack_one_hot = tensor of one-hot encoded states for each game
            # We'll multiply this by our probe's estimated log probs along the `options` dimension, to get probe's estimated log probs for the correct option
            assert isinstance(
                state_stack_one_hot,
                Int[
                    Tensor,
                    f"batch={batch_size} game_len={game_len} rows=8 cols=8 options=3",
                ],
            )

            act_name = utils.get_act_name("resid_post", self.args.layer)

            _, cache = self.model.run_with_cache(
                games_int[:, :-1],
                names_filter=lambda name: name == act_name,
            )
            resid_post: Float[
                Tensor,
                "batch moves d_model"] = cache[act_name][:, self.args.pos_start:self.args.pos_end]

            probe_logits = einops.einsum(
                self.linear_probe,
                resid_post,
                "mode d_model rows cols options, batch moves d_model -> mode moves batch rows cols options",
            )
            probe_logprobs = probe_logits.log_softmax(dim=-1)
            probe_loss = -(probe_logprobs * state_stack_one_hot).sum(dim=-1)

            loss_all = probe_loss[0].mean(0).sum()
            loss_black = probe_loss[1, 0::2].mean(0).sum()
            loss_white = probe_loss[2, 1::2].mean(0).sum()
            loss = loss_all + loss_black + loss_white

            self.log("train_loss", loss)
            self.log("train_loss_blank", loss_all)
            self.log("train_loss_black", loss_black)
            self.log("train_loss_white", loss_white)
            return loss

        def train_dataloader(self):
            """
            Returns `games_int` and `state_stack_one_hot` tensors.
            """
            n_indices = self.args.num_games - (self.args.num_games % self.args.batch_size)
            full_train_indices = t.randperm(self.args.num_games)[:n_indices]
            full_train_indices = einops.rearrange(
                full_train_indices,
                "(batch_idx game_idx) -> batch_idx game_idx",
                game_idx=self.args.batch_size,
            )
            return full_train_indices

        def configure_optimizers(self):
            return t.optim.AdamW(
                [self.linear_probe],
                lr=self.args.lr,
                betas=self.args.betas,
                weight_decay=self.args.wd,
            )

            loss_white = probe_loss[2].mean()

    # %%
    # Create the model & training system
    args = ProbeTrainingArgs()
    litmodel = LitLinearProbe(model, args)

    # You can choose either logger
    # logger = CSVLogger(save_dir=os.getcwd() + "/logs", name=args.probe_name)
    logger = WandbLogger(save_dir=os.getcwd() + "/logs", project=args.probe_name)

    # Train the model
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        log_every_n_steps=1,
    )
    trainer.fit(model=litmodel)
    # %%
    wandb.finish()
    # %%
    black_to_play_index = 0
    white_to_play_index = 1
    blank_index = 0
    their_index = 1
    my_index = 2

    # Creating values for linear probe (converting the "black/white to play" notation into "me/them to play")
    my_linear_probe = t.zeros(cfg.d_model, rows, cols, options, device=device)
    my_linear_probe[..., blank_index] = 0.5 * (litmodel.linear_probe[black_to_play_index, ..., 0] +
                                               litmodel.linear_probe[white_to_play_index, ..., 0])
    my_linear_probe[..., their_index] = 0.5 * (litmodel.linear_probe[black_to_play_index, ..., 1] +
                                               litmodel.linear_probe[white_to_play_index, ..., 2])
    my_linear_probe[..., my_index] = 0.5 * (litmodel.linear_probe[black_to_play_index, ..., 2] +
                                            litmodel.linear_probe[white_to_play_index, ..., 1])

    # Getting the probe's output, and then its predictions
    probe_out = einops.einsum(
        focus_cache["resid_post", 6],
        my_linear_probe,
        "game move d_model, d_model row col options -> game move row col options",
    )
    probe_out_value = probe_out.argmax(dim=-1)

    # Getting the correct answers in the odd cases
    correct_middle_odd_answers = (probe_out_value == focus_states_flipped_value[:, :-1])[:, 5:-5:2]
    accuracies_odd = einops.reduce(correct_middle_odd_answers.float(),
                                   "game move row col -> row col", "mean")

    # Getting the correct answers in all cases
    correct_middle_answers = (probe_out_value == focus_states_flipped_value[:, :-1])[:, 5:-5]
    accuracies = einops.reduce(correct_middle_answers.float(), "game move row col -> row col",
                               "mean")

    plot_square_as_board(
        1 - t.stack([accuracies_odd, accuracies], dim=0),
        title="Average Error Rate of Linear Probe",
        facet_col=0,
        facet_labels=["Black to Play moves", "All Moves"],
        zmax=0.25,
        zmin=-0.25,
    )

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
