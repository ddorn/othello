# %%
import os
import random
from dataclasses import dataclass
from functools import cached_property, partial, wraps
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple, Union
import joblib

import plotly.express as px
import plotly.graph_objects as go
import einops
import numpy as np
import torch as t
import torch.nn.functional as F
import transformer_lens.utils as utils
from IPython.display import display
from jaxtyping import Bool, Float, Int
from torch import Tensor
from tqdm.notebook import tqdm, trange
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint

from plotly_utils import imshow

OTHELLO_ROOT = (Path(__file__).parent / "othello_world").resolve()
OTHELLO_MECHINT_ROOT = (OTHELLO_ROOT / "mechanistic_interpretability").resolve()

if not OTHELLO_ROOT.exists():
    os.system("git clone https://github.com/likenneth/othello_world")

from othello_world.mechanistic_interpretability.mech_interp_othello_utils import (
    OthelloBoardState, )

from plotting import plot_square_as_board

# Conversion methods

# %%
TOKENS_TO_BOARD = t.tensor([-100] + [i for i in range(64) if i not in [27, 28, 35, 36]])
BOARD_TO_TOKENS = t.zeros(64, dtype=t.long) - 100
BOARD_TO_TOKENS[TOKENS_TO_BOARD[1:]] = t.arange(1, 61)


def tokens_to_board(tokens):
    """Map from token index (0 <= t < 60) to board index (0 < b < 64)"""
    return TOKENS_TO_BOARD.to(tokens.device)[tokens]


def board_label_to_row_col(label: str) -> Tuple[int, int]:
    """Converts a board label like 'a0' to a row and column index."""
    assert len(label) == 2
    col = int(label[1])
    row = ord(label[0].lower()) - ord("a")
    assert 0 <= row <= 7
    assert 0 <= col <= 7
    return row, col


def to_board_label(board_index: int) -> str:
    """Convert an index into a board label, e.g. `E2`. 0 ≤ i < 64"""
    row = board_index // 8
    col = board_index % 8
    assert 0 <= row <= 7, f"Expected 0 ≤ row ≤ 7, got {row}"
    assert 0 <= col <= 7, f"Expected 0 ≤ col ≤ 7, got {col}"
    letter = "ABCDEFGH"[row]
    return f"{letter}{col}"


# Get our list of board labels
board_labels = list(map(to_board_label, TOKENS_TO_BOARD[1:]))
"""Map from token index to board label, e.g. `E2`. 0 ≤ i < 60"""
full_board_labels = list(map(to_board_label, range(64)))
"""Map from token index to board label, e.g. `E2`. 0 ≤ i < 64"""


def logits_to_board(
    logits: Float[Tensor, "... 61"],
    mode: Literal["log_prob", "prob", "logits"],
) -> Float[Tensor, "... rows=8 cols=8"]:
    """
    Convert a set of logits into a board state, with each cell being a log prob/prob/logits of that cell being played.

    Returns:
        A tensor of shape `(..., 8, 8)`
    """
    if mode == "log_prob":
        x = logits.log_softmax(-1)
    elif mode == "prob":
        x = logits.softmax(-1)
    elif mode == "logits":
        x = logits
    # Remove the "pass" move (the zeroth vocab item)
    x = x[..., 1:]
    assert x.shape[-1] == 60, f"Expected logits to have 60 items, got {x.shape[-1]}"

    extra_shape = x.shape[:-1]
    temp_board_state = t.zeros((*extra_shape, 64), dtype=t.float32, device=logits.device)
    temp_board_state[..., TOKENS_TO_BOARD[1:]] = x
    # Set all cells to -13 by default, for a very negative log prob - this means the middle cells don't show up as mattering
    if mode == "log_prob":
        temp_board_state[..., [27, 28, 35, 36]] = -13
    return temp_board_state.reshape(*extra_shape, 8, 8)


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


def load_sample_games(
    max_games: int = 100_000,
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
def one_hot(list_of_ints: List[int], num_classes=64) -> Float[Tensor, "num_classes"]:
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

    nb_games = moves_board_index.shape[0]
    if nb_games > 10_000:
        stack = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(move_sequence_to_state)(moves_board_index[i:i + 1000], mode=mode)
            for i in tqdm(
                range(0, nb_games, 1000),
                desc="Converting moves to states",
                unit="1000 games",
            ))
        return t.cat(stack, dim=0)

    if mode == "valid":
        dtype = t.bool
    else:
        dtype = t.float32

    states = t.zeros((*moves_board_index.shape, 8, 8), dtype=dtype)

    if nb_games > 1_000:
        iterator = tqdm(moves_board_index, desc="Converting moves to states")
    else:
        iterator = moves_board_index

    for b, moves in enumerate(iterator):
        board = OthelloBoardState()
        for m, move in enumerate(moves):
            board.umpire(move.item())
            if mode == "valid":
                states[b, m].flatten()[board.get_valid_moves()] = True
            else:
                states[b, m] = t.tensor(board.state)

    if mode == "alternate":
        states = alternate_states(states)
    return states.to(moves_board_index.device)


# %%
def state_stack_to_one_hot(
    state_stack: Float[Tensor, "games moves rows=8 cols=8"]
) -> Bool[Tensor, "games moves rows=8 cols=8 options=3"]:
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
        dtype=t.bool,
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
    data = [[0 if c == "." else 1 if c == "x" else -1 for c in line] for line in lines]
    return t.tensor(data, dtype=t.int8)


# %%


@dataclass
class Metrics:
    loss: float
    """KL divergence between predictions and uniform distribution over valid moves"""
    cell_accuracy: float
    """Frequency of a cell being correct"""
    board_accuracy: float
    """Frenquency of the whole board being correct"""

    true_positives: float
    """The number of true positives (moves that were predicted to be valid and were valid)"""
    true_negatives: float
    """The number of true negatives (moves that were predicted to be invalid and were invalid)"""
    false_positives: float
    """The number of false positives (moves that were predicted to be valid but were not)"""
    false_negatives: float
    """The number of false negatives (moves that were predicted to be invalid but were valid)"""

    def __str__(self) -> str:
        return ("Metrics:\n"
                f"Loss: {self.loss:.4f}\n"
                # Acuracy is in percentage (format: 99.99%)
                f"Cell accuracy: {self.cell_accuracy:.2%}\n"
                f"Board accuracy: {self.board_accuracy:.2%}\n"
                f"True positives: {self.true_positives:.2%}\n"
                f"True negatives: {self.true_negatives:.2%}\n"
                f"False positives: {self.false_positives:.2%}\n"
                f"False negatives: {self.false_negatives:.2%}\n")

    def to_tensor(self) -> Float[Tensor, "metrics=7"]:
        return t.tensor([
            self.loss,
            self.cell_accuracy,
            self.board_accuracy,
            self.true_positives,
            self.true_negatives,
            self.false_positives,
            self.false_negatives,
        ])


@t.inference_mode()
def get_loss(
    model: HookedTransformer,
    games_token: Int[Tensor, "batch game_len rows cols"],
    games_board_index: Int[Tensor, "batch game_len"],
    move_start: int = 5,
    move_end: int = -5,
) -> Metrics:
    """Get the loss of the model on the given games.

    Args:
        model (HookedTransformer): the model to evaluate
        games_token (Int[Tensor, "batch game_len rows cols"]): the tokenized games, integers between 0 and 60
        games_board_index (Int[Tensor, "batch game_len"]): the board index of the games, integers between 0 and 64
        move_start (int, optional): The first move to consider. Defaults to 5.
        move_end (int, optional): The last move to consider. Defaults to -5.

    Returns:
        Metrics: the metrics of the model (see the Metrics class)
    """

    # This is the input to our model
    assert isinstance(games_token, Int[Tensor, "batch full_game_len=60"])
    # Compare the devices robustly
    if games_token.device == t.device("cpu") and model.cfg.device != t.device("cpu"):
        print(
            f"Warning: games_token.device ({games_token.device}) != model.cfg.device ({model.cfg.device}). Moving to model.cfg.device."
        )

    valid_moves = move_sequence_to_state(games_board_index, "valid")
    valid_moves = valid_moves[:, move_start:move_end].to(model.cfg.device)
    # print("valid moves:", valid_moves.shape)
    assert isinstance(valid_moves, Bool[Tensor, "batch game_len rows=8 cols=8"])

    logits = model(games_token[:, :move_end])[:, move_start:]
    # print("model output:", logits.shape)
    log_probs = logits_to_board(logits, "log_prob")
    # print("logit as board:", logits_as_board.shape)

    # Flatten the last 2 dimensions to have a 64-dim vector instead of 8x8
    log_probs = einops.rearrange(
        log_probs,
        "batch move row col -> batch move (row col)",
    )
    valid_moves = einops.rearrange(
        valid_moves,
        "batch move row col -> batch move (row col)",
    )

    loss = F.kl_div(log_probs, valid_moves.float())
    # loss = log_probs * valid_moves
    # print("loss:", loss.shape, loss)
    # loss = -loss.sum(dim=-1).mean()

    # Compute accuracy
    nb_valid_moves = valid_moves.sum(dim=-1, keepdim=True)
    predicted = log_probs.softmax(dim=-1) > 1 / (2 * nb_valid_moves)
    correct = predicted == valid_moves
    cell_accuracy = correct.float().mean()
    board_accuracy = correct.all(dim=-1).float().mean()

    # Compute false positive, false negative, true positive, true negative
    valid_moves = valid_moves.bool()
    false_positive = (predicted & ~valid_moves).float().mean()
    false_negative = (~predicted & valid_moves).float().mean()
    true_positive = (predicted & valid_moves).float().mean()
    true_negative = (~predicted & ~valid_moves).float().mean()

    return Metrics(
        loss=loss.item(),
        cell_accuracy=cell_accuracy.item(),
        board_accuracy=board_accuracy.item(),
        true_positives=true_positive.item(),
        true_negatives=true_negative.item(),
        false_positives=false_positive.item(),
        false_negatives=false_negative.item(),
    )


# %%


@t.inference_mode()
def get_probe_outputs(
    model: HookedTransformer,
    probe: Float[Tensor, "d_model rows cols options"],
    tokens: Int[Tensor, "game move"],
    layer: int = 6,
) -> Float[Tensor, "game min(move,59) rows cols options"]:
    # Make sure all tensors / models are on cpu, if available
    if t.cuda.is_available():
        model = model.cuda()
        probe = probe.cuda()
        tokens = tokens.cuda()

    act_name = utils.get_act_name("resid_post", layer)

    _, cache = model.run_with_cache(
        tokens[:, :59],
        names_filter=lambda name: name == act_name,
    )

    probe_out = einops.einsum(
        cache[act_name],
        probe,
        "game move d_model, d_model row col options -> game move row col options",
    )
    return probe_out


@t.inference_mode()
def plot_aggregate_metric(
    tokens: Int[Tensor, "batch game_len"],
    model: HookedTransformer,
    probe: Optional[Float[Tensor, "d_model rows cols options"]] = None,
    pos_start: int = 5,
    pos_end: int = -5,
    layer: int = 6,
    per_option: bool = False,
    per_move: Literal["no", "board_accuracy", "cell_accuracy"] = "no",
    name: str = "probe",
    # options_stats: Optional[Float[Tensor, "stat=3 move row col"]] = None,
    prediction: Literal["argmax", "softmax", 'logprob'] = "argmax",
    black_and_white: bool = False,
    plot: bool = True,
) -> Float[Tensor, "rows cols *options"]:
    """
    Compute and plot an aggregate metric for the given probe.

    Args:
        tokens (Int[Tensor, "batch game_len"]): the tokenized games on which to evaluate, integers between 0 and 60
        model (HookedTransformer): the model that is probed
        probe (Float[Tensor, "d_model rows cols options"]): the probe to use.
        pos_start (int, optional): The first move to consider. Defaults to 5.
        pos_end (int, optional): The last move to consider. Defaults to -5.
        layer (int, optional): The layer of the model to probe. Defaults to 6.
        per_option (bool, optional): If True, plot the accuracy for blank/mine/theirs separately.
        per_move (Literal["no", "board_accuracy", "cell_accuracy"], optional): Plot the board or cell accuracy for each move. If 'no', plots per cell.
        name (str, optional): The name of the probe for the plot.
        prediction (Literal["argmax", "softmax", 'logprob'], optional): What to plot. 'argmax' plots the accuracy of the argmax, 'softmax' plots the accuracy of the softmax, and 'logprob' plots the log probability of the correct option.
        black_and_white (bool, optional): If True, consider black/white instead of mine/theirs.
        plot (bool, optional): If False, do not plot the results.
    """

    # Compute output
    out = get_probe_outputs(model, probe, tokens, layer)

    # Compute the expected states
    mode = 'normal' if black_and_white else 'alternate'
    states = move_sequence_to_state(tokens_to_board(tokens), mode)
    correct_one_hot = state_stack_to_one_hot(states.to(model.cfg.device))
    correct_one_hot: Bool[Tensor, "game move row col options"]

    # Remove the first and last moves
    pos_end = pos_end % tokens.shape[1]
    out = out[:, pos_start:pos_end]
    correct_one_hot = correct_one_hot[:, pos_start:pos_end]

    # Transform the probe output into the plotted metric
    if prediction == 'argmax':
        probe_values = out.argmax(dim=-1, keepdim=True)
        metric = t.zeros_like(out)
        metric.scatter_(-1, probe_values, 1)
        # correct = (probe_one_hot == states_one_hot).float()
    elif prediction == 'softmax':
        metric = out.softmax(dim=-1)
        # correct[states_one_hot] = probs[states_one_hot]
        # correct[~states_one_hot] = 1 - probs[~states_one_hot]
    elif prediction == 'logprob':
        metric = out.log_softmax(dim=-1)
    else:
        raise ValueError(f"Unknown prediction mode: {prediction}")

    if False:
        # if options_stats is not None:
        assert (options_stats.shape[0] == 3
                ), f"options_stats should have 3 stats, got {options_stats.shape}"

        # Compute weighted accuracy
        weight = einops.rearrange(
            options_stats[:, pos_start:pos_end].to(model.cfg.device),
            "stat move row col -> move row col stat",
        )
        coeff = 1 / weight
        coeff = coeff / coeff.sum(dim=-1, keepdim=True)
        # remove Nan
        coeff[~t.isfinite(coeff)] = 1
        metric = metric * coeff

    metric[~correct_one_hot] = 0
    metric = metric.sum(dim=-1)
    metric: Float[Tensor, "game move row col"]

    # Reduce the metric to the plot we want
    if per_move == 'no':
        # Mean over games and moves
        metric = metric.mean(dim=(0, 1))
    elif per_move == "board_accuracy":
        # Reduce the cell (row+col) dimension
        if prediction == 'logprob':
            # Log prob is additive. It annoys me to make a distinction here
            # but I don't see an other way
            metric = metric.sum((2, 3))
        else:
            # Other metrics are multiplicative
            metric = metric.prod(3).prod(2)
        # Mean over games
        metric = metric.mean(dim=0)
    elif per_move == "cell_accuracy":
        # Mean over games and cells
        metric = metric.mean(dim=(0, 2, 3))
    else:
        raise ValueError(f"Unknown per_move: {per_move}")

    # -------------------------------------------------
    # We computed what we want to plot. Now we plot it.
    # -------------------------------------------------
    if not plot:
        return metric

    # Compute the title of the plot, using `name`, `prediction`, and `per_move`
    prediction_nice = {
        'softmax': 'Average probability',
        'logprob': 'Average log probability',
        'argmax': 'Average argmax',
    }[prediction]
    what = {
        'no': 'each cell being wrong',
        'board_accuracy': 'the whole board being correct',
        'cell_accuracy': 'each cell being correct',
    }[per_move]
    title = f"{prediction_nice} of {what} for {name}"

    if per_move == "no":  # Per cell
        plot_square_as_board(1 - metric, diverging_scale=False, title=title)
    else:  # Per move
        if metric.ndim == 1:
            # If 'board_accuracy' and not per_option
            metric = metric.unsqueeze(1)
        labels = ["Correct"]
        fig = go.Figure()
        for line, label in zip(metric.T, labels):
            fig.add_trace(go.Scatter(
                x=t.arange(pos_start, pos_end),
                y=line.cpu(),
                name=label,
            ))
        fig.update_layout(
            title=title,
            xaxis_title="Move",
            yaxis_title="Accuracy",
            legend_title="Option",
        )
        fig.show()

    return metric

def zero_ablation_hook(activation: Tensor, hook: HookPoint, head: Optional[int] = None):
    if head is not None:
        activation[:, :, head] = 0
    else:
        activation.fill_(0)


@t.inference_mode()
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
    record = t.zeros((n_metrics, model.cfg.n_layers, comp_per_layer), device=model.cfg.device)

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
# %%


def generate_random_game(
) -> (Tuple[Int[Tensor, "moves=60"], Bool[Tensor, "moves=60 rows=8 cols=8"]]):
    """Generate a random game of othello by choosing valid moves uniformly at random.

    Returns:
        tokens: a sequence of tokens, where each move an integer between 1 and 59 (inclusive)
        valid_moves: the valid moves for each move, as a one-hot encoding of the board state
    """
    state = OthelloBoardState()
    moves = []
    valid_moves = t.zeros(60, 64, dtype=t.bool)
    for i in range(60):
        possible = state.get_valid_moves()

        if not possible:  # no valid move before a game of length 60
            return generate_random_game()

        valid_moves[i, possible] = True
        move = random.choice(possible)
        moves.append(move)
        state.umpire(move)

    return BOARD_TO_TOKENS[moves], valid_moves.reshape(60, 8, 8)


def generate_training_data(
    n: int = 100,
    seed: int = 42
) -> Tuple[Int[Tensor, "n_games moves=60"], Bool[Tensor, "n_games moves=60 rows=8 cols=8"]]:
    """Generate training data.

    Args:
        n (int, optional): The number of games to generate.
        seed (int, optional): The seed to use for the random number generator.

    Returns:
        tokens: sequences of tokens, where each move an integer between 1 and 59 (inclusive)
        valid_moves: the valid moves for each move, as a one-hot encoding of the board state
    """
    random.seed(seed)
    training_data = joblib.Parallel(n_jobs=-1)(joblib.delayed(generate_random_game)()
                                               for _ in trange(n))

    games_board_index = t.stack([game for game, _ in training_data])
    games_valid_moves = t.stack([valid_moves for _, valid_moves in training_data])

    return games_board_index, games_valid_moves


# %% Save the training data
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
GAME_TOKENS_PATH = DATA_DIR / "games_tokens.pt"
GAME_VALID_MOVES_PATH = DATA_DIR / "games_valid_moves.pt"


def make_training_data(
    n: int = 100,
    seed: int = 42,
) -> (Tuple[Int[Tensor, "n_games moves=60"], Bool[Tensor, "n_games moves=60 rows=8 cols=8"]]):
    """Generate the training data and save it to disk.

    Returns:
        moves: a sequence of moves, where each move an integer between 0 and 63 (inclusive)
        valid_moves: the valid moves for each move, as a one-hot encoding of the board state
    """
    games_tokens, games_valid_moves = generate_training_data(n, seed)

    for path, data in [
        (GAME_TOKENS_PATH, games_tokens),
        (GAME_VALID_MOVES_PATH, games_valid_moves),
    ]:
        if not path.exists():
            t.save(data, path)
            name = path.stem
            print(f"Saved {name} to {path.resolve()}")
        else:
            print(f"Warning: {path.resolve()} already exists. Not overwriting!")

    return games_tokens, games_valid_moves


# %%
def get_training_data(
) -> (Tuple[Int[Tensor, "n_games moves=60"], Float[Tensor, "n_games moves=60 rows=8 cols=8"]]):
    """Load the training data.

    Returns:
        moves: a sequence of moves, where each move an integer between 0 and 63 (inclusive)
        valid_moves: the valid moves for each move, as a one-hot encoding of the board state
    """
    return t.load(GAME_TOKENS_PATH), t.load(GAME_VALID_MOVES_PATH)


# %%

STATS_PATH = DATA_DIR / "stats.pt"


def compute_stats(
    games_states: Int[Tensor, "games moves rows cols"],
    games_valid_moves: Optional[Bool[Tensor, "games moves rows cols"]] = None,
    n_games: int = 0,
) -> Float[Tensor, "stat cell row col"]:
    """Compute statistics about the training data
    Per move and per position:
    - Frequency of the cell being empty
    - Frequency of the cell being occupied by my piece
    - Frequency of the cell being occupied by the opponent's piece
    - Frequency of the cell being a valid move (if `games_valid_moves` is provided)
    """

    n_games = n_games % len(games_states)
    if n_games == 0:
        n_games = len(games_states)

    n_stats = 3 + (games_valid_moves is not None)
    if games_valid_moves is None:
        games_valid_moves = [None] * n_games
    else:
        assert games_valid_moves.device == games_states.device, (
            f"games_valid_moves.device ({games_valid_moves.device}) != games_states.device ({games_states.device})"
        )

    stats = t.zeros(n_stats, 60, 8, 8, device=games_states.device)
    for board_states, valid_moves in tqdm(
            zip(games_states[:n_games], games_valid_moves[:n_games]),
            desc="Computing stats",
            total=n_games,
    ):
        game_stats = [
            board_states == 0,  # empty
            board_states == 1,  # my piece
            board_states == -1,  # their piece
        ]
        if valid_moves is not None:
            game_stats.append(valid_moves == 1)

        stats += t.stack(game_stats)
    stats /= n_games

    if not STATS_PATH.exists():
        t.save(stats, STATS_PATH)
        print(f"Saved stats to {STATS_PATH.resolve()}")
    else:
        print(f"Warning: {STATS_PATH.resolve()} already exists. Not saving the stats.")

    return stats


# %%
def valid_moves_from_board(board_state: Int[Tensor, "row col"], move_index: int) -> List[int]:
    """Get the valid moves from the board state at the given move index (to indicate the next player)

    move_index is 1-based, so 1 is the first move, 2 is the second move, etc.
    It corresponds to the number of moves that have been played (same as game[:move_index])
    """
    # np_board_state = move_sequence_to_state(tokens_to_board(orig_games), mode="normal")[0, -1].numpy()
    # print(np_board_state)
    state = OthelloBoardState()
    state.state = utils.to_numpy(board_state)
    # -1 or -1. First move is +1
    state.next_hand_color = ((move_index + 1) % 2) * 2 - 1
    return state.get_valid_moves()


# %%
def swap_subspace(
    original: Float[Tensor, "batch d_model"],
    patch: Float[Tensor, "batch d_model"],
    subspace: Float[Tensor, "... d_model"],
    make_orthonormal_basis: bool = True,
) -> Float[Tensor, "batch d_model"]:
    """
    Swap the subspace of the original with the subspace of the patch.

    Args:
        original: the original embedding
        patch: the patch to apply
        subspace: and orthonormal basis of the subspace to swap (except if make_orthogonal_basis is True)
        make_orthonormal_basis: if True, the subspace is made orthonormal before being used.
            This needs to be True if the subspace is not an orthonormal basis, or the result will be wrong.

    Returns:
        original ⊥ subspace + (patch - (patch ⊥ subspace))
    """

    # Step 0. Find a basis of the subspace of the probe
    subspace = subspace.flatten(end_dim=-2)
    if make_orthonormal_basis:
        # normalize the vectors
        subspace = subspace / subspace.norm(dim=-1, keepdim=True)
        # Find the basis using SVD
        _, _, subspace = t.linalg.svd(subspace, full_matrices=False)

    # Step 1. Remove the components that are in the space of the probe
    coefficients = einops.einsum(original, subspace, "batch d_model, dir d_model -> batch dir")
    original = original - einops.einsum(coefficients, subspace,
                                        "batch dir, dir d_model -> batch d_model")

    # 2. Add the component of the from the new_cache
    coefficients = einops.einsum(patch, subspace, "batch d_model, dir d_model -> batch dir")
    return original + einops.einsum(subspace, coefficients,
                                    "dir d_model, batch dir -> batch d_model")


# %%
