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
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
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
t.set_grad_enabled(False)
device = "cuda" if t.cuda.is_available() else "cpu"

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

num_games = 50
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

    Output shape: `(batch, moves, rows=8, cols=8)
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
if False:
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


@t.inference_mode()
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
SHOW_ATTENTION = False
if SHOW_ATTENTION:
    _, focus_cache = model.run_with_cache(focus_games_tokens[:, :-1])

# %%
if SHOW_ATTENTION:
    game_idx = 0
    layer = 0
    labels = [to_board_label(focus_games_board_index[game_idx, i]) for i in range(59)]
    attention_patterns(labels, focus_cache['pattern', layer][game_idx])

# %%
RUN_ABLATIONS = False

if RUN_ABLATIONS:
    individual_heads = True
    get_metrics = lambda model: get_loss(model, focus_games_tokens, focus_games_board_index, 0, -1)
    metrics = zero_ablation(model, get_metrics, individual_heads)
    base_metrics = get_metrics(model)
# %%
# Plotting the results
if RUN_ABLATIONS:
    x = [f"Head {i}" for i in range(model.cfg.n_heads)] + ["All Heads", "MLP"]
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
if RUN_ABLATIONS:
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
# Exploration of the probe
full_linear_probe: Float[Tensor, "mode=3 d_model rows=8 cols=8 options=3"] = t.load(
    OTHELLO_MECHINT_ROOT / "main_linear_probe.pth", map_location=device)

blank_index = 0
black_to_play_index = 1
white_to_play_index = 2
their_index = 1
my_index = 2
# (d_model, rows, cols, options)
linear_probe = t.zeros(cfg.d_model, 8, 8, 3, device=device)
"""The linear probe is a tensor of shape (d_model, rows, cols, options) where options are:
- 0: blank
- 1: their piece
- 2: my piece
"""

linear_probe[..., blank_index] = 0.5 * (full_linear_probe[black_to_play_index, ..., 0] +
                                        full_linear_probe[white_to_play_index, ..., 0])
linear_probe[..., their_index] = 0.5 * (full_linear_probe[black_to_play_index, ..., 1] +
                                        full_linear_probe[white_to_play_index, ..., 2])
linear_probe[..., my_index] = 0.5 * (full_linear_probe[black_to_play_index, ..., 2] +
                                     full_linear_probe[white_to_play_index, ..., 1])

# %%
# Looking at the cosine similarity between the vectors of the blank probe
# (and also the other probes)
blank_probe = linear_probe[..., blank_index]
their_probe = linear_probe[..., their_index]
my_probe = linear_probe[..., my_index]

blank_direction = blank_probe - (their_probe + my_probe) / 2
my_direction = my_probe - their_probe

# %%


def plot_similarities(vectors: Float[Tensor, '*n_vectors dim'], **kwargs):
    vectors = vectors.flatten(end_dim=-2)
    sim = einops.einsum(vectors, vectors, "vec_1 dim, vec_2 dim -> vec_1 vec_2")
    imshow(sim, **kwargs)


for probe, name in zip([blank_probe, their_probe, my_probe], ["blank", "their", "my"]):
    probe = einops.rearrange(probe, "d_model rows cols -> (rows cols) d_model")
    plot_similarities(probe,
                      title=f"Similarity between {name} vectors",
                      x=full_board_labels,
                      y=full_board_labels)


# %% Similarity between mine and theirs (for each square)
def plot_similarities_2(v1: Float[Tensor, '*n_vectors rows cols'],
                        v2: Float[Tensor, '*n_vectors rows cols'],
                        name: str = ""):
    v1 = v1.flatten(end_dim=-3)
    v2 = v2.flatten(end_dim=-3)
    sim = einops.einsum(
        v1 / t.norm(v1, dim=0),
        v2 / t.norm(v2, dim=0),
        "d_model rows cols, d_model rows cols -> rows cols",
    )
    plot_square_as_board(sim, title=f"Cosine similarity between {name}")


plot_similarities_2(my_probe, their_probe)

# %% UMAP on the vectors of the probe
import umap
import umap.plot
import pandas as pd
#%%
vectors = einops.rearrange(linear_probe, "d_model rows cols options -> (options rows cols) d_model")

mapper = umap.UMAP(metric='cosine').fit(vectors.cpu().numpy())
#%%
labels = [probe_name for probe_name in ["blank", "their", "my"] for _ in full_board_labels]
hover_data = pd.DataFrame({
    "square": full_board_labels * 3,
    "probe": labels,
})

umap.plot.show_interactive(
    umap.plot.interactive(mapper, labels=labels, hover_data=hover_data, theme='inferno'))


# %%
def plot_PCA(vectors: Float[Tensor, '*n_vectors dim'], name: str = ""):
    vectors = vectors.flatten(end_dim=-2)
    vectors = StandardScaler().fit_transform(vectors.cpu().numpy())
    pca = PCA()
    pca.fit(vectors)
    # return px.bar(
    #     x=range(1, len(pca.explained_variance_) + 1),
    #     y=pca.explained_variance_,
    #     title=f"PCA explained variance for {name}",
    #     labels={"x": "Component", "y": "Log explained variance"},
    # )
    display(
        px.bar(x=range(len(pca.explained_variance_ratio_)),
               y=pca.explained_variance_ratio_,
               title=f"Explained variance ratio of the PCA on {name}"))

    return pca


# %% Run PCA on the vectors of the probe
vectors = einops.rearrange(linear_probe, "d_model rows cols options -> (options rows cols) d_model")
plot_PCA(vectors, "the probe vectors")

# %% Same PCA but with the unembeddings
plot_PCA(model.W_U, "the unembeddings")

# %%
plot_PCA(model.W_pos, "the embeddings")
# %%
all_vectors = [
    model.W_U.T,
    model.W_E,
    model.W_pos,
    vectors,
]
all_vectors = [(v - v.mean(dim=0)) / v.std(dim=0) for v in all_vectors]

plot_PCA(t.cat(all_vectors, dim=0), "the embeddings and unembeddings")
# %%
plot_PCA(t.cat([my_direction, blank_direction], dim=1).flatten(1).T, "the direction vectors")
# %%
plot_PCA(my_direction.flatten(1).T, "the direction vectors")
# %%
plot_PCA(blank_direction.flatten(1).T, "the direction vectors")
# %%

dim = 512
n_vectors = 1000
vectors = t.randn(n_vectors, dim)
vectors = vectors / t.norm(vectors, dim=1, keepdim=True)
cosine_sim = einops.einsum(vectors, vectors, "vec_1 dim, vec_2 dim -> vec_1 vec_2")

print(f"Mean cosine similarity: {cosine_sim.mean():.3f}")
print(f"Std cosine similarity: {cosine_sim.std():.3f}")
print(f"Mean abs cosine similarity: {cosine_sim.abs().mean():.3f}")

# %% plot histogram of cosine similarities
px.histogram(cosine_sim.flatten(), title="Cosine similarity between random vectors")


# %% Training the probe!
@dataclass
class ProbeTrainingArgs:
    # Which layer, and which positions in a game sequence to probe
    layer: int = 6
    pos_start: int = 5
    pos_end: int = model.cfg.n_ctx - 5
    length: int = pos_end - pos_start
    alternating: Tensor = t.tensor([1 if i % 2 == 0 else -1 for i in range(length)])

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

    # Othognal regularization (to an other probe)
    penalty_weight: float = 1000.0

    # Misc.
    probe_name: str = "main_linear_probe"

    # Code to get randomly initialized probe
    def setup_linear_probe(self, model: HookedTransformer):
        linear_probe = t.randn(
            model.cfg.d_model,
            self.rows,
            self.cols,
            self.options,
            requires_grad=False,
            device=device,
        ) / np.sqrt(model.cfg.d_model)
        # We want to pass this tensor to the optimizer, so it needs to be a leaf,
        # and not be a computation of other tensors (here divison by sqrt(d_model))
        # Thus, we can't use the `requires_grad` argument of `t.randn`.
        linear_probe.requires_grad = True
        return linear_probe


# def seq_to_state_stack(str_moves):
#     board = OthelloBoardState()
#     states = []
#     for move in str_moves:
#         board.umpire(move)
#         states.append(np.copy(board.state))
#     states = np.stack(states, axis=0)
#     return states


class LitLinearProbe(pl.LightningModule):

    def __init__(self, model: HookedTransformer, args: ProbeTrainingArgs, old_probe: Optional[Float[Tensor, "d_model rows cols options"]] = None):
        super().__init__()
        self.model = model
        self.args = args
        self.linear_probe: Float[Tensor,
                                 "d_model rows cols options"] = args.setup_linear_probe(model)
        """shape: (d_model, rows, cols, options)"""
        self.old_probe = old_probe

        pl.seed_everything(42, workers=True)

    def training_step(self, batch: Int[Tensor, "game_idx"], batch_idx: int) -> t.Tensor:
        focus_moves = slice(self.args.pos_start, self.args.pos_end)

        games_token = full_games_tokens[batch.cpu()]
        games_board_index = full_games_board_index[batch.cpu()]
        state_stack = move_sequence_to_state(games_board_index)
        state_stack = state_stack[:, focus_moves]
        state_stack = einops.einsum(state_stack, self.args.alternating,
                                    "batch move row col, move -> batch move row col")
        state_stack_one_hot = state_stack_to_one_hot(state_stack).to(device)
        batch_size = self.args.batch_size
        game_len = self.args.length

        # games_int = tensor of game sequences, each of length 60
        # This is the input to our model
        assert isinstance(games_token, Int[Tensor, f"batch={batch_size} full_game_len=60"])

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

        with t.inference_mode():
            _, cache = self.model.run_with_cache(
                games_token[:, :-1],  # Not the last move
                names_filter=lambda name: name == act_name,
            )

        resid_post: Float[Tensor, "batch moves d_model"]
        resid_post = cache[act_name][:, focus_moves]

        probe_logits = einops.einsum(
            self.linear_probe,
            resid_post,
            "d_model row col option, batch move d_model -> batch move row col option",
        )
        probe_logprobs = probe_logits.log_softmax(dim=-1)
        probe_loss = einops.reduce(
            probe_logprobs * state_stack_one_hot, "batch move row col option -> move row col",
            "mean") * self.args.options  # Multiply to correct for the mean over options

        loss = -probe_loss.mean(0).sum()  # avg over moves, sum over the board

        if self.old_probe is not None:
            penalisation = einops.einsum(
                self.old_probe / t.norm(self.old_probe, dim=0),
                self.linear_probe / t.norm(self.linear_probe, dim=0),
                "d_model row col option, d_model row col option ->",
            ) ** 2 * self.args.penalty_weight
            self.log("penalisation", penalisation)
            return loss + penalisation

        self.log("train_loss", loss)
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

# %% Compute the accuracy of the probe


@t.inference_mode()
def probe_accuracy(
    model: HookedTransformer,
    probe: Float[Tensor, "d_model rows cols options"],
    game_tokens: Int[Tensor, "batch game_len"],
    game_board_index: Int[Tensor, "batch game_len"],
    pos_start: int = 5,
    pos_end: int = -5,
    layer: int = 6,
) -> Float[Tensor, "batch"]:
    # We first convert the board states to be in terms of my (+1) and their (-1), rather than black and white

    alternating = t.tensor([-1 if i % 2 == 0 else 1 for i in range(60)])

    states = move_sequence_to_state(game_board_index)
    flipped_states = states * alternating[:, None, None]

    # We now convert to one-hot encoded vectors
    states_flipped_one_hot = state_stack_to_one_hot(t.tensor(flipped_states))

    # Take the argmax (i.e. the index of option empty/their/mine)
    states_flipped_value = states_flipped_one_hot.argmax(dim=-1)

    moves = slice(pos_start, pos_end)

    state_stack = move_sequence_to_state(game_board_index)
    state_stack = state_stack[:, moves]
    state_stack = einops.einsum(state_stack, alternating[moves],
                                "batch move row col, move -> batch move row col")

    act_name = utils.get_act_name("resid_post", layer)
    with t.inference_mode():
        _, cache = model.run_with_cache(
            game_tokens[:, :-1].to(model.cfg.device),  # Not the last move
            names_filter=lambda name: name == act_name,
        )

    probe_out = einops.einsum(
        cache["resid_post", layer], probe,
        "game move d_model, d_model row col options -> game move row col options")

    probe_out_value = probe_out.argmax(dim=-1)
    print(probe_out_value.shape)
    print(states_flipped_value.shape)

    correct_answers = (probe_out_value.cpu()  == states_flipped_value[:, :-1])[:, moves]
    accuracy = einops.reduce(correct_answers.float(), "game move row col -> row col", "mean")

    print(accuracy.shape)
    plot_square_as_board(
        1 - accuracy,
        title="Accuracy Rate of Linear Probe",
    )

# %%
probe_accuracy(
    model,
    litmodel.linear_probe,
    full_games_tokens[-100:],
    full_games_board_index[-100:],
)
# %%
plot_similarities_2(litmodel.linear_probe[..., 0], blank_probe,
                    "New and old blank probe")
# %%
plot_similarities_2(litmodel.linear_probe[..., 1], my_probe,
                    "New and old mine probe")

# %%
plot_similarities_2(litmodel.linear_probe[..., 2], their_probe,
                    "New and old their probe")

# %%
