# %%
import os

os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import itertools
import random
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import joblib
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
import transformer_lens.utils as utils
import wandb
from IPython.display import HTML, display
from ipywidgets import interact
from jaxtyping import Bool, Float, Int, jaxtyped
from neel_plotly import line, scatter
from rich import print as rprint
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm, trange
from transformer_lens import (ActivationCache, FactoredMatrix, HookedTransformer,
                              HookedTransformerConfig)
from transformer_lens.hook_points import HookedRootModule, HookPoint

from plotly_utils import imshow

from utils import *
from probe_training import get_probe

try:
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger, WandbLogger
except ValueError:
    print("pytorch_lightning working")

# %%
t.set_grad_enabled(False)
device = "cuda" if t.cuda.is_available() else "cpu"

# %%
cfg, model = get_othello_gpt(device)

# %% Loading sample data
full_games_tokens, full_games_board_index = load_sample_games()
full_games_tokens, full_games_board_index = load_sample_games()
num_games = 50
focus_games_tokens = full_games_tokens[:num_games]
focus_games_board_index = full_games_board_index[:num_games]

focus_states = move_sequence_to_state(focus_games_board_index)
focus_valid_moves = move_sequence_to_state(focus_games_board_index, mode="valid")

print("focus states:", focus_states.shape)
print("focus_valid_moves", focus_valid_moves.shape)

# %%
# Zero ablation of every head and MLP
# We try to see how much the overall loss of the model increases when ablating each component
# First, we evalute the loss only on the focus games

print(get_loss(model, focus_games_tokens, focus_games_board_index))
# %%
# Now, we find the loss but after ablating each head

SHOW_ATTENTION = False
if SHOW_ATTENTION:
    _, focus_cache = model.run_with_cache(focus_games_tokens[:, :-1])

    game_idx = 0
    layer = 0
    labels = [to_board_label(focus_games_board_index[game_idx, i]) for i in range(59)]
    attention_patterns(labels, focus_cache['pattern', layer][game_idx])

# %%
RUN_ABLATIONS = True

if RUN_ABLATIONS:
    individual_heads = True
    n_games = 50
    tokens = full_games_tokens[-n_games:].to(device)
    board_index = full_games_board_index[-n_games:].to(device)
    get_metrics = lambda model: get_loss(model, tokens, board_index, 5, -5).to_tensor()
    zero_ablation_metrics = zero_ablation(model, get_metrics, individual_heads).cpu()
    base_metrics = get_metrics(model)
# %%
# Plotting the results
if RUN_ABLATIONS:
    x = [f"Head {i}" for i in range(model.cfg.n_heads)] + ["All Heads", "MLP"]
    y = [f"Layer {i}" for i in range(model.cfg.n_layers)]
    if not individual_heads:
        x = x[-2:]

    imshow(
        zero_ablation_metrics[:3] - base_metrics[:3, None, None],
        title="Metric increase after zeroing each component",
        x=x,
        y=y,
        facet_col=0,
        facet_labels=["Loss", "Cell accuracy", "Board accuracy"],
    )

# %% Are Attention heads even useful?

# Abblate all attention after the layer `n`
if RUN_ABLATIONS:

    def filter(name: str, start_layer: int = 0):
        if not name.startswith("blocks."):
            # 'hook_embed' or 'hook_pos_embed' or 'ln_final.hook_scale' or 'ln_final.hook_normalized'
            return False

        layer = int(name.split(".")[1])

        return layer >= start_layer and "attn_out" in name

    metrics_per_layer = []
    for start_layer in range(model.cfg.n_layers):
        with model.hooks(fwd_hooks=[(partial(filter, start_layer=start_layer),
                                     zero_ablation_hook)]):
            metrics_per_layer.append(get_metrics(model))

# %% Plot
if RUN_ABLATIONS:
    lines = t.stack(metrics_per_layer, dim=1).cpu()
    line(
        lines[:3],
        x=[f"≥ {i}" for i in range(model.cfg.n_layers)],
        facet_col=0,
        #  facet_col_wrap=3,
        facet_labels=['Loss', 'Cell accuracy', 'Board accuracy'
                      ],  # 'False Positive', 'False Negative', 'True Positive', 'True Negative'],
        title="Metrics after zeroing all attention heads above a layer",
    )
# %% Verify what happens when ablating every head
with model.hooks(fwd_hooks=[(lambda name: "attn_out" in name, zero_ablation_hook)]):
    logits = model(focus_games_tokens[:, :20])

game = 1
for game in (game, ):
    # 2 and 15 because the same move was played in the 20th move
    # We check here that the model does the same on both (i.e. attention is not used)
    print(focus_games_tokens[game, :20])
    plot_square_as_board(logits_to_board(logits[game, -1], 'log_prob'))
    plot_single_board(tokens_to_board(focus_games_tokens[game, :20]))

assert t.allclose(logits[2, -1], logits[15, -1])

# %%
print(focus_games_tokens[:, 19])

# %%
# Exploration of the probe

linear_probe = get_neels_probe(device)
"""Shape (d_model, rows, cols, options)
options: 0: blank, 1: my piece, 2: their piece"""

blank_probe, my_probe, their_probe = linear_probe.unbind(dim=-1)
blank_direction = blank_probe - (their_probe + my_probe) / 2
my_direction = my_probe - their_probe

# %%

for probe, name in zip([blank_probe, their_probe, my_probe], ["blank", "their", "my"]):
    probe = einops.rearrange(probe, "d_model rows cols -> (rows cols) d_model")
    plot_similarities(probe,
                      title=f"Similarity between {name} vectors",
                      x=full_board_labels,
                      y=full_board_labels)

# %% Similarity between mine and theirs (for each square)
plot_similarities_2(my_probe, their_probe)

# %% UMAP on the vectors of the probe
UMAP = False
if UMAP:
    import umap
    import umap.plot
    import pandas as pd

    vectors = einops.rearrange(linear_probe,
                               "d_model rows cols options -> (options rows cols) d_model")

    mapper = umap.UMAP(metric='cosine').fit(vectors.cpu().numpy())

    labels = [probe_name for probe_name in ["blank", "their", "my"] for _ in full_board_labels]
    hover_data = pd.DataFrame({
        "square": full_board_labels * 3,
        "probe": labels,
    })
    mapper = umap.UMAP(metric='cosine').fit(vectors.cpu().numpy())

    labels = [probe_name for probe_name in ["blank", "their", "my"] for _ in full_board_labels]
    hover_data = pd.DataFrame({
        "square": full_board_labels * 3,
        "probe": labels,
    })

    umap.plot.show_interactive(
        umap.plot.interactive(mapper, labels=labels, hover_data=hover_data, theme='inferno'))
    umap.plot.show_interactive(
        umap.plot.interactive(mapper, labels=labels, hover_data=hover_data, theme='inferno'))

# %%
PLOT_PCAS = True
# %%
PLOT_PCAS = True  # if false, disable all PCAs


def plot_PCA(vectors: Float[Tensor, '*n_vectors dim'],
             name: str = "",
             absolute: bool = False,
             flip_dim_order: bool = False):
    """Plot the PCA of the vectors

    Args:
        vectors (Float[Tensor, "*n_vectors dim"): The vectors to do the PCA on
        name (str, optional): The name for the plot.
        absolute (bool, optional): If true, plots the explained variance instead of the ratio.
        flip_dim_order (bool, optional): If true, the first dimension of the input is
            the dimension of the vectors. Otherwise it is the last.
    """
    if not PLOT_PCAS:
        return

    if flip_dim_order:
        vectors = einops.rearrange(vectors, "dim ... -> ... dim")

    vectors = vectors.flatten(end_dim=-2)
    vectors = StandardScaler(with_mean=False).fit_transform(vectors.cpu().numpy())
    pca = PCA()
    pca.fit(vectors)
    # return px.bar(
    #     x=range(1, len(pca.explained_variance_) + 1),
    #     y=pca.explained_variance_,
    #     title=f"PCA explained variance for {name}",
    #     labels={"x": "Component", "y": "Log explained variance"},
    # )
    if absolute:
        y = pca.explained_variance_
    else:
        y = pca.explained_variance_ratio_

    display(
        px.bar(x=range(len(pca.explained_variance_ratio_)),
               y=y,
               title=f"Explained variance ratio of the PCA on {name}"))

    return pca


# %% Run PCA on the vectors of the probe
vectors = einops.rearrange(linear_probe, "d_model rows cols options -> (options rows cols) d_model")
plot_PCA(vectors, "the probe vectors")
# %% The same be per option
for i in range(3):
    plot_PCA(linear_probe[..., i],
             f"the probe vectors for option {i}",
             flip_dim_order=True,
             absolute=True)

# %% Normalise the probe then run PCA
normalised_probe = linear_probe / linear_probe.norm(dim=-1, keepdim=True)
plot_PCA(normalised_probe, "the normalised probe vectors", flip_dim_order=True)

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

# %% Probe exploration
new_probe = get_probe(0, device=device)
ortho_probe = get_probe(1, device=device)

# %%
for probe, name in zip([new_probe, ortho_probe, ortho_probe_2],
                       ["new probe", "orthogonal probe", "orthogonal probe 2"]):
    plot_probe_accuracy(
        model,
        probe.to(device),
        full_games_tokens[-100:],
        full_games_board_index[-100:],
        per_option=True,
        name=name,
    )

# %%
# the_probe = new_probe
# the_probe = ortho_probe
the_probe = ortho_probe_2
# the_probe = linear_probe
plot_probe_accuracy(
    model,
    the_probe.to(device),
    full_games_tokens[-100:],
    full_games_board_index[-100:],
    per_option=True,
    name="orthogonal probe 2",
)

# %%

plot_similarities_2(new_probe[..., 0], blank_probe, "New and old blank probe")
plot_similarities_2(new_probe[..., 1], their_probe, "New and old mine probe")
plot_similarities_2(new_probe[..., 2], my_probe, "New and old their probe")

# %%
for i in range(3):
    plot_similarities_2(ortho_probe[..., i], linear_probe[..., i], f"Orthogonal and new probe {i}")

# %% Similarities between the blank probe and the token embeddings

token_embs = model.W_E[1:]
token_embs_64 = t.zeros((64, token_embs.shape[1]), device=token_embs.device)
token_embs_64[TOKENS_TO_BOARD] = token_embs
print(token_embs.shape)
token_embs_64 = einops.rearrange(token_embs_64, "(rows cols) d_model -> d_model rows cols", rows=8)

plot_similarities_2(
    linear_probe[..., 0],
    token_embs_64,
    name="Blank probe and token embeddings",
)

# %%

# ---------------------- #
# --- World building --- #
# ---------------------- #


def make_residual_stream(
    world: Union[Int[Tensor, "row=8 cols=8"], str],
    probe: Float[Tensor, "d_model rows cols options=3"],
) -> Float[Tensor, "d_model"]:
    """
    Create the embedding of a board state according to the probe

    Args:
        world: the board state, with 0 for blank, +1 for mine, -1 for theirs
        probe: directions in the residual stream that correspond to each square.
            The last dimension is the options, with 0 for blank, 1 for mine, 2 for theirs
    """

    if isinstance(world, str):
        world = board_to_tensor(world)

    d_model = probe.shape[0]
    blank_direction = probe[..., 0] - (probe[..., 1] + probe[..., 2]) / 2
    my_direction = probe[..., 1] - probe[..., 2]

    world = world.to(probe.device)
    embedding = t.zeros(d_model, device=probe.device)
    for row in range(world.shape[0]):
        for col in range(world.shape[1]):
            if world[row, col] == 0:
                embedding += blank_direction[:, row, col]
            else:
                embedding += my_direction[:, row, col] * world[row, col]

    return embedding


# %%

board = """
........
........
........
...xo...
...ox...
........
........
........
"""

board_2 = """
........
........
........
..xxxx..
.xooooo.
.o..ox..
.x.oxo..
........
"""
board_3 = """
........
........
........
...xo...
...oo...
....o...
........
........
"""

board_tensor = board_to_tensor(board_3)
resid = make_residual_stream(board_tensor, linear_probe)
line(resid)
plot_square_as_board(board_tensor)

# %% Try to run the model on a virtual residual stream


def hook(activation: Float[Tensor, "game move d_model"], hook: HookPoint):
    activation[:, -1] = resid


layer = 4
act_name = utils.get_act_name("resid_pre", layer)
osef_input = focus_games_tokens[:1, :20]  # 1 game, 20 moves
logits = model.run_with_hooks(osef_input, fwd_hooks=[(act_name, hook)])

# Plot what the model predicts
logits = logits_to_board(logits[0, -1], 'log_prob')
plot_square_as_board(logits, title="Model predictions")

# %%
# Compute and show probe vector norms
probe_norm = new_probe.norm(dim=0)
# histogram
px.histogram(probe_norm.cpu().flatten(), title="Probe vector norms", labels={"value": "norm"})

# %%


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


@t.inference_mode()
def modify_resid_given_probe(
        model: HookedTransformer,
        moves_orig: Int[Tensor, "move"],
        moves_new: Int[Tensor, "move"],
        *probes: Float[Tensor, "d_model rows cols options=3"],
        layer: int = 6,
        cells: Tuple[str, ...] = (),
):
    act_name = utils.get_act_name("resid_pre", layer)
    new_logits, new_cache = model.run_with_cache(
        moves_new,
        names_filter=lambda name: name == act_name,
    )

    def hook(orig_activation: Float[Tensor, "game move d_model"], hook: HookPoint):
        # Step 0. Find a basis of the subspace of the probe
        # collect the probe vectors
        all_probes = t.stack(probes, dim=-1).to(orig_activation.device)
        if cells:
            rows_cols = t.tensor([board_label_to_row_col(cell) for cell in cells])
            all_probes = all_probes[:, rows_cols[:, 0], rows_cols[:, 1]]

        probe_vectors = einops.rearrange(all_probes, "d_model ... -> (...) d_model")

        orig_activation[:, -1] = swap_subspace(
            orig_activation[:, -1],
            new_cache[act_name][:, -1],
            probe_vectors,
        )

    patched_logits = model.run_with_hooks(
        moves_orig,
        fwd_hooks=[(act_name, hook)],
    )

    # display the logits
    orig_valid_moves = move_sequence_to_state(tokens_to_board(moves_orig), mode="valid")
    if cells:
        rows_cols = [board_label_to_row_col(cell) for cell in cells]
        index = tuple(zip(*rows_cols))

        # Compute the state of orig and new board
        new_board_state = move_sequence_to_state(tokens_to_board(moves_new), mode="normal")[0, -1]
        orig_board_state = move_sequence_to_state(tokens_to_board(moves_orig), mode="normal")[0, -1]
        # Put the cells of the new board in the orig board
        orig_board_state[index] = new_board_state[index]

        valid_cells = valid_moves_from_board(orig_board_state, moves_orig.shape[1])
        new_valid_moves = one_hot(valid_cells).reshape(1, 1, 8, 8)

    else:
        new_valid_moves = move_sequence_to_state(tokens_to_board(moves_new), mode="valid")

    orig_logits = logits_to_board(model(moves_orig)[0, -1], 'log_prob')
    patched_logits = logits_to_board(patched_logits[0, -1], 'log_prob')
    new_logits = logits_to_board(new_logits[0, -1], 'log_prob')

    scale = new_logits.abs().max().cpu()

    to_stack = [
        orig_valid_moves[0, -1] * scale,
        orig_logits,
        patched_logits,
        new_logits,
        new_valid_moves[0, -1] * scale,
        patched_logits - orig_logits,
    ]

    all_logits = t.stack([t.cpu() for t in to_stack], dim=-1)
    plot_square_as_board(all_logits,
                         title="Model predictions",
                         facet_col=-1,
                         facet_col_wrap=3,
                         facet_labels=[
                             'New expected',
                             "new logits",
                             'logit diff (patch - orig)',
                             "original expected",
                             'original logits',
                             "patched logits",
                         ])

    # plot_square_as_board(logits_to_board(new_logits[0, -1], 'log_prob'),
    #                      title="Model predictions (new)")
    # plot_square_as_board(logits_to_board(patched_logits[0, -1], 'log_prob'),
    #                      title="Model predictions (patched)")

    # with model.hooks(fwd_hooks=[(act_name, hook)]):
    #     plot_board_log_probs(
    #         tokens_to_board(moves_new[0]),
    #         patched_logits[0],
    #     )


orig_index = 2
new_index = 3
move_index = 20
layer = 4
orig_games = focus_games_tokens[orig_index:orig_index + 1, :move_index]
new_games = focus_games_tokens[new_index:new_index + 1, :move_index]

modify_resid_given_probe(model,
                         orig_games,
                         new_games,
                         new_probe,
                         ortho_probe,
                         layer=layer,
                         cells=['D2'])

# %%
plot_single_board(focus_games_board_index[orig_index, :move_index], title="Original game")

# plot_single_board(focus_games_board_index[new_index, :move_index], title="New game")


# %%
def valid_moves_from_board(board_state: Int[Tensor, 'row col'], move_index: int) -> List[int]:
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

np_board_state = move_sequence_to_state(tokens_to_board(orig_games), mode="valid")[0, -1].numpy()
np_board_state
# %%
from probe_training import ProbeTrainingArgs, LitLinearProbe, PROBE_DIR

# %%
wandb.finish()

args = ProbeTrainingArgs(train_tokens=full_games_tokens,
                         train_board_indices=full_games_board_index,
                         probe_name='orthogonal_probe')
lit_ortho_probe = LitLinearProbe(model, args, new_probe, ortho_probe)

logger = WandbLogger(save_dir=os.getcwd() + "/logs", project=args.probe_name)
trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    logger=logger,
    log_every_n_steps=1,
)
trainer.fit(model=lit_ortho_probe)
wandb.finish()

ortho_probe_2 = lit_ortho_probe.linear_probe
path = PROBE_DIR / "orthogonal_probe_2.pt"
if not path.exists():
    t.save(ortho_probe, path)
    print(f"Saved probe to {path.resolve()}")
else:
    print(f"Warning: {path.resolve()} already exists. Not saving the probe.")

# %%
MAKE_NEW_TRAINING_DATA = False
if MAKE_NEW_TRAINING_DATA:
    games_tokens, games_valid_moves = make_training_data()
else:
    games_tokens, games_valid_moves = get_training_data()

# %% Compute the game states
COMPUTE_STATS = False
if COMPUTE_STATS:
    games_states = move_sequence_to_state(games_tokens, mode="alternate")
    compute_stats(games_states, games_valid_moves)
else:
    stats = t.load(STATS_PATH)
    print(stats.shape)

stat_names = ["Empty", "My piece", "Their piece", "Valid move"]

# %% Plot stats per cell
plot_square_as_board(
    stats.mean(1),
    title="Average frequency of each cell being ...",
    facet_col=0,
    facet_labels=stat_names,
)

# %% Plot stats per cell and move
moves_to_show = [0, 5, 10, 20, 30, 40, 50, 55]
x = einops.rearrange(stats[:, moves_to_show], "option m r c -> (m option) r c")
labels = [f"{name} (move {move})" for move in moves_to_show for name in stat_names]

plot_square_as_board(
    x,
    facet_col=0,
    facet_col_wrap=4,
    facet_labels=labels,
    title="Average frequency of each cell being ...",
    height=3000,
)

# %%
