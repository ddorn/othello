# %%
import os

# os.environ["ACCELERATE_DISABLE_RICH"] = "1"
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

from utils import *

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
num_games = 50
focus_games_tokens = full_games_tokens[:num_games]
focus_games_board_index = full_games_board_index[:num_games]

focus_states = move_sequence_to_state(focus_games_board_index)
focus_valid_moves = move_sequence_to_state(focus_games_board_index, mode="valid")

print("focus states:", focus_states.shape)
print("focus_valid_moves", focus_valid_moves.shape)

# flipped_focus_states = alternate_states(focus_states)

# # We now convert to one-hot encoded vectors
# focus_states_flipped_one_hot = state_stack_to_one_hot(flipped_focus_states)

# # Take the argmax (i.e. the index of option empty/their/mine)
# focus_states_flipped_value = focus_states_flipped_one_hot.argmax(dim=-1).to(device)

# %%
# Zero ablation of every head and MLP
# We try to see how much the overall loss of the model increases when ablating each component
# First, we evalute the loss only on the focus games

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
#%%
if UMAP:
    vectors = einops.rearrange(linear_probe,
                               "d_model rows cols options -> (options rows cols) d_model")

    mapper = umap.UMAP(metric='cosine').fit(vectors.cpu().numpy())

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
# %%
neel_acc = probe_accuracy(
    model,
    linear_probe,
    full_games_tokens[-100:],
    full_games_board_index[-100:],
    per_option=True,
)

# %%

my_acc = probe_accuracy(
    model,
    new_probe,
    full_games_tokens[-100:],
    full_games_board_index[-100:],
    per_option=True,
)

# %%
plot_square_as_board(
    my_acc - neel_acc,
    title="Difference in Error Rate of Linear Probe",
    facet_col=-1,
    facet_labels=["Blank", "Mine", "Theirs"],
)

# %%
plot_similarities_2(new_probe[..., 0], blank_probe, "New and old blank probe")
# %%
plot_similarities_2(new_probe[..., 1], their_probe, "New and old mine probe")

# %%
plot_similarities_2(new_probe[..., 2], my_probe, "New and old their probe")

# %%
probe_accuracy(
    model,
    ortho_probe,
    full_games_tokens[-300:],
    full_games_board_index[-300:],
    per_option=True,
)
# %%
for i in range(3):
    plot_similarities_2(ortho_probe[..., i], linear_probe[..., i], f"Orthogonal and new probe {i}")

# %%
plot_similarities_2(linear_probe[..., 1], linear_probe[..., 2], "Title")

# %% Similarities between the blank probe and the token embeddings

token_embs = model.W_E[1:]
token_embs_64 = t.zeros((64, token_embs.shape[1]), device=token_embs.device)
token_embs_64[tokens_to_board] = token_embs
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
@t.inference_mode()
def modify_resid_given_probe(
    model: HookedTransformer,
    moves_orig: Int[Tensor, "move"],
    moves_new: Int[Tensor, "move"],
    probe: Float[Tensor, "d_model rows cols options=3"],
    layer: int = 6,
):
    act_name = utils.get_act_name("resid_pre", layer)
    new_logits, new_cache = model.run_with_cache(
        moves_new,
        names_filter=lambda name: name == act_name,
    )
    print(*new_cache)

    def hook(orig_activation: Float[Tensor, "game move d_model"], hook: HookPoint):
        # Step 1. Remove the components that are in the space of the probe
        # collect the probe vectors
        probe_vectors = einops.rearrange(
            probe,
            "d_model rows cols options -> (options rows cols) d_model").to(orig_activation.device)
        # normalize the probe vectors
        probe_vectors = probe_vectors / probe_vectors.norm(dim=1, keepdim=True)
        residual = orig_activation[:, -1]
        # compute the coefficients of the probe vectors in the residual
        coefficients = einops.einsum(residual, probe_vectors,
                                     "game d_model, dir d_model -> game dir")
        # remove the components of the probe vectors
        residual -= einops.einsum(probe_vectors, coefficients,
                                  "dir d_model, game dir -> game d_model")

        # 2. Add the component of the from the new_cache
        new_residual = new_cache[act_name][:, -1]
        # compute the coefficients of the probe vectors in the new residual
        coefficients = einops.einsum(new_residual, probe_vectors,
                                     "game d_model, dir d_model -> game dir")
        # and finally add the components in the direction of the probe vectors of the new residual
        # into the residual
        residual += einops.einsum(probe_vectors, coefficients,
                                  "dir d_model, game dir -> game d_model")

    patched_logits = model.run_with_hooks(
        moves_orig,
        fwd_hooks=[(act_name, hook)],
    )

    # display the logits
    plot_square_as_board(logits_to_board(new_logits[0, -1], 'log_prob'),
                         title="Model predictions (new)")
    plot_square_as_board(logits_to_board(patched_logits[0, -1], 'log_prob'),
                         title="Model predictions (patched)")


orig_index = 0
new_index = 1
move_index = 20
orig_games = focus_games_tokens[orig_index:orig_index+1, :move_index]
new_games = focus_games_tokens[new_index:new_index+1, :move_index]
modify_resid_given_probe(model, orig_games, new_games, linear_probe)

# %%
plot_single_board(focus_games_board_index[orig_index, :move_index], title="Original game")
plot_single_board(focus_games_board_index[new_index, :move_index], title="New game")
# %%
