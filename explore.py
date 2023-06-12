# %%
import os

os.environ["ACCELERATE_DISABLE_RICH"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from functools import partial
from typing import Tuple, Union

from circuitsvis.attention import attention_patterns
import einops
import torch as t
import transformer_lens.utils as utils
import wandb
from jaxtyping import Float, Int
from neel_plotly import line
from torch import Tensor
from transformer_lens import (
    HookedTransformer,
)
from transformer_lens.hook_points import HookPoint

from plotly_utils import imshow

from utils import *
from probes import get_probe
from othello_world.mechanistic_interpretability.mech_interp_othello_utils import (
    plot_single_board, )

try:
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger, WandbLogger
except ValueError:
    print("pytorch_lightning working")

#  %%
# %%
# Zero ablation of every head and MLP
# We try to see how much the overall loss of the model increases when ablating each component
# First, we evalute the loss only on the focus games

# print(get_loss(model, focus_games_tokens, focus_games_board_index))
# %%
# Now, we find the loss but after ablating each head

SHOW_ATTENTION = False
if SHOW_ATTENTION:
    _, focus_cache = model.run_with_cache(focus_games_tokens[:, :-1])

    game_idx = 0
    layer = 0
    labels = [to_board_label(focus_games_board_index[game_idx, i]) for i in range(59)]
    attention_patterns(labels, focus_cache["pattern", layer][game_idx])

# %%
# %%
# %% Probe exploration
probes = [get_probe(i, device=device) for i in range(3)]

# %%
plot_probe_accuracy(
    model,
    linear_probe,
    full_games_tokens[-100:],
    full_games_board_index[-100:],
    per_option=True,
    per_move="board_accuracy",
    # per_move='cell_accuracy',
    name="neel's probe",
    # mode='softmax',
)

# %%
EXPLORE_PROBE = True

if EXPLORE_PROBE:
    for probe, name in zip(probes, ["new probe", "orthogonal probe", "orthogonal probe 2"]):
        plot_probe_accuracy(
            model,
            probe.to(device),
            full_games_tokens[-100:],
            full_games_board_index[-100:],
            per_option=True,
            name=name,
        )

# %%

if False:
    plot_similarities_2(new_probe[..., 0], blank_probe, "New and old blank probe")
    plot_similarities_2(new_probe[..., 1], their_probe, "New and old mine probe")
    plot_similarities_2(new_probe[..., 2], my_probe, "New and old their probe")

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
# plot_square_as_board(board_tensor)

# %% Try to run the model on a virtual residual stream


def hook(activation: Float[Tensor, "game move d_model"], hook: HookPoint):
    activation[:, -1] = resid


layer = 4
act_name = utils.get_act_name("resid_pre", layer)
osef_input = focus_games_tokens[:1, :20]  # 1 game, 20 moves
logits = model.run_with_hooks(osef_input, fwd_hooks=[(act_name, hook)])

# Plot what the model predicts
logits = logits_to_board(logits[0, -1], "log_prob")
plot_square_as_board(logits, title="Model predictions")

# %%
# Compute and show probe vector norms
# probe_norm = new_probe.norm(dim=0)
# histogram
# px.histogram(probe_norm.cpu().flatten(), title="Probe vector norms", labels={"value": "norm"})

# %%


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

    orig_logits = logits_to_board(model(moves_orig)[0, -1], "log_prob")
    patched_logits = logits_to_board(patched_logits[0, -1], "log_prob")
    new_logits = logits_to_board(new_logits[0, -1], "log_prob")

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
    plot_square_as_board(
        all_logits,
        title="Model predictions",
        facet_col=-1,
        facet_col_wrap=3,
        facet_labels=[
            "New expected",
            "new logits",
            "logit diff (patch - orig)",
            "original expected",
            "original logits",
            "patched logits",
        ],
    )

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

modify_resid_given_probe(model, orig_games, new_games, *probes, layer=layer, cells=["D2"])

# %%
plot_single_board(focus_games_board_index[orig_index, :move_index], title="Original game")
plot_single_board(focus_games_board_index[new_index, :move_index], title="New game")

# %%

# %%
MAKE_NEW_TRAINING_DATA = False
if MAKE_NEW_TRAINING_DATA:
    games_board_index, games_valid_moves = make_training_data()
else:
    games_board_index, games_valid_moves = get_training_data()

games_tokens = BOARD_TO_TOKENS[games_board_index]
games_states = move_sequence_to_state(games_board_index, mode="alternate")

# %%
valid_board_index, _ = generate_training_data(1_000, seed=69)
valid_states = move_sequence_to_state(valid_board_index, mode="alternate")
valid_tokens = BOARD_TO_TOKENS[valid_board_index]

# %% Compute the game states

COMPUTE_STATS = False
if COMPUTE_STATS:
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
if 0:
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
from probes import ProbeTrainingArgs, LitLinearProbe, PROBE_DIR

# %%
wandb.finish()

# %%
probes = []

for num_probe in range(3):
    args = ProbeTrainingArgs(
        lr=1e-3,
        max_epochs=4,
        wd=0.05,
        train_tokens=games_tokens,
        train_states=games_states,
        valid_tokens=valid_tokens,
        valid_states=valid_states,
        correct_for_dataset_bias=False,
        probe_name=f"orthogonal_probe_{num_probe}",
    )
    lit_ortho_probe = LitLinearProbe(model, args, *probes)

    logger = WandbLogger(save_dir=os.getcwd() + "/logs", project=args.probe_name)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        log_every_n_steps=1,
        val_check_interval=100,
        check_val_every_n_epoch=None,
    )

    trainer.fit(model=lit_ortho_probe)
    probes.append(lit_ortho_probe.linear_probe)
    model = model.to(device)  # pl trainer moves model to cpu at the end :shrug:
    plot_probe_accuracy(
        model,
        lit_ortho_probe.linear_probe,
        valid_tokens,
        valid_board_index,
        per_option=True,
        per_move="board_accuracy",
    )

    wandb.finish()

    path = PROBE_DIR / f"orthogonal_probe_{num_probe}.pt"
    if not path.exists():
        t.save(lit_ortho_probe.linear_probe, path)
        print(f"Saved probe to {path.resolve()}")
    else:
        print(f"Warning: {path.resolve()} already exists. Not saving the probe.")

# %%
