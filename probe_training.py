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
import torch.utils.data as data
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
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
)
from transformer_lens.hook_points import HookedRootModule, HookPoint

from plotly_utils import imshow

from utils import *

try:
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger, WandbLogger
except ValueError:
    print("pytorch_lightning working")

PROBE_DIR = Path(__file__).parent / "probes"
PROBE_DIR.mkdir(exist_ok=True)

# %%


@dataclass
class ProbeTrainingArgs:
    # Data  # TODO: Use a proper data loader, this is a hack for now
    train_tokens: Int[Tensor, "num_games full_game_len=60"]
    """Tokens between 0 and 60"""
    train_states: Int[Tensor, "num_games full_game_len=60 rows cols"]

    valid_tokens: Int[Tensor, "num_games full_game_len=60"]
    valid_states: Int[Tensor, "num_games full_game_len=60 rows cols"]

    correct_for_dataset_bias: Bool = True
    """Wether to correct for options that are present more often, per move and per game"""

    # Which layer, and which positions in a game sequence to probe
    layer: int = 6
    pos_start: int = 5
    pos_end: int = 54  # 59 - 5. 59 is the second to last move in a game, last one which we make a prediction

    # Game state (options are blank/mine/theirs)
    options: int = 3
    rows: int = 8
    cols: int = 8

    # Standard training hyperparams
    max_epochs: int = 8

    # Hyperparams for optimizer
    batch_size: int = 250
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.99)
    wd: float = 0.01

    # Othognal regularization (to an other probe)
    penalty_weight: float = 1_000.0

    # Misc.
    probe_name: str = "main_linear_probe"

    def __post_init__(self):
        assert self.pos_start < self.pos_end, "pos_start should be smaller than pos_end"
        assert (self.train_tokens < 60).all(), "Train tokens should be between 0 and 60"
        assert (self.valid_tokens < 60).all(), "Valid tokens should be between 0 and 60"

        if self.correct_for_dataset_bias:
            stats = compute_stats(self.train_states)
            self.dataset_stats = einops.rearrange(stats,
                                                  "stat move rows cols -> move rows cols stat")

    def setup_linear_probe(self,
                           model: HookedTransformer) -> Float[Tensor, "d_model rows cols options"]:
        """Returns a randomly initialized linear probe.

        The probe is a tensor of shape (d_model, rows, cols, options)."""
        linear_probe = t.randn(
            model.cfg.d_model,
            self.rows,
            self.cols,
            self.options,
            requires_grad=False,
            device=model.cfg.device,
        ) / np.sqrt(model.cfg.d_model)
        # We want to pass this tensor to the optimizer, so it needs to be a leaf,
        # and not be a computation of other tensors (here divison by sqrt(d_model))
        # Thus, we can't use the `requires_grad` argument of `t.randn`.
        linear_probe.requires_grad = True
        return linear_probe

    @property
    def length(self) -> int:
        return self.pos_end - self.pos_start


class LitLinearProbe(pl.LightningModule):
    """A linear probe for a transformer with its training configured for pytorch-lightning."""

    def __init__(
        self,
        model: HookedTransformer,
        args: ProbeTrainingArgs,
        *old_probes: Float[Tensor, "d_model rows cols options"],
    ):
        super().__init__()
        self.model = model
        self.args = args
        self.linear_probe = t.nn.Parameter(args.setup_linear_probe(model))
        """shape: (d_model, rows, cols, options)"""
        self.old_probes = [
            (old_probe / t.norm(old_probe, dim=0)).to(self.model.cfg.device).detach()
            for old_probe in old_probes
        ]

        if args.correct_for_dataset_bias:
            args.dataset_stats = args.dataset_stats.to(self.model.cfg.device)

        pl.seed_everything(42, workers=True)

    # def training_step(self, batch: Int[Tensor, "game_idx"], batch_idx: int) -> t.Tensor:
    def training_step(
        self,
        batch: Tuple[Int[Tensor, "game move"], Int[Tensor, "game move row col"]],
        batch_idx: int,
    ) -> Float[Tensor, ""]:
        """Return the loss for a batch."""

        game_len = self.args.length
        focus_moves = slice(self.args.pos_start, self.args.pos_end)

        tokens, states = batch

        state_stack_one_hot = state_stack_to_one_hot(states)[:, focus_moves]

        # state_stack_one_hot = tensor of one-hot encoded states for each game
        # We'll multiply this by our probe's estimated log probs along the `options` dimension, to get probe's estimated log probs for the correct option
        assert isinstance(
            state_stack_one_hot,
            Bool[
                Tensor,
                f"batch game_len={game_len} rows=8 cols=8 options=3",
            ],
        ), state_stack_one_hot.shape

        act_name = utils.get_act_name("resid_post", self.args.layer)

        with t.inference_mode():
            _, cache = self.model.run_with_cache(
                tokens[:, :-1],  # Not the last move
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
        loss = probe_logprobs * state_stack_one_hot.float()
        if self.args.correct_for_dataset_bias:
            loss = loss / (self.args.dataset_stats[focus_moves] + 1e-5)
        # Multiply to correct for the mean over options
        loss = -loss.mean() * self.args.options

        penalisation = t.tensor(0.0, device=self.model.cfg.device)
        for old_probe in self.old_probes:
            cosine_sim_sq = (einops.einsum(
                old_probe,
                self.linear_probe / t.norm(self.linear_probe, dim=0),
                "d_model row col option, d_model row col option -> row col option",
            )**2)
            penalisation += cosine_sim_sq.mean()
        penalisation = penalisation * self.args.penalty_weight

        total_loss = loss + penalisation

        self.log("loss", loss)
        self.log("penalisation", penalisation)
        self.log("total_loss", total_loss)
        return total_loss

    def train_dataloader(self):
        """
        Returns `games_int` and `state_stack_one_hot` tensors.
        """
        device = self.model.cfg.device

        games_tokens = self.args.train_tokens.to(device)
        games_states = self.args.train_states.to(device)

        data_loader = data.DataLoader(
            data.TensorDataset(games_tokens, games_states),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0,
        )

        return data_loader

    def validation_step(
        self,
        batch: Tuple[Int[Tensor, "game move"], Int[Tensor, "game move row col"]],
        batch_idx: int,
    ) -> Float[Tensor, ""]:
        tokens, states = batch

        acc: Float[Tensor, "move option=3"]
        acc = plot_probe_accuracy(
            self.model,
            self.linear_probe,
            tokens,
            TOKENS_TO_BOARD.to(tokens.device)[tokens],
            per_option=True,
            per_move="board_accuracy",
            plot=False,
        )
        acc_per_option = acc.mean(dim=0)

        self.log("Validation board accuracy", acc.mean())
        self.log("Validation board accuracy - blank", acc_per_option[0])
        self.log("Validation board accuracy - mine", acc_per_option[1])
        self.log("Validation board accuracy - their", acc_per_option[2])
        self.log("Validation board accuracy - Move variance", acc.mean(1).var())

        return acc.mean()

    def val_dataloader(self):
        device = self.model.cfg.device

        games_tokens = self.args.valid_tokens.to(device)
        games_states = self.args.valid_states.to(device)

        data_loader = data.DataLoader(
            data.TensorDataset(games_tokens, games_states),
            batch_size=len(games_tokens),
            num_workers=0,
        )

        return data_loader

    def configure_optimizers(self):
        return t.optim.AdamW(
            [self.linear_probe],
            lr=self.args.lr,
            betas=self.args.betas,
            weight_decay=self.args.wd,
        )


# %%
# Create the model & training system
FIRST_PROBE_PATH = PROBE_DIR / f"main_linear_probe.pt"
SECOND_PROBE_PATH = PROBE_DIR / f"orthogonal_probe.pt"
THIRD_PROBE_PATH = PROBE_DIR / f"orthogonal_probe_2.pt"
PROBE_PATHS = [FIRST_PROBE_PATH, SECOND_PROBE_PATH, THIRD_PROBE_PATH]


def get_probe(n: int = 0,
              flip_mine: bool = False,
              device="cpu") -> Float[Tensor, "d_model rows cols options"]:
    """
    Load the probes that I trained.

    Args:
        n: The index of the probe to load. The n'th probe is orthogonal to all the previous ones.
        flip_mine: Whether to flip the mine and opponent channels.
        device: The device to load the probe onto.

    Returns:
        A probe with shape (d_model, rows, cols, options)
        Options are:
            0: Blank
            1: Mine (or opponent if `flip_mine` is True)
            2: Opponent (or mine if `flip_mine` is True)
    """

    path = PROBE_DIR / f"orthogonal_probe_{n}.pt"
    if not path.exists():
        valid_probes = list(PROBE_DIR.glob("orthogonal_probe_*.pt"))
        if len(valid_probes) == 0:
            raise FileNotFoundError(f"Could not find any probes in {PROBE_DIR}.")
        raise FileNotFoundError(f"Could not find probe {n} in {PROBE_DIR}. "
                                f"Valid probes are {valid_probes}.")

    print(f"Loading probe from {path.resolve()}")
    probe = t.load(path, map_location=device)

    if flip_mine:
        probe = t.stack(
            [
                probe[..., 0],
                probe[..., 2],
                probe[..., 1],
            ],
            dim=-1,
        )

    return probe.to(device)


# %%

TRAIN_FIRST_PROBE = False
if TRAIN_FIRST_PROBE:
    args = ProbeTrainingArgs()
    litmodel = LitLinearProbe(model, args)

    logger = WandbLogger(save_dir=os.getcwd() + "/logs", project=args.probe_name)

    # Train the model
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        log_every_n_steps=1,
    )
    trainer.fit(model=litmodel)
    wandb.finish()

    new_probe = litmodel.linear_probe
    if not FIRST_PROBE_PATH.exists():
        t.save(new_probe, FIRST_PROBE_PATH)
        print(f"Saved probe to {FIRST_PROBE_PATH.resolve()}")

# %% Training an orthogonal probe

TRAIN_ORTHINGAL_PROBE = False
if TRAIN_ORTHINGAL_PROBE:
    args = ProbeTrainingArgs(probe_name="orthogonal_probe")
    lit_ortho_probe = LitLinearProbe(model, args, new_probe)

    logger = WandbLogger(save_dir=os.getcwd() + "/logs", project=args.probe_name)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        log_every_n_steps=1,
    )
    trainer.fit(model=lit_ortho_probe)
    wandb.finish()

    ortho_probe = lit_ortho_probe.linear_probe
    if not SECOND_PROBE_PATH.exists():
        t.save(ortho_probe, SECOND_PROBE_PATH)
        print(f"Saved probe to {SECOND_PROBE_PATH.resolve()}")
