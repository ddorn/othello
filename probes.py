import os

# os.environ["ACCELERATE_DISABLE_RICH"] = "1"
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import einops
import numpy as np
import torch as t
import torch.utils.data as data
import transformer_lens.utils as utils
from jaxtyping import Bool, Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer

from utils import *

try:
    from pytorch_lightning import LightningModule
    import pytorch_lightning as pl
except ValueError:
    pl = None
    LightningModule = object
    print("pytorch_lightning not working. Cannot train probes.")

PROBE_DIR = Path(__file__).parent / "probes"
PROBE_DIR.mkdir(exist_ok=True)


def get_probe(n: int = 0,
              flip_mine: bool = False,
              device="cpu",
              base_name="orthogonal_probe") -> Float[Tensor, "d_model rows cols options"]:
    """
    Load the probes that I trained.

    Args:
        n: The index of the probe to load. The n'th probe is orthogonal to all the previous ones.
        flip_mine: Whether to flip the mine and opponent channels (useful to fix probes trained on the wrong player)
        device: The device to load the probe onto.
        base_name: The base name of the probe to load. The full name is PROBE_DIR/{base_name}_{n}.pt

    Returns:
        A probe with shape (d_model, rows, cols, options)
        Options are:
            0: Blank
            1: Mine
            2: Opponent
    """

    path = PROBE_DIR / f"{base_name}_{n}.pt"
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


def get_neels_probe(merge_mine_theirs: bool = True,
                    device="cpu") -> Float[Tensor, "d_model rows=8 cols=8 options=3"]:
    """Returns the linear probe trained by Neel Nanda.

    Args:
        merge_mine_theirs: Whether to return the probe trained on mine and theirs separately.
            If False, return the probe trained on every move.

    Returns:
        The linear probe is a tensor of shape (d_model, rows, cols, options) where options are:
        - 0: blank
        - 1: my piece
        - 2: their piece
    """

    full_linear_probe: Float[Tensor, "mode=3 d_model rows=8 cols=8 options=3"] = t.load(
        OTHELLO_MECHINT_ROOT / "main_linear_probe.pth", map_location=device)

    blank_index = 0
    black_to_play_index = 1
    white_to_play_index = 2
    my_index = 1
    their_index = 2
    # (d_model, rows, cols, options)
    if merge_mine_theirs:
        linear_probe = t.zeros(512, 8, 8, 3, device=device)

        linear_probe[..., blank_index] = 0.5 * (full_linear_probe[black_to_play_index, ..., 0] +
                                                full_linear_probe[white_to_play_index, ..., 0])
        linear_probe[..., their_index] = 0.5 * (full_linear_probe[black_to_play_index, ..., 1] +
                                                full_linear_probe[white_to_play_index, ..., 2])
        linear_probe[..., my_index] = 0.5 * (full_linear_probe[black_to_play_index, ..., 2] +
                                             full_linear_probe[white_to_play_index, ..., 1])
    else:
        linear_probe = full_linear_probe[0]

    return linear_probe


@dataclass
class ProbeTrainingArgs:
    # Data  # TODO: Use a proper data loader, this is a hack for now
    train_tokens: Int[Tensor, "num_games full_game_len=60"]
    """Tokens between 0 and 60"""

    valid_tokens: Int[Tensor, "num_games full_game_len=60"]

    black_and_white: bool = False
    """Whether to predict black/white cells or mine/theirs"""

    correct_for_dataset_bias: Bool = False
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
        assert (self.train_tokens < 61).all(), (
            f"Train tokens should be between 0 and 60, got {self.train_tokens.max()}")
        assert (self.valid_tokens < 61).all(), (
            f"Valid tokens should be between 0 and 60, got {self.valid_tokens.max()}")

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


class LitLinearProbe(LightningModule):
    """A linear probe for a transformer with its training configured for pytorch-lightning."""

    dataset_stats: Optional[Float[Tensor, "move rows cols stat"]]

    def __init__(
        self,
        model: HookedTransformer,
        args: ProbeTrainingArgs,
        *old_probes: Float[Tensor, "d_model rows cols options"],
    ):
        assert LightningModule is not object, ("pytorch_lightning is not working. "
                                               "import it manually to see the error message.")

        super().__init__()

        self.model = model
        self.args = args
        self.linear_probe = t.nn.Parameter(args.setup_linear_probe(model))
        """shape: (d_model, rows, cols, options)"""
        self.old_probes = [
            (old_probe / t.norm(old_probe, dim=0)).to(self.model.cfg.device).detach()
            for old_probe in old_probes
        ]

        self.dataset_stats = None

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
            loss = loss / (self.dataset_stats[focus_moves] + 1e-5)
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

    def validation_step(
        self,
        batch: Tuple[Int[Tensor, "game move"], Int[Tensor, "game move row col"]],
        batch_idx: int,
    ) -> Float[Tensor, ""]:
        tokens, states = batch

        acc: Float[Tensor, "move option=3"]
        acc = plot_aggregate_metric(
            self.model,
            self.linear_probe,
            tokens,
            per_option=True,
            per_move="board_accuracy",
            black_and_white=self.args.black_and_white,
            plot=False,
        )
        acc_per_option = acc.mean(dim=0)

        self.log("Validation board accuracy", acc.mean())
        self.log("Validation board accuracy - blank", acc_per_option[0])
        self.log("Validation board accuracy - mine", acc_per_option[1])
        self.log("Validation board accuracy - their", acc_per_option[2])
        self.log("Validation board accuracy - Move variance", acc.mean(1).var())

        return acc.mean()

    def make_dataset(self, tokens: Float[Tensor, "game move"]):
        device = self.model.cfg.device

        states = move_sequence_to_state(
            TOKENS_TO_BOARD[tokens],
            "normal" if self.args.black_and_white else "alternate",
        ).to(device)

        return tokens.to(device), states

    def train_dataloader(self):
        """
        Returns `games_int` and `state_stack_one_hot` tensors.
        """
        tokens, states = self.make_dataset(self.args.train_tokens)

        if self.args.correct_for_dataset_bias:
            self.dataset_stats = einops.rearrange(compute_stats(states),
                                                  "stat move rows cols -> move rows cols stat")

        data_loader = data.DataLoader(
            data.TensorDataset(tokens, states),
            batch_size=self.args.batch_size,
            shuffle=True,
        )

        return data_loader

    def val_dataloader(self):
        return data.DataLoader(
            data.TensorDataset(*self.make_dataset(self.args.valid_tokens)),
            batch_size=len(self.args.valid_tokens),
        )

    def configure_optimizers(self):
        return t.optim.AdamW(
            [self.linear_probe],
            lr=self.args.lr,
            betas=self.args.betas,
            weight_decay=self.args.wd,
        )