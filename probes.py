import dataclasses
import os

from utils import *
import wandb

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import einops
import numpy as np
import torch as t
from rich import print as rprint
import torch.utils.data as data
import transformer_lens.utils as utils
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens import HookedTransformer
from plotting import plot_aggregate_metric


try:
    from pytorch_lightning import LightningModule
    import pytorch_lightning as pl
except ValueError:
    pl = None
    LightningModule = object
    print("pytorch_lightning not working. Cannot train probes.")

PROBE_DIR = Path(__file__).parent / "probes"
PROBE_DIR.mkdir(exist_ok=True)


def get_probe(
    n: int = 0, flip_mine: bool = False, device="cpu", base_name="orthogonal_probe"
) -> Float[Tensor, "d_model rows cols options"]:
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
        valid_probes = list(PROBE_DIR.glob(f"{base_name}_*.pt"))
        if len(valid_probes) == 0:
            raise FileNotFoundError(f"Could not find any probes in {PROBE_DIR}.")
        raise FileNotFoundError(
            f"Could not find probe {n} in {PROBE_DIR}. " f"Valid probes are {valid_probes}."
        )

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


def get_neels_probe(
    merge_mine_theirs: bool = True, device="cpu"
) -> Float[Tensor, "d_model rows=8 cols=8 options=3"]:
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
        OTHELLO_MECHINT_ROOT / "main_linear_probe.pth", map_location=device
    )

    blank_index = 0
    black_to_play_index = 1
    white_to_play_index = 2
    my_index = 1
    their_index = 2
    # (d_model, rows, cols, options)
    if merge_mine_theirs:
        linear_probe = t.zeros(512, 8, 8, 3, device=device)

        linear_probe[..., blank_index] = 0.5 * (
            full_linear_probe[black_to_play_index, ..., 0]
            + full_linear_probe[white_to_play_index, ..., 0]
        )
        linear_probe[..., their_index] = 0.5 * (
            full_linear_probe[black_to_play_index, ..., 1]
            + full_linear_probe[white_to_play_index, ..., 2]
        )
        linear_probe[..., my_index] = 0.5 * (
            full_linear_probe[black_to_play_index, ..., 2]
            + full_linear_probe[white_to_play_index, ..., 1]
        )
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
    """Whether to correct for options that are present more often, per move and per game"""

    # Which layer, and which positions in a game sequence to probe
    layer: int = 6
    pos_start: int = 5
    pos_end: int = (
        54  # 59 - 5. 59 is the second to last move in a game, last one which we make a prediction
    )

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

    # Orthogonal regularization (to another probe)
    penalty_weight: float = 1_000.0

    # Misc.
    probe_name: str = "main_linear_probe"

    def __post_init__(self):
        assert self.pos_start < self.pos_end, "pos_start should be smaller than pos_end"
        assert (
            self.train_tokens < 61
        ).all(), f"Train tokens should be between 0 and 60, got {self.train_tokens.max()}"
        assert (
            self.valid_tokens < 61
        ).all(), f"Valid tokens should be between 0 and 60, got {self.valid_tokens.max()}"

    def setup_linear_probe(
        self, model: HookedTransformer
    ) -> Float[Tensor, "d_model rows cols options"]:
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
        # and not be a computation of other tensors (here division by sqrt(d_model))
        # Thus, we can't use the `requires_grad` argument in the line above.
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
        assert LightningModule is not object, (
            "pytorch_lightning is not working. " "import it manually to see the error message."
        )

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
        probe_log_probs = probe_logits.log_softmax(dim=-1)
        loss = probe_log_probs * state_stack_one_hot.float()
        if self.args.correct_for_dataset_bias:
            loss = loss / (self.dataset_stats[focus_moves] + 1e-5)
        # Multiply to correct for the mean over options
        loss = -loss.mean() * self.args.options

        penalisation = t.tensor(0.0, device=self.model.cfg.device)
        for old_probe in self.old_probes:
            cosine_sim_sq = (
                einops.einsum(
                    old_probe,
                    self.linear_probe / t.norm(self.linear_probe, dim=0),
                    "d_model row col option, d_model row col option -> row col option",
                )
                ** 2
            )
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
            tokens,
            self.model,
            self.linear_probe,
            # per_option=True,
            per_move="board_accuracy",
            black_and_white=self.args.black_and_white,
            plot=False,
        )
        acc_per_option = acc.mean(dim=0)

        self.log("Validation board accuracy", acc.mean())
        # self.log("Validation board accuracy - blank", acc_per_option[0])
        # self.log("Validation board accuracy - mine", acc_per_option[1])
        # self.log("Validation board accuracy - their", acc_per_option[2])
        # self.log("Validation board accuracy - Move variance", acc.mean(1).var())

        return acc.mean()

    def make_dataset(self, tokens: Float[Tensor, "game move"]):
        device = self.model.cfg.device

        states = move_sequence_to_state(
            TOKENS_TO_BOARD[tokens],
            "black-white" if self.args.black_and_white else "mine-their",
        ).to(device)

        return tokens.to(device), states

    def train_dataloader(self):
        """
        Returns `games_int` and `state_stack_one_hot` tensors.
        """
        tokens, states = self.make_dataset(self.args.train_tokens)

        if self.args.correct_for_dataset_bias:
            self.dataset_stats = einops.rearrange(
                compute_stats(states), "stat move rows cols -> move rows cols stat"
            )

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


TokensType = Literal["tokens", "activations"]
Tokens = Union[Int[Tensor, "game move"], Float[Tensor, "game dmodel"]]


class Probe(t.nn.Module):
    """
    A linear probe from the activation of a model.

    This class is `num_probes` linear probes, each predicting `options` classes.

    To configure for a specific task (all optional):
    - create a Config subclass to add more options.
    - over
    """

    probe: Float[Tensor, "probes options dmodel"]

    @dataclass
    class Config:
        layer: int
        """The layer on which the probe is trained."""
        probe_point: str = "resid_post"
        """The activation with this name are the input of the probe."""

        num_probes: int = 1
        """How many probes to train."""
        options: int = 2
        """How many options each probe predicts."""

        device: str = "cuda"

        # Training
        use_wandb: bool = False
        seed: int = 42

        epochs: int = 4
        lr: float = 0.01
        wd: float = 0.0
        betas: Tuple[float, float] = (0.9, 0.99)

        batch_size: int = 500
        num_train_games: int = 10_000
        num_val_games: int = 200
        validate_every: int = 10

        @property
        def act_name(self) -> str:
            return utils.get_act_name(self.probe_point, self.layer)

        def name(self) -> str:
            return f"probe_{self.layer}_{self.act_name}_{self.num_probes}x{self.options}"

    def __init__(self, model: HookedTransformer, config: Config) -> None:
        super().__init__()
        self.config = config
        # This is a hack to avoid having the model registered as a submodule.
        self._model = (model,)
        self.probe = t.nn.Parameter(
            t.randn(config.num_probes, config.options, model.cfg.d_model, device=config.device)
            / np.sqrt(model.cfg.d_model)
        )

    @property
    def model(self) -> HookedTransformer:
        return self._model[0]

    # Step 0: Convert tokens to correct answers

    def get_correct_answers(
        self, tokens: Int[Tensor, "batch token"]
    ) -> Int[Tensor, "batch probe *activation"]:
        raise NotImplementedError("Override this method to compute the correct answers.")

    def dataloader(
        self, tokens: Int[Tensor, "game move"], for_validation: bool = False
    ) -> DataLoader:
        if for_validation:
            max_games = self.config.num_val_games
        else:
            max_games = self.config.num_train_games

        return DataLoader(
            TensorDataset(tokens[:max_games], self.get_correct_answers(tokens[:max_games])),
            batch_size=self.config.batch_size,
            shuffle=not for_validation,
        )

    # Step 1: Convert tokens to activations

    def get_activations(
        self, tokens: Int[Tensor, "batch token"]
    ) -> Float[Tensor, "batch *activation dmodel"]:
        """Transform a batch of tokens into activations to feed the probes."""

        # Get the residual stream of the model on the right layer
        with t.no_grad():
            tokens = tokens.to(self.config.device)
            _, cache = self.model.run_with_cache(
                tokens,
                names_filter=lambda n: n == self.config.act_name,
                stop_at_layer=self.config.layer + 1,
            )

        # Select the activations we want
        return self.select_activations(tokens, cache[self.config.act_name])

    def select_activations(
        self, tokens: Int[Tensor, "batch token"], cache: Float[Tensor, "batch *activation dmodel"]
    ) -> Float[Tensor, "batch *activation dmodel"]:
        return cache  # XXX

    # Step 1 bonus: Pre-compute activations

    def dataloader_activations(
        self,
        tokens: Int[Tensor, "game move"],
        batch_size: int = 500,
        for_validation: bool = False,
    ) -> DataLoader:
        if for_validation:
            max_games = self.config.num_val_games
        else:
            max_games = self.config.num_train_games

        # Convert the tokens into a dataloader to iterate over in batches
        tokens_loader = DataLoader(TensorDataset(tokens[:max_games]), batch_size=batch_size)

        # Big tensor to store the activations on cpu
        activations = t.empty(max_games, self.model.cfg.d_model, device="cpu")
        game = 0
        for (batch,) in tqdm(tokens_loader, desc="Computing activations"):
            activations[game : game + len(batch)] = self.get_activations(batch).cpu()
            game += len(batch)

        correct = self.get_correct_answers(tokens[:max_games])
        return DataLoader(
            TensorDataset(activations, correct),
            batch_size=self.config.batch_size,
            shuffle=not for_validation,
        )

    # Step 2: Apply the probe on the activations

    def forward(
        self,
        tokens: Union[Int[Tensor, "batch token"], Float[Tensor, "batch *activation dmodel"]],
    ) -> Float[Tensor, "batch probe *activation option"]:
        """
        Compute the predictions (logits) of the probe on the given tokens/activations.

        If tokens is a tensor of int64, it is passed through the model first to get the activations.
        Otherwise, if tokens is a tensor of float32, it is assumed to be activations and is passed directly to the probe.

        Args:
            tokens: A batch of tokens or activations.

        Returns:
            A tensor of shape [batch, probes, options] containing the logits of the probe.
        """

        # Compute the activations
        if tokens.dtype == t.float32:
            residual_stream = tokens
        elif tokens.dtype == t.int64:
            residual_stream = self.get_activations(tokens)
        else:
            raise ValueError(
                f"Expected tokens to be a tensor of float32 or int64 but got: {tokens.dtype}"
            )

        # Apply the probe
        return self.activation_forward(residual_stream.to(self.config.device))

    def activation_forward(
        self, activations: Float[Tensor, "*activation dmodel"]
    ) -> Float[Tensor, "batch probes *activation option"]:
        # By default, we multiply along the last dimension of both, and concatenate the other dims.

        return einops.einsum(
            self.probe,
            activations,
            "probe option dmodel, batch ... dmodel -> batch probe ... option",
        )

    # Step 3: Compute the loss

    def loss(
        self,
        tokens: Union[Int[Tensor, "batch token"], Float[Tensor, "batch *activation dmodel"]],
        correct: Int[Tensor, "batch probe *activation"],
        per_probe: bool = False,
        return_accuracy: bool = False,
    ) -> Tensor:
        """
        Compute the loss of the probe on a batch of games.

        Args:
            tokens: Either the tokens to feed into the model or the pre-recorded activations.
            correct: TODO: Document loss()
            per_probe: If True, return the loss per move. Otherwise, return the average loss.
            return_accuracy: Whether to also compute the accuracy of the probe. Return a tensor [loss, accuracy] if True.

        Returns:
            A tensor whose size depends on `per_move` and `return_accuracy`.
            - If `per_move` and `return_accuracy`, the tensor is of shape [2, moves].
            - If `per_move` and not `return_accuracy`, the tensor is of shape [moves].
            - If not `per_move` and `return_accuracy`, the tensor is of shape [2].
            - If not `per_move` and not `return_accuracy`, is a scalar.
        """

        # Get the predictions of the probe
        logits = self(tokens)
        logits: Int[Tensor, "batch probe *activation option"]

        # Ensure that we have 3 predictions
        # if not self.config.has_blank:
        #     # Add a fake blank option, with logits of -1000
        #     new_shape = logits.shape[:-1] + (3,)
        #     l = t.full(new_shape, -100, device=logits.device, dtype=logits.dtype)
        #     l[..., 1:] = logits
        #     logits = l

        # Ensure the predictions are on the right device and dtype
        correct = correct.to(device=self.config.device, dtype=t.long)

        # Compute the loss
        loss = t.nn.functional.cross_entropy(
            logits.flatten(0, -2),  # all but the `option` dimension
            correct.flatten(),
            reduction="none" if per_probe else "mean",
        )

        if per_probe:
            # mean over all but `probe` dimension
            dims = [d for d in range(loss.ndim) if d != 1]
            loss = loss.reshape(correct.shape).mean(dims)

        if not return_accuracy:
            return loss

        # Compute the accuracy
        # noinspection PyUnresolvedReferences
        accuracy = (logits.argmax(dim=-1) == correct).float()
        if per_probe:
            # noinspection PyUnboundLocalVariable
            accuracy = accuracy.mean(dims)  # mean on game, keep move
        else:
            accuracy = accuracy.mean()  # mean on game and move

        return t.stack([loss, accuracy])

    # Step 4: Train the probe

    def train_loop(
        self,
        train_data: DataLoader,
        val_data: DataLoader,
    ) -> None:
        rprint(self.config)
        make_deterministic(self.config.seed)
        t.set_grad_enabled(True)

        # Optimizer
        optim = t.optim.AdamW(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.wd,
            betas=self.config.betas,
        )

        step = 0
        nb_games = 0
        for epoch in trange(self.config.epochs, desc="Epoch"):
            for batch in train_data:
                optim.zero_grad()
                loss = self.loss(*batch)
                # Avoid having the gradients scale with number of probes
                (loss * t.tensor(self.config.num_probes)).backward()
                optim.step()

                if self.config.use_wandb:
                    wandb.log({"loss": loss.item(), "games": nb_games, "epoch": epoch})

                if step % self.config.validate_every == 0:
                    self.validate(val_data)

                step += 1
                nb_games += len(batch[0])

        val_loss = self.validate(val_data)
        print(f"End validation (loss, accuracy): {val_loss.mean(1)}")

    # Step 5: Evaluate the probe

    @t.inference_mode()
    def validate(
        self,
        val_loader: DataLoader,
        use_wandb: Optional[bool] = None,
    ) -> Float[Tensor, "metric=2 probe"]:
        # Compute the average loss and accuracy, taking into account that batches might have different sizes
        total_games = 0
        metric = t.zeros(2, self.config.num_probes, device=self.config.device)
        for batch in val_loader:
            batch_metric = self.loss(*batch, per_probe=True, return_accuracy=True)
            metric += batch_metric * len(batch[0])
            total_games += len(batch[0])
        metric /= total_games

        # Log the metrics if needed
        if self.config.use_wandb and use_wandb is not False:
            accu = metric[1]
            mid_pos = len(accu) // 2
            wandb.log(
                {
                    "val_acc_probe_0": accu[0],
                    f"val_acc_probe_{mid_pos}": accu[mid_pos],
                    "val_acc_second_last_probe": accu[-2],
                    "val_acc_last_probe": accu[-1],
                    "val_loss": metric[0].mean(),
                    "val_accuracy": accu.mean(),
                },
                commit=False,
            )
        return metric

    @classmethod
    def load(cls, model: HookedTransformer, **config_args) -> "Probe":
        config = cls.Config(**config_args)
        weight = t.load(PROBE_DIR / f"{config.name()}.pt")
        probe = cls(model, config)
        probe.probe = weight
        return probe

    def save(self, force: bool = False) -> None:
        path = PROBE_DIR / f"{self.config.name}.pt"
        if path.exists() and not force:
            print(f"Warning: {path.resolve()} already exists. Not saving the probe.")
            return
        t.save(self.probe, path)
        print(f"Probe saved to {path.resolve()}")


class OthelloProbe(Probe):
    @dataclass
    class Config(Probe.Config):
        cell: str = "A1"
        options: int = 3

        # @property
        # def row(self) -> int:
        #     return ord(self.cell[0].upper()) - ord("A")
        #
        # @property
        # def col(self) -> int:
        #     return int(self.cell[1])
        #
        # @property
        # def options(self) -> int:
        #     return 3 if self.has_blank else 2
        #
        def row_col(self) -> Tuple[int, int]:
            return ord(self.cell[0].upper()) - ord("A"), int(self.cell[1])

        @property
        def trained_on(self):
            return self.num_probes - 1

        def name(self) -> str:
            return f"probe-{self.cell}-L{self.layer}{self.probe_point}-M{self.trained_on}"

    config: Config

    def get_correct_answers(
        self, tokens: Int[Tensor, "batch token"]
    ) -> Int[Tensor, "batch probe *activation"]:
        assert (
            tokens.shape[1] == self.config.trained_on + 1
        ), f"Expected {self.config.trained_on + 1} moves, got {tokens.shape[1]}"

        row, col = self.config.row_col()

        states = move_sequence_to_state(tokens_to_board(tokens), mode="mine-their")
        states = states[:, :, row, col]
        states = state_stack_to_correct_option(states)
        states: Int[Tensor, "game move"]  # Index of the correct cell state

        if self.config.options == 2:  # Remove the blank option
            # noinspection PyUnresolvedReferences
            assert (states > 0).all(), "Blank option in the data!"
            states -= 1

        return states

    def select_activations(
        self, tokens: Int[Tensor, "game move"], cache: Float[Tensor, "game move dmodel"]
    ) -> Float[Tensor, "game dmodel"]:
        # Get the residual stream at the move where the probe is trained
        return cache[:, self.config.trained_on, :]


# Baselines


class ConstantProbe(OthelloProbe):
    def __init__(
        self, logits: List[float], model: HookedTransformer, config: OthelloProbe.Config
    ) -> None:
        super().__init__(model, config)
        self.logits = logits

    def forward(
        self,
        tokens: Tokens,
    ) -> Float[Tensor, "game first_moves option"]:
        assert tokens.dtype == t.int64, f"Expected int64, got {tokens.dtype}"
        l = t.tensor(self.logits, device=self.config.device, dtype=t.float32)
        return l[None, None].expand(*tokens.shape, -1)


class StatsProbe(OthelloProbe):
    logits: Float[Tensor, "move option"]

    def __init__(
        self,
        stats: Float[Tensor, "stat=3 move=60 row=8 col=8"],
        model: HookedTransformer,
        config: Probe.Config,
    ) -> None:
        super().__init__(model, config)
        assert stats.shape == (
            3,
            60,
            8,
            8,
        ), f"Expected stats to have shape (3, 60, 8, 8), got {stats.shape}"
        row, col = self.config.row_col()
        if self.config.options == 2:
            stats = stats[1:]
        # We go from the probability of each cell to logits that correspond to the
        # probabilities. Which are just the log of the probabilities.
        logits = t.log(stats[:, : self.config.trained_on + 1, row, col])
        self.logits = einops.rearrange(logits, "stat move -> move stat")

    def forward(
        self,
        tokens: Int[Tensor, "game move"],
        tokens_type: TokensType = "tokens",
    ) -> Float[Tensor, "game first_moves option"]:
        assert tokens_type == "tokens", "Heuristic probe only works with tokens"

        return self.logits[None].expand(tokens.shape[0], -1, -1).to(self.config.device)


def baselines(probes, val_data, type: Literal["random", "stats"] = "random"):
    if type == "stats":
        stats = t.load(STATS_PATH)
    else:
        stats = []

    baseline_probes = []
    for probe in probes:
        config = OthelloProbe.Config(**dataclasses.asdict(probe.config))
        config.use_wandb = False
        if type == "random":
            baseline_probes.append(ConstantProbe([1] * config.options, probe.model, config))
        elif type == "stats":
            baseline_probes.append(StatsProbe(stats[:3], probe.model, config))
        else:
            raise ValueError(f"Unknown baseline type {type}")

    return t.stack([probe.validate(val_data) for probe in baseline_probes])
