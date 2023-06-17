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
    """Wether to correct for options that are present more often, per move and per game"""

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

    # Othognal regularization (to an other probe)
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
        probe_logprobs = probe_logits.log_softmax(dim=-1)
        loss = probe_logprobs * state_stack_one_hot.float()
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


@dataclass
class ProbeConfig:
    cell: str
    """Which cell to predict. Eg. 'A2'."""
    trained_on: int
    """The probe will be applied only on the residual stream at this move."""
    layer: int
    """The layer on which the probe is trained."""
    probe_point: str = "resid_post"
    """The activation with this name are the input of the probe."""
    device: str = "cuda"
    trained_on_type: Literal["move", "token"] = "move"

    has_blank: bool = True

    # Training
    use_wandb: bool = False
    seed: int = 42

    epochs: int = 4
    lr: float = 0.01
    wd: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.99)
    val_loss_early_stop: float = 0.0

    batch_size: int = 500
    num_train_games: int = 10_000
    num_valid_games: int = 200
    validate_every: int = 10

    def __post_init__(self) -> None:
        assert self.cell in FULL_BOARD_LABELS
        assert 0 <= self.trained_on < 59

    @property
    def row(self) -> int:
        return ord(self.cell[0].upper()) - ord("A")

    @property
    def col(self) -> int:
        return int(self.cell[1])

    @property
    def options(self) -> int:
        return 3 if self.has_blank else 2

    @property
    def name(self) -> str:
        return f"probe-{self.cell}-L{self.layer}{self.probe_point}-M{self.trained_on}"

    @property
    def act_name(self) -> str:
        return utils.get_act_name(self.probe_point, self.layer)

    def load(self, model: HookedTransformer) -> "Probe":
        weight = t.load(PROBE_DIR / f"{self.name}.pt")
        probe = Probe(model, self)
        probe.probe = weight
        return probe


TokensType = Literal["tokens", "activations"]
Tokens = Union[Int[Tensor, "game move"], Float[Tensor, "game dmodel"]]


class Probe(t.nn.Module):
    """A probe that predicts the state of the board"""

    probe: Float[Tensor, "options move dmodel"]

    def __init__(self, model: HookedTransformer, config: ProbeConfig) -> None:
        super().__init__()
        self._model = (model,)
        self.probe = t.nn.Parameter(
            t.randn(config.options, config.trained_on + 1, model.cfg.d_model, device=config.device)
            / np.sqrt(model.cfg.d_model)
        )
        self.config = config

    @property
    def model(self) -> HookedTransformer:
        return self._model[0]

    def save(self, force: bool = False) -> None:
        path = PROBE_DIR / f"{self.config.name}.pt"
        if path.exists() and not force:
            print(f"Warning: {path.resolve()} already exists. Not saving the probe.")
            return
        t.save(self.probe, path)
        print(f"Probe saved to {path.resolve()}")

    def get_activations(self, tokens: Int[Tensor, "game move"]) -> Float[Tensor, "game dmodel"]:
        # Get the residual stream of the model on the right layer
        with t.no_grad():
            _, cache = self.model.run_with_cache(
                tokens[:, : self.config.trained_on + 1].to(self.config.device),
                names_filter=lambda n: n == self.config.act_name,
                stop_at_layer=self.config.layer + 1,
            )

        return self.select_activations(tokens, cache[self.config.act_name])

    def select_activations(
        self, tokens: Int[Tensor, "game move"], cache: Float[Tensor, "game move dmodel"]
    ) -> Float[Tensor, "game dmodel"]:
        # Get the residual stream at the move where the probe is trained
        return cache[self.config.act_name][:, self.config.trained_on, :]

    def forward(
        self,
        tokens: Tokens,
        tokens_type: TokensType = "tokens",
    ) -> Float[Tensor, "game first_moves option"]:
        # Compute the activations
        if tokens_type == "tokens":
            residual_stream = self.get_activations(tokens)
        elif tokens_type == "activations":
            residual_stream = tokens
        else:
            raise ValueError(f"Unknown tokens_type: {tokens_type}")

        # move is just the number of features / variables we want to predict
        # and each has 3 options

        # Apply the probe
        return einops.einsum(
            self.probe,
            residual_stream.to(self.config.device),
            "option move dmodel, game dmodel -> game move option",
        )

    def loss(
        self,
        tokens: Tokens,
        cell_state: Optional[Int[Tensor, "game move"]],
        tokens_type: TokensType = "tokens",
        per_move: bool = False,
        return_accuracy: bool = False,
    ) -> Tensor:
        """
        Compute the loss of the probe on a batch of games.

        Args:
            tokens: Either the tokens to feed into the model or the pre-recorded activations.
            cell_state: The correct predictions of the probe. This corresponds to output.argmax(dim=-1) for a perfect probe.
            tokens_type: if "tokens", tokens is the tokens to feed into the model. If "activations", tokens is the activations of the model.
            per_move: If True, return the loss per move. Otherwise, return the average loss.
            return_accuracy: Whether to also compute the accuracy of the probe. Return a tensor [loss, accuracy] if True.

        Returns:
            A tensor whose size depends on `per_move` and `return_accuracy`.
            - If `per_move` and `return_accuracy`, the tensor is of shape [2, moves].
            - If `per_move` and not `return_accuracy`, the tensor is of shape [moves].
            - If not `per_move` and `return_accuracy`, the tensor is of shape [2].
            - If not `per_move` and not `return_accuracy`, is a scalar.
        """

        # We don't need states further than the trained_on_move
        correct_options = cell_state[:, : self.config.trained_on + 1].to(
            device=self.config.device, dtype=t.long
        )

        # Get the predictions of the probe
        logits = self(tokens, tokens_type=tokens_type)

        # Ensure that we have 3 predictions
        if not self.config.has_blank:
            # Add a fake blank option, with logits of -1000
            new_shape = logits.shape[:-1] + (3,)
            l = t.full(new_shape, -100, device=logits.device, dtype=logits.dtype)
            l[..., 1:] = logits
            logits = l
        logits: Int[Tensor, "game move option=3"]

        # Compute the loss
        loss = t.nn.functional.cross_entropy(
            logits.flatten(0, 1),
            correct_options.flatten(),
            reduction="none" if per_move else "mean",
        )
        if per_move:
            loss = loss.reshape(correct_options.shape).mean(0)  # mean on game, keep move

        if not return_accuracy:
            return loss

        # noinspection PyUnresolvedReferences
        accuracy = (logits.argmax(dim=-1) == correct_options).float()
        if per_move:
            accuracy = accuracy.mean(0)  # mean on game, keep move
        else:
            accuracy = accuracy.mean()  # mean on game and move

        return t.stack([loss, accuracy])

    def _parse_data(
        self,
        data: Union[Int[Tensor, "game move"], DataLoader],
        data_type: TokensType,
        shuffle: bool = True,
    ) -> DataLoader:
        assert data_type in ["tokens", "activations"]
        if isinstance(data, Tensor):
            assert data_type == "tokens", "If train_tokens is a tensor, train_type must be 'tokens'"
            return self.dataloader(data, self.config.num_train_games, shuffle=shuffle)
        else:
            assert isinstance(
                data, DataLoader
            ), f"train_data must be a tensor or a dataloader, got: {type(data)}"
            return data

    def train_loop(
        self,
        train_data: Union[Int[Tensor, "game move"], DataLoader],
        val_data: Union[Int[Tensor, "game move"], DataLoader],
        train_type: TokensType = "tokens",
        val_type: TokensType = "tokens",
    ) -> None:
        rprint(self.config)
        make_deterministic(self.config.seed)
        t.set_grad_enabled(True)

        # Get the dataloaders
        train_loader = self._parse_data(train_data, train_type)
        val_loader = self._parse_data(val_data, val_type)

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
            for batch in train_loader:
                optim.zero_grad()
                loss = self.loss(*batch, tokens_type=train_type)
                # We correct for the mean on the move, so that the learning rate does
                # not depend on the number of moves.
                loss = loss * t.tensor(self.config.trained_on + 1)
                loss.backward()
                optim.step()

                if self.config.use_wandb:
                    wandb.log({"loss": loss.item(), "games": nb_games, "epoch": epoch})

                if step % self.config.validate_every == 0:
                    val_loss = self.validate(val_loader, val_type)
                    if val_loss.max() < self.config.val_loss_early_stop:
                        return

                step += 1
                nb_games += len(batch[0])

        val_loss = self.validate(val_loader, val_type)
        print(f"End validation (loss, accuracy): {val_loss.mean(1)}")

    @t.inference_mode()
    def validate(
        self,
        val_loader: DataLoader,
        val_type: TokensType = "tokens",
        use_wandb: Optional[bool] = None,
    ) -> Float[Tensor, "metric=2 move"]:
        metric = t.zeros(2, self.config.trained_on + 1, device=self.config.device)
        total_games = 0
        for batch in val_loader:
            batch_metric = self.loss(
                *batch, tokens_type=val_type, per_move=True, return_accuracy=True
            )
            metric += batch_metric * len(batch)
            total_games += len(batch)
        metric /= total_games

        mid_pos = metric.shape[1] // 2
        if self.config.use_wandb and use_wandb is not False:
            accu = metric[1]
            wandb.log(
                {
                    "val_acc_move_0": accu[0],
                    f"val_acc_move_{mid_pos}": accu[mid_pos],
                    "val_acc_move_second_last": accu[-3],
                    "val_acc_move_prev": accu[-2],
                    "val_acc_move_self": accu[-1],
                    "val_loss": metric[0].mean(),
                    "val_accuracy": accu.mean(),
                },
                commit=False,
            )
        return metric

    def dataset(self, tokens: Int[Tensor, "game move"], max_games: int) -> TensorDataset:
        assert (
            tokens.shape[0] >= max_games
        ), f"Only {tokens.shape[0]} games available but {max_games} requested"
        assert (
            tokens.shape[1] >= self.config.trained_on + 1
        ), f"Only {tokens.shape[1]} moves available but {self.config.trained_on + 1} requested"

        tokens = tokens[:max_games, : self.config.trained_on + 1]
        states = move_sequence_to_state(tokens_to_board(tokens), mode="mine-their")
        states = states[:max_games, :, self.config.row, self.config.col]
        states = state_stack_to_correct_option(states)
        states: Int[Tensor, "game move"]  # Index of the correct cell state

        return TensorDataset(tokens, states)

    def dataloader(
        self, tokens: Int[Tensor, "game move"], max_games: int, shuffle: bool = True
    ) -> DataLoader:
        return DataLoader(
            self.dataset(tokens, max_games),
            batch_size=self.config.batch_size,
            shuffle=shuffle,
        )

    def dataloader_activations(
        self,
        tokens: Int[Tensor, "game move"],
        max_games: int,
        shuffle: bool = True,
        batch_size: int = 500,
    ) -> DataLoader:
        tokens_data = self.dataset(tokens, max_games)
        tokens_loader = DataLoader(tokens_data, batch_size=batch_size)

        activations = t.empty(max_games, self.model.cfg.d_model, device="cpu")
        game = 0
        for tokens, _ in tqdm(tokens_loader, desc="Computing activations"):
            act = self.get_activations(tokens)
            activations[game : game + len(tokens)] = act.cpu()
            game += len(tokens)

        correct = tokens_data.tensors[1]
        return DataLoader(
            TensorDataset(activations, correct),
            batch_size=self.config.batch_size,
            shuffle=shuffle,
        )


class ConstantProbe(Probe):
    def __init__(self, logits: List[float], model: HookedTransformer, config: ProbeConfig) -> None:
        super().__init__(model, config)
        self.logits = logits

    def forward(
        self,
        tokens: Tokens,
        tokens_type: TokensType = "tokens",
    ) -> Float[Tensor, "game first_moves option"]:
        assert tokens_type == "tokens", "Heuristic probe only works with tokens"
        l = t.tensor(self.logits, device=self.config.device)
        return l[None, None].expand(*tokens.shape, -1)


class StatsProbe(Probe):
    logits: Float[Tensor, "move option"]

    def __init__(
        self,
        stats: Float[Tensor, "stat=3 move=60 row=8 col=8"],
        model: HookedTransformer,
        config: ProbeConfig,
    ) -> None:
        super().__init__(model, config)
        assert stats.shape == (3, 60, 8, 8)
        # We go from the probability of each cell to logits that correspond to the
        # probabilities. Which are just the log of the probabilities.
        logits = t.log(
            stats[
                1 - self.config.has_blank :,
                : self.config.trained_on + 1,
                self.config.row,
                self.config.col,
            ]
        )
        self.logits = einops.rearrange(logits, "stat move -> move stat")

    def forward(
        self,
        tokens: Int[Tensor, "game move"],
        tokens_type: TokensType = "tokens",
    ) -> Float[Tensor, "game first_moves option"]:
        assert tokens_type == "tokens", "Heuristic probe only works with tokens"

        return self.logits[None].expand(tokens.shape[0], -1, -1).to(self.config.device)
