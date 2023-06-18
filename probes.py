"""
Collection of probes for Othello and reusable base classes.
"""

import copy
from pathlib import Path
from typing import Tuple, List, Literal, Union, Optional

import einops
import numpy as np
import torch as t
import transformer_lens.utils as utils
from dataclasses import dataclass
from jaxtyping import Float, Int
from rich import print as rprint
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm.autonotebook import tqdm, trange
from transformer_lens import HookedTransformer

import wandb
from magic_tensor import MagicTensor
from utils import (
    OTHELLO_MECHINT_ROOT,
    make_deterministic,
    board_label_to_row_col,
    move_sequence_to_state,
    TOKEN_NAMES,
    state_stack_to_correct_option,
    tokens_to_board,
    STATS_PATH,
)

PROBE_DIR = Path(__file__).parent / "probes"
PROBE_DIR.mkdir(exist_ok=True)


def get_neels_probe(
    merge_mine_theirs: bool = True, device: str = "cpu"
) -> Float[Tensor, "d_model rows=8 cols=8 options=3"]:
    """Returns the linear probe trained by Neel Nanda.

    Args:
        merge_mine_theirs: Whether to return the probe trained on mine and theirs separately.
            If False, return the probe trained on every move.
        device: The device to load the probe onto.

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

    ACTIVATIONS_EXTRA_SHAPE: Tuple[int, ...] = ()

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
        activations = t.empty(
            max_games, *self.ACTIVATIONS_EXTRA_SHAPE, self.model.cfg.d_model, device="cpu"
        )
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
            f"probe option dmodel, batch ... dmodel -> batch probe ... option",
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
            tokens: Either the tokens to feed into the model (it dtype==int64) or the pre-recorded activations (if dtype==float32).
            correct: The correct answers for the batch of games.
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
        # noinspection PyUnresolvedReferences
        if not ((correct >= 0).all() and (correct < logits.shape[-1]).all()):
            # I had bugs about this too many times to not put a check. Otherwise, it crashes the kernel.
            print("correct", correct, correct.min(), correct.max())
            raise ValueError()
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
        """
        Train the probe on the given data.

        Args:
            train_data: Either a DataLoader of (tokens, correct) or (activations, correct).
            val_data: Either a DataLoader of (tokens, correct) or (activations, correct).
        """

        rprint(self.config)
        make_deterministic(self.config.seed)
        t.set_grad_enabled(True)

        # Optimizer
        optim = t.optim.SGD(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.wd,
            # betas=self.config.betas,
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
        """
        Evaluate the probe on the given data.

        Args:
            val_loader: Either a DataLoader of (tokens, correct) or (activations, correct).
            use_wandb: If given, override the config.use_wandb setting.

        Returns:
            See `loss`.
        """

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
        """Load a probe from disk."""
        config = cls.Config(**config_args)
        weight = t.load(PROBE_DIR / f"{config.name()}.pt")
        probe = cls(model, config)
        probe.probe = weight
        return probe

    def save(self, force: bool = False) -> None:
        path = PROBE_DIR / f"{self.config.name()}.pt"
        if path.exists() and not force:
            print(f"Warning: {path.resolve()} already exists. Not saving the probe.")
            return
        t.save(self.probe, path)
        print(f"Probe saved to {path.resolve()}")

    def check_behaviour(
        self, tokens: Int[Tensor, "batch token"], nb_games: int = 1
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Check that the probe behaves as expected on a subset of the data."""
        print(self)

        tokens = tokens[:nb_games]
        print("Tokens shape:", tuple(tokens.shape))
        print("• Expected:", f"(batch=1, tokens)")
        print("•", tokens[0])

        correct = self.get_correct_answers(tokens)
        correct: Int[Tensor, "batch probe *activation"]
        print("Correct shape:", tuple(correct.shape))
        print(f"• Expected: (batch=1, probe={self.config.num_probes}, *activation)")
        print("•", correct[0])

        activations = self.get_activations(tokens)
        activations: Float[Tensor, "batch probe *activation"]
        print("Activations shape:", tuple(activations.shape))
        print(f"• Expected: (batch=1, *activation, dmodel={self.model.cfg.d_model})")
        print("•", activations[0, ..., :10], "truncated on dmodel")

        logits = self(tokens)
        logits: Float[Tensor, "batch probe *activation option"]
        print("Logits shape:", tuple(logits.shape))
        print(
            f"• Expected: (batch=1, probe={self.config.num_probes}, *activation, option={self.config.options})"
        )

        return tokens.cpu(), correct.cpu(), activations.cpu(), logits.cpu()


class OthelloProbe(Probe):
    @dataclass
    class Config(Probe.Config):
        cell: str = "A1"
        options: int = 3

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

        row, col = board_label_to_row_col(self.config.cell)

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

    def check_behaviour(self, tokens: Int[Tensor, "batch token"], nb_games: int = 1) -> None:
        out = tokens, correct, activations, logits = super().check_behaviour(tokens, nb_games)

        predicted = logits.argmax(-1)
        # noinspection PyUnresolvedReferences
        mistakes = (predicted != correct).int()
        (
            MagicTensor(t.stack([correct, predicted, mistakes], dim=1), ["game", "kind", "probe"])
            .tokens_to_board("probe")
            .plot(facet_labels=["Correct", "Predicted", "Mistakes"])
        )

        # plot_board(tokens_to_board(tokens)[0])
        logits = t.cat([logits, predicted.unsqueeze(-1).float()], dim=-1)
        (
            MagicTensor(logits, ["game", "probe", "option"])
            .tokens_to_board("probe", float("nan"))
            .plot(facet_labels=["Blank", "Mine", "Their", "Correct"])
        )

        # return out


class OthelloProbeWhenCellPlayed(OthelloProbe):
    @dataclass
    class Config(OthelloProbe.Config):
        target_cell: str = "A1"
        num_probes: int = 1
        mode: Literal["mine-their", "black-white"] = "mine-their"

        def name(self) -> str:
            return f"probe-{self.cell}-L{self.layer}{self.probe_point}-when-{self.target_cell}"

    config: Config

    def select_activations(
        self, tokens: Int[Tensor, "game move"], cache: Float[Tensor, "game move dmodel"]
    ) -> Float[Tensor, "game dmodel"]:
        # Get the residual stream at the move where the target cell is played
        target_token = TOKEN_NAMES.index(self.config.target_cell)
        activations = t.zeros(cache.shape[0], cache.shape[-1], device=cache.device)
        for i, (game, act) in enumerate(zip(tokens, cache)):
            mask = game == target_token
            # noinspection PyUnresolvedReferences
            if not mask.sum() == 1:
                print(f"Warning: no target ({self.config.target_cell}) in game", i)
                # print("Game:", " ".join(TOKEN_NAMES[t] for t in game))
                activations[i] = act[-1]
            else:
                activations[i] = act[mask]
        return activations

    def get_correct_answers(
        self, tokens: Int[Tensor, "batch token"]
    ) -> Int[Tensor, "batch probe *activation"]:
        # assert (
        #     tokens.shape[1] == self.config.trained_on + 1
        # ), f"Expected {self.config.trained_on + 1} moves, got {tokens.shape[1]}"

        row, col = board_label_to_row_col(self.config.cell)
        target_token = TOKEN_NAMES.index(self.config.target_cell)
        states = move_sequence_to_state(tokens_to_board(tokens), mode=self.config.mode)

        good_states = t.zeros(tokens.shape[0], 1, device=tokens.device, dtype=t.long)

        for i, (game, state) in enumerate(zip(tokens, states)):
            mask = game == target_token
            # noinspection PyUnresolvedReferences
            if not mask.sum() == 1:
                print(f"Warning: no target ({self.config.target_cell}) in game", i)
                # print("Game:", " ".join(TOKEN_NAMES[t] for t in game))
                good_states[i] = 1 if np.random.rand() < 0.5 else -1
            else:
                good_states[i] = state[mask, row, col]
        good_states = state_stack_to_correct_option(good_states)
        good_states: Int[Tensor, "game 1"]  # Index of the correct cell state

        if self.config.options == 2:  # Remove the blank option
            # noinspection PyUnresolvedReferences
            assert (good_states > 0).all(), "Blank option in the data!"
            good_states -= 1

        return good_states


class OthelloProbePredictOnPlay(OthelloProbe):
    ACTIVATIONS_EXTRA_SHAPE = (61,)

    @dataclass
    class Config(OthelloProbe.Config):
        num_probes: int = 61
        mode: Literal["mine-their", "black-white"] = "mine-their"

        def name(self) -> str:
            return f"probe-{self.cell}-L{self.layer}{self.probe_point}-predict-on-play"

    config: Config

    def select_activations(
        self, tokens: Int[Tensor, "game move"], cache: Float[Tensor, "game move dmodel"]
    ) -> Float[Tensor, "game dmodel"]:
        activations = t.zeros(
            cache.shape[0], self.config.num_probes, cache.shape[-1], device=cache.device
        )

        act: Tensor
        for i, (game, act) in enumerate(zip(tokens, cache)):
            for m, move in enumerate(game):
                activations[i, move] = act[m]
        return activations

    def get_correct_answers(
        self, tokens: Int[Tensor, "batch token"]
    ) -> Int[Tensor, "batch probe *activation"]:
        correct = t.zeros(tokens.shape[0], self.config.num_probes, device=tokens.device)
        states = move_sequence_to_state(tokens_to_board(tokens), mode=self.config.mode)
        row, col = board_label_to_row_col(self.config.cell)
        states = states[:, :, row, col]

        for i, (game, state) in enumerate(zip(tokens, states)):
            correct[i, game] = state
        return state_stack_to_correct_option(correct)

    def activation_forward(
        self, activations: Float[Tensor, "*activation dmodel"]
    ) -> Float[Tensor, "batch probes *activation option"]:
        return einops.einsum(
            self.probe,
            activations,
            f"probe option dmodel, batch probe dmodel -> batch probe option",
        )


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
        return l[None, None].expand(tokens.shape[0], self.config.num_probes, -1)


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
        row, col = board_label_to_row_col(self.config.cell)
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


def baselines(probes, val_data, type: Literal["mine", "their", "stats"] = "mine"):
    if type == "stats":
        stats = t.load(STATS_PATH)
    else:
        stats = []

    baseline_probes = []
    for probe in probes:
        config = copy.deepcopy(probe.config)
        config.use_wandb = False
        if type == "stats":
            baseline_probes.append(StatsProbe(stats[:3], probe.model, config))
        else:
            if config.options == 2:
                logits = [1.1, 1.0]
            elif config.options == 3:
                logits = [0, 4.1, 4.0]
            else:
                logits = [1.0] * config.options
            if type == "their":
                logits[-2:] = logits[-2:][::-1]
            baseline_probes.append(ConstantProbe(logits, probe.model, config))

    return t.stack([probe.validate(val_data) for probe in baseline_probes])
