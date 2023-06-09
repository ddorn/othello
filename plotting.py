# %%
from dataclasses import dataclass
from functools import cached_property
import os
from typing import Dict, List, Literal, Optional

import einops
import plotly.express as px
import plotly.graph_objects as go

import torch as t
from IPython.display import display
from jaxtyping import Float, Int, Bool
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from transformer_lens import HookedTransformer

from plotly_utils import imshow
from utils import *


# %%
def plot_square_as_board(state: Float[Tensor, "... rows=8 cols=8"],
                         diverging_scale: bool = True,
                         **kwargs):
    """Takes a square input (8 by 8) and plot it as a board. Can do a stack of boards via facet_col=0"""
    kwargs = {
        "y": list("ABCDEFGH"),
        "x": [str(i) for i in range(8)],
        # "color_continuous_scale": "plasma" if diverging_scale else "Blues",
        "color_continuous_scale": "RdBu" if diverging_scale else "Blues",
        "color_continuous_midpoint": 0.0 if diverging_scale else None,
        "aspect": "equal",
        **kwargs,
    }
    imshow(state, **kwargs)


def plot_similarities(vectors: Float[Tensor, "*n_vectors dim"], **kwargs):
    """Plot the dot product between each pair of vectors"""
    vectors = vectors.flatten(end_dim=-2)
    sim = einops.einsum(vectors, vectors, "vec_1 dim, vec_2 dim -> vec_1 vec_2")
    imshow(sim, **kwargs)


def plot_similarities_2(
    v1: Float[Tensor, "*n_vectors rows cols"],
    v2: Float[Tensor, "*n_vectors rows cols"],
    name: str = "vectors",
):
    """Plot the dot product between each pair of vectors"""
    if v1.ndim > 2:
        v1 = v1.flatten(end_dim=-3)
    if v2.ndim > 2:
        v2 = v2.flatten(end_dim=-3)
    sim = einops.einsum(
        v1 / t.norm(v1, dim=0),
        v2 / t.norm(v2, dim=0),
        "d_model rows cols, d_model rows cols -> rows cols",
    )
    plot_square_as_board(sim, title=f"Cosine similarity between {name}")


def plot_PCA(
    vectors: Float[Tensor, "*n_vectors dim"],
    name: str = "",
    absolute: bool = False,
    flip_dim_order: bool = False,
):
    """Plot the PCA of the vectors

    Args:
        vectors (Float[Tensor, "*n_vectors dim"): The vectors to do the PCA on
        name (str, optional): The name for the plot.
        absolute (bool, optional): If true, plots the explained variance instead of the ratio.
        flip_dim_order (bool, optional): If true, the first dimension of the input is
            the dimension of the vectors. Otherwise it is the last.
    """

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
        px.bar(
            x=range(len(pca.explained_variance_ratio_)),
            y=y,
            title=f"Explained variance ratio of the PCA on {name}",
        ))

    return pca


@dataclass(frozen=True)
class ModelMetricPlotter:
    tokens: Int[Tensor, "batch game_len"]
    model: HookedTransformer
    pos_start: int = 0
    pos_end: int = -1
    # options_stats: Optional[Float[Tensor, "stat=3 move row col"]] = None

    @cached_property
    @t.inference_mode()
    def out(self) -> Float[Tensor, "game move cell"]:
        out = logits_to_board(self.model(self.tokens[:, :59]), "logits")
        return out[:, self.focus_moves].flatten(start_dim=-2)

    @cached_property
    @t.inference_mode()
    def expected(self) -> Bool[Tensor, "game move cell"]:
        is_valid = move_sequence_to_state(tokens_to_board(self.tokens), "valid")
        return is_valid[:, self.focus_moves].flatten(start_dim=-2).to(self.out.device)

    @cached_property
    def nb_valid_moves(self) -> Int[Tensor, "game move"]:
        return self.expected.sum(dim=-1)

    @property
    def focus_moves(self) -> slice:
        pos_end = self.pos_end % self.tokens.shape[1]
        return slice(self.pos_start, pos_end)

    # Different metrics

    @cached_property
    def loss(self) -> Float[Tensor, "game move"]:
        """Loss of the model per move/token"""
        l = -self.out.log_softmax(dim=-1) * self.expected.float()
        return l.sum(dim=-1)

    @cached_property
    def loss_scaled(self) -> Float[Tensor, "game move"]:
        """Loss of the model per move/token, scaled by the number of valid moves"""
        metric = self.out.log_softmax(dim=-1) * self.expected.float()
        metric = -metric.sum(dim=-1)
        return metric / self.nb_valid_moves

    @cached_property
    def high_logit(self) -> Float[Tensor, "game move cell"]:
        """1 when the probability of the cell is higher than 1/2 * nb_valid_moves, 0 otherwise"""
        predicted = self.out.softmax(-1) > 1 / (2 * self.nb_valid_moves[..., None])
        return (predicted == self.expected).float()

    @cached_property
    def scaled_probabilities(self):
        metric = self.out.softmax(dim=-1)
        metric *= self.nb_valid_moves[..., None]
        metric[~self.expected] = 1 - metric[~self.expected]
        return metric

    def reduce(self, metric: Float[Tensor, "game move *cell"],
               mode: Literal['per_cell', 'per_move', 'board_sum', 'board_prod'] = "per_move") -> Tensor:
        assert mode in ("per_cell", "per_move", "board_sum", "board_prod"), f"Unknown mode: {mode}"
        if metric.ndim != 3:
            assert (mode == "per_move"
                    ), f"Can only reduce per_move for 3-dim tensors, got {metric.shape}"
            return metric.mean(0)

        # Reduce the metric to the plot we want
        if mode == "per_cell":  # -> 8x8
            metric = metric.mean(dim=(0, 1)).reshape(8, 8)
        elif mode == "per_move":  # -> moves
            metric = metric.mean(dim=(0, 2))
        elif mode == "board_sum":  # moves
            metric = metric.sum(2).mean(0)
        elif mode == "board_prod":  # moves
            metric = metric.prod(2).mean(0)
        else:
            raise ValueError(f"Unknown per_move: {mode}")

        return metric

    # Different plots

    def plot_loss_per_move(self, rescaled: bool = False, name: str = "OthelloGPT"):
        title = "Loss scaled by nb of valid moves" if rescaled else "Loss"
        self.plot_by_move(
            f"{title} of {name}",
            [title],
            self.reduce(self.loss_scaled if rescaled else self.loss, "per_move"),
            yaxis_title="Loss",
        )

    def plot_whole_board_accuracy(self, name: str = "OthelloGPT"):
        self.plot_by_move(
            f"Frequency of the whole board being correct for {name}",
            ["Correct board"],
            self.reduce(self.high_logit, "board_prod"),
            yaxis_title="Frequency",
        )


    def plot_by_move(
        self,
        title: str,
        labels: List[str],
        *lines: Float[Tensor, "x"],
        yaxis_title: str = "Accuracy",
    ):
        assert len(labels) == len(lines), f"Got {len(labels)} labels and {len(lines)} lines"

        fig = go.Figure()
        for line, label in zip(lines, labels):
            fig.add_trace(
                go.Scatter(
                    x=t.arange(self.pos_start, self.pos_end % self.tokens.shape[1]),
                    y=line.cpu(),
                    name=label,
                ))
        fig.update_layout(
            title=title,
            xaxis_title="Move",
            yaxis_title=yaxis_title,
            legend_title="Legend",
        )
        fig.show()


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
    prediction: Literal["argmax", "softmax", "logprob"] = "argmax",
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
    mode = "black-white" if black_and_white else "mine-their"
    states = move_sequence_to_state(tokens_to_board(tokens), mode)
    correct_one_hot = state_stack_to_one_hot(states.to(model.cfg.device))
    correct_one_hot: Bool[Tensor, "game move row col options"]

    # Remove the first and last moves
    pos_end = pos_end % tokens.shape[1]
    out = out[:, pos_start:pos_end]
    correct_one_hot = correct_one_hot[:, pos_start:pos_end]

    # Transform the probe output into the plotted metric
    if prediction == "argmax":
        probe_values = out.argmax(dim=-1, keepdim=True)
        metric = t.zeros_like(out)
        metric.scatter_(-1, probe_values, 1)
        # correct = (probe_one_hot == states_one_hot).float()
    elif prediction == "softmax":
        metric = out.softmax(dim=-1)
        # correct[states_one_hot] = probs[states_one_hot]
        # correct[~states_one_hot] = 1 - probs[~states_one_hot]
    elif prediction == "logprob":
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
    if per_move == "no":
        # Mean over games and moves
        metric = metric.mean(dim=(0, 1))
    elif per_move == "board_accuracy":
        # Reduce the cell (row+col) dimension
        if prediction == "logprob":
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
        "softmax": "Average probability",
        "logprob": "Average log probability",
        "argmax": "Average argmax",
    }[prediction]
    what = {
        "no": "each cell being wrong",
        "board_accuracy": "the whole board being correct",
        "cell_accuracy": "each cell being correct",
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

def explain_game(
    tokens: Int[Tensor, 'move'],
    model: HookedTransformer,
    move: int = None,
    # Extra args
    extra_boards: Optional[Dict[str, Tensor]] = None,
) -> None:
    if isinstance(tokens, list):
        tokens = t.tensor(tokens, dtype=t.long)
    if move is None:
        move = tokens.shape[0] - 1
    move = move % tokens.shape[0]
    if extra_boards is None:
        extra_boards = {}

    tokens = tokens[None, :move+1]
    board_indices = TOKENS_TO_BOARD[tokens]

    logits = model(tokens)

    scale = logits.abs().max() * 0.5
    predictions = logits_to_board(logits, 'log_prob')
    state = move_sequence_to_state(board_indices, mode='black-white') * scale
    valid = move_sequence_to_state(board_indices, mode="valid") * scale
    turn = move_sequence_to_state(board_indices, mode="turn")
    if turn[0, -1, 0, 0].item() == +1:
        turn_label = "Blue"
    else:
        turn_label = "Red"

    # Scale extra boards if they are boolean
    for k, v in extra_boards.items():
        if v.dtype == t.bool:
            extra_boards[k] = v * scale

    # Plotting
    boards = t.stack([
        state[0, -1],
        valid[0, -1],
        predictions[0, -1],
        *extra_boards.values(),
    ], dim=0)
    plot_square_as_board(
        boards,
        facet_col=0,
        facet_col_wrap=3 if len(boards) != 4 else 2,
        facet_labels=["State before", "Valid moves", "Model logits", *extra_boards.keys()],
        title=f"Game after {to_board_label(board_indices[0, -1].item())} - {turn_label} to play - move {move}",
    )


@t.inference_mode()
def find_failed_predictions(
    model: HookedTransformer,
    tokens: Int[Tensor, 'game move'],
    threshold: float = 2,
    nb_examples: int = 10,
) -> None:
    """
    Find sequences of moves where the model fails to predict the valid moves.

    Plots the board for each datapoint where the model fails to predict the valid moves.
    """

    # Get the probabilities for each cell
    logits = model(tokens[:, :59])
    probabilities = logits_to_board(logits, 'prob')  # [game, move, row, col]
    # Compute the number of valid moves
    valid_moves = move_sequence_to_state(TOKENS_TO_BOARD[tokens], mode="valid")[:, :59]
    nb_valid_moves = valid_moves.sum(dim=(-1, -2), keepdim=True)  # [game, move, 1, 1]

    # Find which moves are considered correct
    predictions = probabilities > (1 / (threshold * nb_valid_moves))
    correct = predictions == valid_moves
    correct_boards = (predictions == valid_moves).all(-1).all(-1)

    # Find the indices of the incorrect moves
    incorrect_indices = t.nonzero(~correct_boards, as_tuple=False)
    # Sample from the incorrect indices
    incorrect_indices = incorrect_indices[t.randperm(len(incorrect_indices))]
    incorrect_indices = incorrect_indices[:nb_examples]

    # Plot the incorrect moves
    for game, move in incorrect_indices:
        explain_game(tokens[game],
                     model,
                     move,
                     extra_boards={
                         "Model's mistakes": ~correct[game, move],
                     })
        # plot_board(TOKENS_TO_BOARD[tokens[game]])


def logit_attribution_emb_ov_circuit(model: HookedTransformer,
                                     layer: int = 0,
                                     include_out_bias: bool = False,
                                     include_unembed_bias: bool = False,
                                     ignore_pass: bool = False) -> None:
    """
    Show the direct logit attribution of the token embedding through the OV circuit of a layer.

    The bias terms are are expluded by default, because they are constant and do not interact with the embeddings.
    They are also the same for avery head. If included, remember this when interpreting the plot.

    Args:
        model (HookedTransformer): The model.
        layer (int, optional): Plot the OV circuit of the heads of this layer. Defaults to 0.
        include_out_bias (bool, optional): Whether to include the output bias (b_O).
        include_unembed_bias (bool, optional): Whether to include the unembedding bias (b_U).
        ignore_pass (bool, optional): Whether to plot information about the pass token. Defaults to False.
    """
    # We * Wov * Wu
    out = model.W_E @ model.W_V[layer] @ model.W_O[layer]
    if include_out_bias:
        out += model.b_O[layer]
    out = out @ model.W_U
    if include_unembed_bias:
        out += model.b_U

    if ignore_pass:
        out[:, 0] = 0
        out[:, :, 0] = 0

    imshow(
        out,
        facet_col=0,
        facet_col_wrap=3,
        height=1200,
        x=TOKEN_NAMES,
        y=TOKEN_NAMES,
        title=f"Direct logit attribution of the embedding through the OV circuit of layer {layer}",
    )


__all__ = [
    "plot_square_as_board",
    "plot_similarities",
    "plot_similarities_2",
    "plot_PCA",
    "ModelMetricPlotter",
    "plot_aggregate_metric",
    "find_failed_predictions",
    "explain_game",
    "logit_attribution_emb_ov_circuit",
]
