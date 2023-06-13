# %%
from dataclasses import dataclass
from functools import cached_property
import os
from typing import List

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
    def out(self) -> Float[Tensor, "game move cell"]:
        out = logits_to_board(self.model(self.tokens[:, :59]), "logits")
        return out[:, self.focus_moves].flatten(start_dim=-2)

    @cached_property
    def expected(self) -> Bool[Tensor, "game move cell"]:
        is_valid = move_sequence_to_state(tokens_to_board(self.tokens), 'valid')
        return is_valid[:, self.focus_moves].flatten(start_dim=-2)

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
        l = -self.out.log_softmax(dim=-1) * self.expected.float()
        return l.sum(dim=-1)

    @cached_property
    def loss_scaled(self) -> Float[Tensor, "game move"]:
        metric = self.out.log_softmax(dim=-1) * self.expected.float()
        metric = -metric.sum(dim=-1)
        return metric / self.nb_valid_moves

    @cached_property
    def high_logit(self) -> Float[Tensor, "game move cell"]:
        predicted = self.softmax(dim=-1) > 1 / (2 * self.nb_valid_moves[..., None])
        return (predicted == self.expected).float()

    @cached_property
    def scaled_probabilities(self):
        metric = self.out.softmax(dim=-1)
        metric *= self.nb_valid_moves[..., None]
        metric[~self.expected] = 1 - metric[~self.expected]
        return metric

    def reduce(self, metric: Float[Tensor, "game move *cell"], mode: str = 'per_move') -> None:
        if metric.ndim != 3:
            assert mode == 'per_move', f"Can only reduce per_move for 3-dim tensors, got {metric.shape}"
            return metric.mean(0)

        # Reduce the metric to the plot we want
        if mode == 'per_cell':  # -> 8x8
            metric = metric.mean(dim=(0, 1)).reshape(8, 8)
        elif mode == "per_move":  # -> moves
            metric = metric.mean(dim=(0, 2))
        elif mode == "board-sum":  # moves
            metric = metric.sum(2).mean(0)
        elif mode == "board-prod":  # moves
            metric = metric.prod(2).mean(0)
        else:
            raise ValueError(f"Unknown per_move: {mode}")

        return metric

    # Different plots

    def plot_loss_per_move(self, rescaled: bool = False, name: str = "OthelloGPT"):
        title = 'Loss scaled by nb of valid moves' if rescaled else 'Loss'
        self.plot_by_move(
            f"{title} of {name}",
            [title],
            self.reduce(self.loss_scaled if rescaled else self.loss, 'per_move'),
            yaxis_title="Loss",
        )

    def plot_by_move(self,
                   title: str,
                   labels: List[str],
                     *lines: Float[Tensor, "x"],
                   yaxis_title: str = "Accuracy",
                   ):
        assert len(labels) == len(lines), f"Got {len(labels)} labels and {len(lines)} lines"

        fig = go.Figure()
        for line, label in zip(lines, labels):
            fig.add_trace(go.Scatter(
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



__all__ = [
    name for name, value in globals().items()
    if hasattr(value, "__module__") and value.__module__ == __name__ and name.startswith("plot_")
]
