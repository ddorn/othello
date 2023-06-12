# %%
import os

import einops
import plotly.express as px
import torch as t
from IPython.display import display
from jaxtyping import Float
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import Tensor

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


__all__ = [
    name for name, value in globals().items()
    if hasattr(value, "__module__") and value.__module__ == __name__ and name.startswith("plot_")
]
