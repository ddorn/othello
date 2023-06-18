from __future__ import annotations

import itertools
from typing import Callable, Iterable, List, Dict, Union, Optional, Tuple, Literal, Any

import einops
import numpy as np
import plotly.express as px
import torch
from jaxtyping import Int
from torch import Tensor
from transformer_lens import HookedTransformer, utils

from plotly_utils import imshow, line
from utils import logits_to_board, TOKEN_NAMES

try:
    from typing import Self
except ImportError:
    Self = "Kuit"

Labels = Dict[str, List[str]]


class MagicTensor:
    """
    MagicTensor is a tensor that wraps a tensor with named dimensions,
    manipulating the dimensions automatically.
    """

    value: Tensor
    shape: List[str]
    labels: Labels

    def __init__(self, value: Tensor, shape: List[str], labels: Optional[Labels] = None) -> None:
        self.value = value
        self.shape = shape
        self.labels = labels or {}

        assert self.value.ndim == len(
            self.shape
        ), f"Shape {self.shape} does not match value shape {self.value.shape}"

    def __repr__(self) -> str:
        shape = ", ".join(f"{dim}={size}" for dim, size in zip(self.shape, self.value.shape))
        return f"{self.__class__.__name__}({shape})"

    def print(self, title: str = "") -> Self:
        """Print the circuit. Useful for debugging the shapes."""
        if title:
            print(f"{title}:", self)
        else:
            print(self)
        return self

    def edited(
        self, value: Tensor, shape: Optional[List[str]] = None, labels: Optional[Labels] = None
    ) -> Self:
        """Return a new circuit with the given value and shape. If not given, use the current shape."""
        if shape is None:
            shape = self.shape
        if labels is None:
            labels = self.labels
        return self.__class__(value, shape, labels)

    @property
    def einops_pattern(self) -> str:
        return " ".join(self.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    # Functions to manipulate the dimensions order/names

    def rename_(self, old: str, new: str) -> None:
        assert old in self.shape, f"Old dimension {old} not in {self.shape}"
        assert new not in self.shape, f"New dimension {new} already in {self.shape}"

        self.shape = [new if d == old else d for d in self.shape]
        if old in self.labels:
            self.labels[new] = self.labels.pop(old)

    def rearrange(self, new_shape: List[str]) -> Self:
        value = einops.rearrange(self.value, f"{self.einops_pattern} -> {' '.join(new_shape)}")
        return self.edited(value, new_shape)

    def by(self, *dims: str, last: bool = False) -> Self:
        """Rearrange the dimensions so that the given ones are first.

        Args:
            last (bool, optional): If True, put the given dimensions last instead of first. The last dim passed will be the last of the tensor.
        """

        assert all(dim in self.shape for dim in dims), f"Dims {dims} not a subset of {self}"

        other = [d for d in self.shape if d not in dims]
        if last:
            new_shape = other + list(dims)
        else:
            new_shape = list(dims) + other

        return self.rearrange(new_shape)

    def new_dim(self, dim: str) -> Self:
        """Insert a new dimension of size 1."""
        assert dim not in self.shape, f"Cannot add {dim} to {self}, already present"
        return self.edited(self.value.unsqueeze(-1), self.shape + [dim])

    def _preferred_dim(self, shape: Iterable[str], hint: Optional[str] = None) -> str:
        """Return the preferred dimension to multiply with.

        This always return a dimension present in `self.shape` and `shape`.
        If hint is given and in both shapes, return it.
        If the two shapes have exactly one dimension in common, return it.
        Otherwise, raise ValueError.
        """

        intersection = set(self.shape) & set(shape)
        if hint is not None and hint in intersection:
            return hint
        elif len(intersection) == 1:
            return intersection.pop()
        elif hint is not None:
            raise ValueError(
                f"Could not find a preferred dim to multiply between {self} and {shape}. Hint '{hint}' not in intersection {intersection}"
            )
        else:
            raise ValueError(
                f"Could not find a preferred dim to multiply between {self} and {shape}"
            )

    def _dim_name_to_index(self, dim: Optional[str] = None) -> int:
        """Convert a dimension name to its index in the shape.

        If dim is None, the dimension is implicit. It is only possible for 1D tensors.
        Useful for parsing input dimension names.
        """

        if dim is None and self.ndim == 1:
            return 0
        elif dim is None:
            raise ValueError(f"Dimension is implicit but tensor is not 1D: {self}")

        try:
            return self.shape.index(dim)
        except ValueError:
            raise ValueError(f"Dimension {dim} not in {self.shape}")

    def _get_shape_and_labels_without(
        self, dim: Optional[Union[str, int]] = None, keepdim: bool = False
    ) -> Tuple[List[str], Labels]:
        """Return the shape without the given dimension and associated labels.

        If keepdim is True, return the current shape and the labels without the given dimension.
        This assumes that keepdim is True when changing the value to something of size one.
        """
        if keepdim:
            return self.shape, self.labels

        if isinstance(dim, str):
            dim_index = self._dim_name_to_index(dim)
        else:
            dim_index = dim
            dim = self.shape[dim_index]

        if dim in self.labels:
            labels = {d: v for d, v in self.labels.items() if d != dim}
        else:
            labels = self.labels

        return self.shape[:dim_index] + self.shape[dim_index + 1 :], labels

    def __getitem__(self, index: Union[int, slice, Tuple[str, Union[int, slice]]]) -> Self:
        """Index the tensor along one dimension.

        Valid index are:
            - (str, int): index the given dimension with the given index
            - (str, slice): slice the given dimension
        If the tensor is 1D, the dimension can be implicit, and the index can be an int or a slice.
        """
        if isinstance(index, tuple):
            dim, index = index
        elif isinstance(index, (int, slice)):
            dim = None
        else:
            raise ValueError(f"Invalid index type: {type(index)}")

        dim_index = self._dim_name_to_index(dim)

        indexing = [slice(None)] * self.ndim
        indexing[dim_index] = index

        if isinstance(index, int):
            new_shape, new_labels = self._get_shape_and_labels_without(dim)
        elif isinstance(index, slice):
            new_shape = self.shape
            new_labels = {d: l if d != dim else l[index] for d, l in self.labels.items()}
        else:
            raise ValueError(f"Invalid index type: {type(index)}")

        return self.edited(self.value[indexing], new_shape, new_labels)

    def with_labels(self, dim: str, labels: Labels) -> Self:
        """Set the labels of the given dimension."""
        assert dim in self.shape, f"Dimension {dim} not in {self.shape}"
        assert (
            len(labels) == self.shape[self._dim_name_to_index(dim)]
        ), f"Labels {labels} do not match dimension {dim} of {self.shape}"
        return self.edited(self.value, self.shape, {**self.labels, dim: labels})

    def tokens_to_board(self, dim: Optional[str] = None, fill: float = 0) -> Self:
        """Convert the given dimension to two dimensions, row and col.

        If no dimension is given, use only one that starts with "vocab".
        """
        assert "row" not in self.shape, f"Cannot convert to board, already has a row dimension"
        assert "col" not in self.shape, f"Cannot convert to board, already has a col dimension"

        if dim is None:
            vocab_dims = [d for d in self.shape if d.startswith("vocab")]
            assert len(vocab_dims) == 1, f"Cannot infer which dimension to use among {vocab_dims}"
            dim = vocab_dims[0]

        # move the dimension to the end
        out = self.by(dim, last=True)

        value = logits_to_board(out.value, "logits", fill_value=fill)
        new_shape = out.shape[:-1] + ["row", "col"]
        new_labels = {d: l for d, l in out.labels.items() if d != dim}
        new_labels["row"] = list("ABCDEFGH")
        new_labels["col"] = list("12345678")
        return out.edited(value, new_shape, new_labels)

    # Functions of arity 1

    def softmax(self, dim: Optional[str] = None) -> Self:
        """Apply softmax along the given dimension."""
        dim_index = self._dim_name_to_index(dim)
        return self.edited(self.value.softmax(dim=dim_index))

    def norm(self, dim: Optional[str] = None, keepdim: bool = False) -> Self:
        """Compute the norm along the given dimension."""
        dim_index = self._dim_name_to_index(dim)
        new_shape, new_labels = self._get_shape_and_labels_without(dim, keepdim)
        value = torch.linalg.vector_norm(self.value, dim=dim_index, keepdim=keepdim)
        return self.edited(value, new_shape, new_labels)

    def normalise(self, dim: Optional[str] = None) -> Self:
        """Normalise the tensor along the given dimension."""
        dim_index = self._dim_name_to_index(dim)
        return self.edited(self.value / self.value.norm(dim=dim_index, keepdim=True))

    def remove_diag(self) -> Self:
        """Set the diagonal of the last two dimensions to 0."""
        assert self.ndim >= 2, f"Cannot remove diag of {self}, not enough dimensions"
        assert (
            self.value.shape[-1] == self.value.shape[-2]
        ), f"Cannot remove diag of {self}, last two dimensions are not equal"
        return self.edited(self.value - self.value.diag().diag())

    def flatten(
        self,
        *dims: str,
        new_name: Optional[str] = None,
        generate_labels: bool = True,
        sep: str = "-",
    ) -> Self:
        """Flatten the two dimensions into one"""
        for dim in dims:
            assert dim in self.shape, f"Dimension {dim} not in {self}"
        assert len(set(dims)) == len(dims), f"Repeated dimensions in {dims}"

        if not dims:
            dims = self.shape
        elif len(dims) == 1:
            raise ValueError(f"Cannot flatten only one dimension: {dims}")

        # Put the dimensions at the end
        out = self.by(*dims, last=True)

        if new_name is None:
            new_name = f"flat_{'_'.join(dims)}"

        new_shape = out.shape[: -len(dims)] + [new_name]
        value = self.value.flatten(len(new_shape) - 1)

        new_labels = {k: v for k, v in out.labels.items() if k not in dims}
        if generate_labels and all(dim in out.labels for dim in dims):
            new_labels[new_name] = [
                sep.join(labels) for labels in itertools.product(*(out.labels[dim] for dim in dims))
            ]

        return out.edited(value, new_shape, new_labels)

    def sum(self, dim: Optional[str] = None, keepdim: bool = False) -> Self:
        """Sum along the given dimension."""
        dim_index = self._dim_name_to_index(dim)
        new_shape, new_labels = self._get_shape_and_labels_without(dim, keepdim)
        return self.edited(self.value.sum(dim=dim_index, keepdim=keepdim), new_shape, new_labels)

    def apply(self, func: Callable[[Tensor], Tensor]) -> Self:
        """Apply a function to the tensor."""
        out = func(self.value)
        assert (
            out.shape == self.value.shape
        ), f"Function changed the shape of {self} to {out.shape}. Shape must be kept."
        return self.edited(out)

    def __call__(self, func: Callable[[Tensor], Tensor]) -> Self:
        """Apply a function to the tensor."""
        return self.apply(func)

    def branch(self, func: Callable[[Self], Any]) -> Self:
        """Calls the function on the current MagicTensor and returns the original (!) MagicTensor.
        Useful when you don't want to store the result before doing two similar plots.
        """
        func(self)
        return self

    # Functions of arity 2

    def mul(
        self,
        other: MagicTensor,
        preferred_dim: Optional[str] = None,
    ) -> Self:
        """Multiply the tensor with another one, automatically finding the dimension to multiply.

        Args:
            other (MagicTensor): The other tensor to multiply with.
            preferred_dim (Optional[str], optional): If given, use this dimension to multiply (according to Circuit._preferred_dim).

        Returns:
            Self: self * other + bias
        """

        if not self.shape:  # 0D tensor
            return self.edited(self.value * other.value, other.shape, other.labels)

        # Find the dimension on which to multiply
        possible_multiplied_dims = set(self.shape) & set(other.shape)
        dim = self._preferred_dim(possible_multiplied_dims, preferred_dim)

        # Put the unmatched dimensions of the second tensor at the end
        new_shape = [d for d in self.shape if d != dim] + [
            d for d in other.shape if d not in possible_multiplied_dims
        ]
        pattern_target = " ".join(new_shape)

        # Keep the labels of the first tensor, and add the unmatched labels of the second tensor
        new_labels = {k: v for k, v in self.labels.items() if k != dim}
        new_labels.update(
            {k: v for k, v in other.labels.items() if k not in possible_multiplied_dims}
        )

        value = einops.einsum(
            self.value,
            other.value,
            f"{self.einops_pattern}, {other.einops_pattern} -> {pattern_target}",
        )

        return self.edited(value, new_shape, new_labels)

    def add(self, other: MagicTensor) -> Self:
        """Add the value of the other circuit to this one.

        One of the shape must be a subset of the other. The addition broadcast on the other dims.
        See also: Circuit.new_dim to extend a shape.
        """

        # We keep the one with the largest number of dimensions
        if self.ndim < other.ndim:
            small = self
            large = other
        else:
            small = other
            large = self

        assert set(small.shape) <= set(large.shape), f"{small} dims must be a subset of {large}"

        small_new_shape = [d if d in small.shape else "1" for d in large.shape]
        small_value = einops.rearrange(
            small.value, f"{small.einops_pattern} -> {' '.join(small_new_shape)}"
        )

        # Merge the labels after broadcasting: keep the labels of the non-1 dims
        new_labels = {}
        for d, name in enumerate(large.shape):
            if name in self.labels and self.value.shape[d] > 1:
                new_labels[name] = self.labels[name]
            elif name in other.labels and other.value.shape[d] > 1:
                new_labels[name] = other.labels[name]

        return self.edited(large.value + small_value, large.shape, labels=new_labels)

    def histogram(self, **kwargs) -> Self:
        return self.plot("histogram", **kwargs)

    def line(self, **kwargs) -> Self:
        return self.plot("line", **kwargs)

    def plot(self, plot_type: Literal["imshow", "line", "histogram"] = "imshow", **kwargs) -> Self:
        """Plot the tensor!

        TODO: Document how the plot is made.
        """

        # Change the default for 1D tensors to line
        if self.value.ndim == 1 and plot_type == "imshow":
            plot_type = "line"

        dims_to_handle = list(range(self.ndim))

        # The dimension of the type of graph
        dim_plot = 2 if plot_type == "imshow" else 1

        # If we have more than one plot, we use facets
        if len(dims_to_handle) > dim_plot:
            # Default: the dimension before the plot dimension(s)
            facet_dim = kwargs.setdefault("facet_col", dims_to_handle[-dim_plot - 1])
            # If the facet dimension is a name, convert it to an index
            if isinstance(facet_dim, str):
                facet_dim = self._dim_name_to_index(facet_dim)
                kwargs["facet_col"] = facet_dim
            # That's one dimension we took care of!
            dims_to_handle.remove(facet_dim)

            facet_name = self.shape[facet_dim]
            nb_plots = self.value.shape[facet_dim]
            if nb_plots <= 4:
                wrap = nb_plots
            elif nb_plots % 3 == 0:
                wrap = 3
            elif nb_plots % 4 == 0:
                wrap = 4
            else:
                wrap = 3
            wrap = kwargs.setdefault("facet_col_wrap", wrap)
            kwargs.setdefault("height", 500 * np.ceil(nb_plots / wrap))
            kwargs.setdefault("width", 500 * wrap)

            # Set the facet labels
            labels = self.get_label(facet_name, nb_plots)
            if labels is not None:
                kwargs.setdefault("facet_labels", labels)
            else:
                # For facet, we always want labels, otherwise ugly ones are generated
                kwargs.setdefault(
                    "facet_labels", [f"{facet_name.capitalize()} {i}" for i in range(nb_plots)]
                )
        else:
            kwargs.setdefault("height", 500)
            kwargs.setdefault("width", 500)

        # We still have too many dimensions to plot, we use animation frames
        if len(dims_to_handle) > dim_plot:
            dim_anim = kwargs.setdefault("animation_frame", dims_to_handle[-dim_plot - 1])
            # If the animation dimension is a name, convert it to an index
            if isinstance(dim_anim, str):
                dim_anim = self._dim_name_to_index(dim_anim)
                kwargs["animation_frame"] = dim_anim
            # That's one dimension we took care of!
            dims_to_handle.remove(dim_anim)
            # No labels for animation frames :s

        # If we still have dimensions to plot, game is over :/
        if len(dims_to_handle) > dim_plot:
            raise ValueError(f"Cannot plot {self}: {self.ndim} is too many dimensions 😅")

        kwargs.setdefault("title", str(self))

        if plot_type == "histogram":
            px.histogram(x=utils.to_numpy(self.value), **kwargs).show()
            return self

        # Set the x labels and title
        x_dim = dims_to_handle.pop()
        x_name = self.shape[x_dim]
        kwargs.setdefault("xaxis_title", x_name)
        labels = self.get_label(x_name, self.value.shape[x_dim])
        if labels is not None:
            kwargs.setdefault("x", labels)

        if plot_type == "imshow":
            # Set the y labels and title
            y_dim = dims_to_handle.pop()
            y_name = self.shape[y_dim]
            kwargs.setdefault("yaxis_title", y_name)
            labels = self.get_label(y_name, self.value.shape[y_dim])
            if labels is not None:
                kwargs.setdefault("y", labels)

            try:
                imshow(self.value, **kwargs)
            except Exception:
                print("Error while plotting", self)
                raise

        elif plot_type == "line":
            line(self.value, **kwargs)
        else:
            raise ValueError(f"Unknown type: {plot_type}")
        return self

    def get_label(self, dim: str, size: int) -> Optional[List[str]]:
        """Return the label of the given dimension, if it exists. Can be overriden by subclasses to generate labels on the fly."""
        if dim in self.labels:
            labels = self.labels[dim]
            assert len(labels) == size, f"Dimension {dim} has size {size} but {len(labels)} labels!"
            return labels
        return None


class Kuit(MagicTensor):
    """
    Circuit is a class useful to compute sub-parts of a transformer model.

    It is a wrapper around a tensor, with named dimensions,
    and all method return a new circuit with the modification applied,
    usually deducing the dimensions along which to multiply tensors.
    """

    model: HookedTransformer

    BATCH_NAMES = {"batch", "game"}

    def __init__(
        self,
        model: HookedTransformer,
        value: Optional[Tensor] = None,
        shape: Optional[List[str]] = None,
        labels: Optional[Labels] = None,
    ) -> None:
        self.model = model
        if value is None:
            value = torch.tensor(1.0, device=model.cfg.device)
        if shape is None:
            assert value.ndim == 0, f"Shape must be given for non-scalar tensors: {value}"
            shape = []
        super().__init__(value, shape, labels)

    def edited(
        self, value: Tensor, shape: Optional[List[str]] = None, labels: Optional[Labels] = None
    ) -> Self:
        """Return a new circuit with the given value and shape. If not given, use the current shape."""
        # We need to re-implement this method because we need to pass the model to the new circuit
        if shape is None:
            shape = self.shape
        if labels is None:
            labels = self.labels
        return self.__class__(self.model, value, shape, labels)

    # Leafs of the computation graph (input and output)

    def embedding(self, normalize: bool = False) -> Self:
        """Multiply self with the embedding matrix.

        New dimensions: vocab, dmodel. Possibly "vocab_1", "vocab_2" if multiplying along dmodel and self has a vocab dimension.
        """
        we = self.model.W_E
        if normalize:
            we = we / we.norm(dim=1, keepdim=True)

        if "vocab" in self.shape and "dmodel" in self.shape:
            # Prefer multiply of dmodel than vocab
            self.rename_("vocab", "vocab_1")
            vocab_dim = "vocab_2"
        else:
            vocab_dim = "vocab"
        return self.mul(MagicTensor(we, [vocab_dim, "dmodel"], {vocab_dim: TOKEN_NAMES}))

    def pos_embedding(self, normalize: bool = False, max_pos: Optional[int] = None) -> Self:
        """Multiply self with the positional embedding matrix.

        Args:
            normalize (bool, optional): If True, normalize the positional embedding. Useful to simulate a LayerNorm.
            max_pos (Optional[int], optional): If given, only use the first max_pos positions.

        New dimensions: pos, dmodel. Possibly "pos_1", "pos_2" if multiplying along dmodel and self has a pos dimension.
        """
        pos = self.model.W_pos
        if max_pos is not None:
            pos = pos[:max_pos]
        if normalize:
            pos = pos / pos.norm(dim=1, keepdim=True)

        if "pos" in self.shape:
            self.rename_("pos", "pos_1")
            return self.mul(MagicTensor(pos, ["pos_2", "dmodel"]))
        else:
            return self.mul(MagicTensor(pos, ["pos", "dmodel"]))

    def unembed(self, bias: bool = False) -> Self:
        """Multiply self with the unembedding matrix.

        New dimensions: dmodel, vocab_out.
        Renames "vocab" to "vocab_in" if present and multiplying along dmodel.
        """

        # Here, we assume that we want to multiply along dmodel.
        if "vocab" in self.shape and "dmodel" in self.shape:
            # Multiply along dmodel
            self.rename_("vocab", "vocab_in")
            vocab_dim = "vocab_out"
        else:
            vocab_dim = "vocab"
        shape = ["dmodel", vocab_dim]

        out = self.mul(MagicTensor(self.model.W_U, shape, {vocab_dim: TOKEN_NAMES}))
        if bias:
            out = out.add(MagicTensor(self.model.b_U, [vocab_dim]))
        return out

    def unembed_bias(self) -> Self:
        """Add the unembedding bias."""
        if self.ndim == 0:
            # We override the value.
            self.value[...] = 0
        return self.add(MagicTensor(self.model.b_U, ["vocab_out"], {"vocab_out": TOKEN_NAMES}))

    def embed(
        self,
        tokens: Int[Tensor, "*game token"],
        positional: bool = True,
    ) -> Self:
        """Embed the given tokens."""

        assert tokens.ndim < 4, f"Cannot embed {tokens.shape}, too many dimensions"
        tokens = tokens.to(self.model.cfg.device)
        embedded = self.model.W_E[tokens]

        if positional:
            print(embedded.shape, self.model.W_pos.shape)
            embedded += self.model.W_pos[: embedded.shape[-2]]

        shape = ["game", "pos", "dmodel"][-len(embedded.shape) :]
        return self.mul(MagicTensor(embedded, shape))

    # Specific circuits

    def ov(self, layer: int, head: Optional[int] = None) -> Self:
        """Pass through the OV circuit.

        New dimensions: "head" if head is None
        """

        if head is None:
            index = (layer,)
            dims_v = ["head", "dmodel", "dhead"]
            dims_o = ["head", "dhead", "dmodel"]
        else:
            index = (layer, head)
            dims_v = ["dmodel", "dhead"]
            dims_o = ["dhead", "dmodel"]

        out = self.mul(MagicTensor(self.model.W_V[index], dims_v))
        out = out.mul(MagicTensor(self.model.W_O[index], dims_o), preferred_dim="dhead")
        return out.add(MagicTensor(self.model.b_O[layer], ["dmodel"]))

    def qk(
        self, layer: Optional[int], head: Optional[int] = None, softmax: bool = False, *, key: Self
    ) -> Self:
        """Compute the query-key matrix.

        Args:
            layer (Optional[int]): If not None, only compute for this layer.
            head (Optional[int], optional): If not None, only compute for this head.
            softmax (bool, optional): If True and both key and query have a "pos" dimension, apply the softmax after the mask.
            key (Circuit): The key of the QK circuit (the query is self).

        New dimensions: Any repeated dimensions in both the key and query are suffixed with "_q" and "_k" respectively.
        If both have a "pos" dimension, apply a triangular mask to simulate the attention mask.
        """
        assert self.model is key.model, "Both circuits must have the same model"

        dims = []
        index = ()

        if layer is None:
            dims += ["layer"]
        else:
            index = (layer,)

        if head is None:
            dims += ["head"]
        else:
            index += (head,)

        dims += ["dmodel", "dhead"]
        dims_bias = dims[:]
        dims_bias.remove("dmodel")

        # Apply the query
        query = self.mul(MagicTensor(self.model.W_Q[index], dims), preferred_dim="dmodel")
        query = query.add(MagicTensor(self.model.b_Q[index], dims_bias))

        # Apply the key
        key = key.mul(MagicTensor(self.model.W_K[index], dims), preferred_dim="dmodel")
        key = key.add(MagicTensor(self.model.b_K[index], dims_bias))

        # Combine the two
        # Find the intersection of the two shapes
        possible_multiplied_dims = (
            set(query.shape) & set(key.shape) - {"dhead", "head", "layer"} - self.BATCH_NAMES
        )
        # We always want to match the head dimension, and there might be another dimension
        # that is shared between the two, which we don't want to match
        for d in possible_multiplied_dims:
            query.rename_(d, d + "_q")
            key.rename_(d, d + "_k")

        out = query.mul(key, preferred_dim="dhead")

        # If we have two "pos" dimensions, use the mask to simulate the triangular matrix
        tokens_dims = ["pos_q", "pos_k"]
        if set(tokens_dims) <= set(out.shape):
            # Reshape to put pos_q and pos_k at the end
            out = out.by(*tokens_dims, last=True)

            size = out.value.shape[-1]
            mask = torch.triu(torch.ones(size, size), diagonal=1).to(out.value.device, bool)
            out.value.masked_fill_(mask, float("-inf"))

            if softmax:
                out = out.softmax(dim="pos_k")

        return out

    # Other functions

    def softmax(self, dim: Optional[str] = None) -> Self:
        """Apply softmax along the given dimension. If dim is None, tries to use the "pos_k" dimension."""
        if dim is None and "pos_k" in self.shape:
            dim = "pos_k"
        return super().softmax(dim)


if __name__ == "__main__":

    def main():
        from utils import get_othello_gpt

        cfg, model = get_othello_gpt("cpu")

        c = Kuit(model)
        c.embedding().plot()

        c.unembed().by("dmodel").plot(height=800)

    main()
