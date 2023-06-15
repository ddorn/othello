# %%
from typing import Callable, Iterable, List, Dict, Union, Optional, Tuple, Literal, Self
import plotly.express as px
import torch
import einops
from torch import Tensor
from transformer_lens import HookedTransformer
from dataclasses import dataclass, field

from utils import logits_to_board, TOKEN_NAMES, CELL_TOKEN_NAMES
from plotly_utils import imshow, line


@dataclass
class Kuit:
    """
    Circuit is a class useful to compute sub-parts of a transformer model.

    It is a wrapper around a tensor, with named dimensions,
    and all method return a new circuit with the modification applied,
    usually deducing the dimensions along which to multiply tensors.
    """

    model: HookedTransformer
    value: Tensor = field(default_factory=lambda: torch.tensor(1))
    _shape: List[str] = field(default_factory=lambda: [])

    LABELS = {}

    def __repr__(self) -> str:
        shape = ', '.join(f'{dim}={size}' for dim, size in zip(self._shape, self.value.shape))
        return f"{self.__class__.__name__}({shape})"

    @property
    def shape(self) -> Dict[str, int]:
        return {d: s for d, s in zip(self._shape, self.value.shape)}

    def print(self, title: str = "") -> Self:
        """Print the circuit. Useful for debugging the shapes."""
        if title:
            print(f"{title}:", self)
        else:
            print(self)
        return self

    def edited(self, value: Optional[Tensor], shape: Optional[List[str]] = None) -> Self:
        """Return a new circuit with the given value and shape. If not given, use the current shape."""
        if shape is None:
            shape = self._shape
        assert len(shape) == len(
            value.shape), f"Shape {shape} does not match value shape {value.shape}"
        return self.__class__(self.model, value, shape)

    # Functions to manipulate the dimensions order/names

    def rearange(self, new_shape: List[str]):
        value = einops.rearrange(self.value, f"{' '.join(self._shape)} -> {' '.join(new_shape)}")
        return self.edited(value, new_shape)

    def by(self, *dims: str, last: bool = False) -> Self:
        """Rearange the dimensions so that the given ones are first.

        Args:
            last (bool, optional): If True, put the given dimensions last instead of first. The last dim passed will be the last of the tensor.
        """

        assert all(dim in self._shape for dim in dims), f"Dims {dims} not in {self._shape}"

        other = [d for d in self._shape if d not in dims]
        if last:
            new_shape = other + list(dims)
        else:
            new_shape = list(dims) + other

        return self.rearange(new_shape)

    def rename_(self, old: str, new: str) -> None:
        """Rename a dimension inplace"""
        assert old in self._shape, f"Old dimension {old} not in {self._shape}"
        assert new not in self._shape, f"New dimension {new} already in {self._shape}"

        self._shape = [new if d == old else d for d in self._shape]

    def new_dim(self, dim: str) -> Self:
        """Insert a new dimension of size 1."""
        assert dim not in self._shape, f"Cannot add {dim} to {self}, already present"
        return self.edited(self.value.unsqueeze(-1), self._shape + [dim])

    def _prefered_dim(self, shape: Iterable[str], hint: Optional[str] = None) -> str:
        """Return the prefered dimension to multiply with.

        This always return a dimension present in `self._shape` and `shape`.
        If hint is given and in both shapes, return it.
        If the two shapes have exactly one dimension in common, return it.
        Otherwise, raise ValueError.
        """

        intersection = set(self._shape) & set(shape)
        if hint is not None and hint in intersection:
            return hint
        elif len(intersection) == 1:
            return next(iter(intersection))
        elif hint is not None:
            raise ValueError(
                f"Could not find a prefered dim to multiply between {self._shape} and {shape}. Hint '{hint}' not in intersection {intersection}"
            )
        else:
            raise ValueError(
                f"Could not find a prefered dim to multiply between {self._shape} and {shape}")

    def _dim_name_to_index(self, dim: Optional[str] = None) -> int:
        """Convert a dimension name to its index in the shape.

        If dim is None, the dimension is implicit. It is only possible for 1D tensors.
        Useful for parsing input dimension names.
        """

        if dim is None:
            assert len(self._shape
                       ) == 1, f"Implicit dimention is possible only for 1D tensor. Got: {self}."
            return 0
        else:
            assert dim in self._shape, f"Dimension {dim} not in {self._shape}"
            return self._shape.index(dim)

    def _get_shape_without(self,
                           dim: Optional[Union[str, int]] = None,
                           keepdim: bool = False) -> List[str]:
        """Return the shape without the given dimension.

        If keepdim is True, return the current shape.
        """
        if keepdim:
            return self._shape

        if isinstance(dim, str):
            dim_index = self._dim_name_to_index(dim)

        return self._shape[:dim_index] + self._shape[dim_index + 1:]

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

        indexing = [slice(None)] * len(self._shape)
        indexing[dim_index] = index

        if isinstance(index, int):
            new_shape = self._get_shape_without(dim)
        elif isinstance(index, slice):
            new_shape = self._shape
        else:
            raise ValueError(f"Invalid index type: {type(index)}")

        return self.edited(self.value[indexing], new_shape)

    # Leafs of the computation graph (input and output)

    def embedding(self, normalize: bool = False) -> Self:
        """Multiply self with the embedding matrix.

        New dimensions: vocab, dmodel. Possibly "vocab_1", "vocab_2" if multiplying along dmodel and self has a vocab dimension.
        """
        we = self.model.W_E
        if normalize:
            we = we / we.norm(dim=1, keepdim=True)

        if 'vocab' in self._shape and 'dmodel' in self._shape:  # Prefer multiply of dmodel than vocab
            self.rename_("vocab", "vocab_1")
            return self._mul(we, ["vocab_2", "dmodel"])
        else:
            return self._mul(we, ["vocab", "dmodel"])

    def pos_embedding(self, normalize: bool = False) -> Self:
        """Multiply self with the positional embedding matrix.

        New dimensions: pos, dmodel. Possibly "pos_1", "pos_2" if multiplying along dmodel and self has a pos dimension.
        """
        pos = self.model.W_pos
        if normalize:
            pos = pos / pos.norm(dim=1, keepdim=True)

        if 'pos' in self._shape:
            self.rename_("pos", "pos_1")
            return self._mul(pos, ["pos_2", "dmodel"])
        else:
            return self._mul(pos, ["pos", "dmodel"])

    def unembed(self, bias: bool = False) -> Self:
        """Multiply self with the unembedding matrix.

        New dimensions: dmodel, vocab_out.
        Renames "vocab" to "vocab_in" if present and multiplying along dmodel.
        """

        # Here, we assume that we want to multiply along dmodel.
        if 'vocab' in self._shape and 'dmodel' in self._shape:
            # Multiply along dmodel
            self.rename_("vocab", "vocab_in")
            shape = ['dmodel', 'vocab_out']
        else:
            shape = ['dmodel', 'vocab']

        return self._mul(self.model.W_U, shape, self.model.b_U if bias else None)

    def unembed_bias(self) -> Self:
        """Add the unembedding bias."""
        if not self._shape:
            return self.edited(self.model.b_U, ["vocab_out"])
        return self.add(Kuit(self.model, self.model.b_U, ["vocab_out"]))

    # Functions of arrity 1

    def ov(self, layer: int, head: Optional[int] = None) -> Self:
        """Pass through the OV circuit.

        New dimensions: "head" if head is None
        """

        if head is None:
            index = (layer, )
            dims_v = ["head", "dmodel", "dhead"]
            dims_o = ["head", "dhead", "dmodel"]
        else:
            index = (layer, head)
            dims_v = ["dmodel", "dhead"]
            dims_o = ["dhead", "dmodel"]

        out = self._mul(self.model.W_V[index], dims_v)
        out = out._mul(self.model.W_O[index], dims_o, self.model.b_O[index])
        return out.add(Kuit(self.model, self.model.b_O[index], ["dmodel"]))

    def softmax(self, dim: Optional[str] = None) -> Self:
        """Apply softmax along the given dimension."""
        dim_index = self._dim_name_to_index(dim)
        return self.edited(self.value.softmax(dim=dim_index))

    def norm(self, dim: Optional[str] = None, keepdim: bool = False) -> Self:
        """Compute the norm along the given dimension."""
        dim_index = self._dim_name_to_index(dim)
        new_shape = self._get_shape_without(dim, keepdim)
        value = torch.linalg.vector_norm(self.value, dim=dim_index, keepdim=keepdim)
        return self.edited(value, new_shape)

    def normalise(self, dim: Optional[str] = None) -> Self:
        """Normalise the tensor along the given dimension."""
        dim_index = self._dim_name_to_index(dim)
        return self.edited(self.value / self.value.norm(dim=dim_index, keepdim=True))

    def remove_diag(self) -> Self:
        """Set the diagonal of the last two dimensions to 0."""
        assert len(self._shape) >= 2, f"Cannot remove diag of {self}, not enough dimensions"
        assert self.value.shape[-1] == self.value.shape[
            -2], f"Cannot remove diag of {self}, last two dimensions are not equal"
        return self.edited(self.value - self.value.diag().diag())

    def flatten(self, *dims: str) -> Self:
        """Flatten the two dimensions into one"""
        for dim in dims:
            assert dim in self._shape, f"Dimension {dim} not in {self._shape}"
        assert len(set(dims)) == len(dims), f"Repeated dimensions in {dims}"

        if not dims:
            dims = self._shape
        elif len(dims) == 1:
            raise ValueError(f"Cannot flatten only one dimension: {dims}")

        # Put the dimensions at the end
        out = self.by(*dims, last=True)

        new_name = f"flat_{'_'.join(dims)}"
        new_shape = out._shape[:-len(dims)] + [new_name]
        value = self.value.flatten(len(new_shape) - 1)

        return out.edited(value, new_shape)

    def sum(self, dim: Optional[str] = None, keepdim: bool = False) -> Self:
        """Sum along the given dimension."""
        dim_index = self._dim_name_to_index(dim)
        new_shape = self._get_shape_without(dim, keepdim)
        return self.edited(self.value.sum(dim=dim_index, keepdim=keepdim), new_shape)

    def apply(self, func: Callable[[Tensor], Tensor]) -> Self:
        """Apply a function to the tensor."""
        out = func(self.value)
        assert out.shape == self.value.shape, f"Function changed the shape of {self} to {out.shape}. Shape must be kept."
        return self.edited(out)

    def __call__(self, func: Callable[[Tensor], Tensor]) -> Self:
        """Apply a function to the tensor."""
        return self.apply(func)

    # Functions of arriy 2

    def _mul(self,
             other: Tensor,
             other_shape: List[str],
             bias: Optional[Tensor] = None,
             prefered_dim: Optional[str] = None) -> Self:
        """Multiply the tensor with another one, automatically finding the dimension to multiply.

        Args:
            other (Tensor): The other tensor to multiply with.
            other_shape (List[str]): The shape of the other tensor.
            bias (Optional[Tensor], optional): Optianally add a bias. Bias is assumed to have the same shape as other, except for the dim that is multiplied.
            prefered_dim (Optional[str], optional): If given, use this dimension to multiply (according to Circuit._prefered_dim).

        Returns:
            Self: self * other + bias
        """

        # Check that the other shape is coherent
        assert len(other.shape) == len(
            other_shape), f"Shape {other_shape} does not match value shape {other.shape}"
        # Check that no dim is repeated in each tensor
        assert len(set(self._shape)) == len(self._shape), f"Repeated dimensions in {self._shape}"
        assert len(set(other_shape)) == len(other_shape), f"Repeated dimensions in {other_shape}"

        if not self._shape:  # 0D tensor
            return self.edited(self.value * other, other_shape)

        # Find the dimension on which to multiply
        possible_multiplied_dims = set(self._shape) & set(other_shape)
        dim = self._prefered_dim(possible_multiplied_dims, prefered_dim)

        pattern_1 = " ".join(self._shape)
        pattern_2 = " ".join(other_shape)

        shape = [d for d in self._shape if d != dim
                 ] + [d for d in other_shape if d not in possible_multiplied_dims]
        pattern_target = ' '.join(shape)

        value = einops.einsum(
            self.value,
            other,
            f"{pattern_1}, {pattern_2} -> {pattern_target}",
        )

        if bias is not None:
            bias_shape = [d for d in other_shape if d != dim]
            assert len(bias.shape) == len(
                bias_shape), f"Shape {bias_shape} does not match value shape {bias.shape}"
            target_shape = [d if d in bias_shape else "1" for d in shape]  # broadcast the bias
            bias = einops.rearrange(bias, f"{' '.join(bias_shape)} -> {' '.join(target_shape)}")
            value += bias

        return self.edited(value, shape)

    def qk(self,
           layer: Optional[int],
           head: Optional[int] = None,
           softmax: bool = False,
           *,
           key: Self) -> Self:
        """Compute the query-key matrix.

        Args:
            layer (Optional[int]): If not None, only compute for this layer.
            head (Optional[int], optional): If not None, only compute for this head.
            softmax (bool, optional): If True and both key and query have a "pos" dimension, apply the softmax after the mask.
            key (Circuit): The key of the QK circuit (the query is self).

        New dimensions: Any repeated dimensions in both the key and query are suffixed with "_q" and "_k" respectively.
        If both have a "pos" dimension, the apply a triangular mask to simulate the attention mask.
        """
        assert self.model is key.model, "Both circuits must have the same model"

        dims = []
        index = ()

        if layer is None:
            dims += ["layer"]
        else:
            index = (layer, )

        if head is None:
            dims += ["head"]
        else:
            index += (head, )

        dims += ["dmodel", "dhead"]

        # Apply the query
        query = self._mul(self.model.W_Q[index], dims, self.model.b_Q[index], prefered_dim="dmodel")

        # Apply the key
        key = key._mul(self.model.W_K[index], dims, self.model.b_K[index], prefered_dim="dmodel")

        # Combine the two
        # Find the intersection of the two shapes
        possible_multiplied_dims = set(query._shape) & set(key._shape) - {'dhead', 'head', 'layer'}
        # We always want to match the head dimension, and there might be an other dimension
        # that is shared between the two, which we dont want to match
        query._shape = [d + "_q" if d in possible_multiplied_dims else d for d in query._shape]
        key._shape = [d + "_k" if d in possible_multiplied_dims else d for d in key._shape]

        out = query._mul(key.value, key._shape, prefered_dim="dhead")

        # If we have two "pos" dimensions, use the mask to simulate the triangular matrix
        tokens_dims = ["pos_q", "pos_k"]
        if all(d in out._shape for d in tokens_dims):
            # Reshape to put pos_q and pos_k at the end
            out = out.by(*tokens_dims, last=True)

            size = out.shape['pos_q']
            mask = torch.triu(torch.ones(size, size), diagonal=1).to(out.value.device, bool)
            out.value.masked_fill_(mask, float("-inf"))

            if softmax:
                out = out.softmax(dim='pos_k')

        return out

    def add(self, other: Self) -> Self:
        """Add the value of the other circuit to this one.

        One of the shape must be a subset of the other. The addition broadcast on the other dims.
        See also: Circuit.new_dim to extend a shape.
        """
        assert self.model is other.model, "Both circuits must have the same model"

        # We keep the one with the largest number of dimensions
        if len(self._shape) < len(other._shape):
            return other.add(self)

        assert set(other.shape) <= set(self.shape), f"Cannot add {other.shape} to {self.shape}"

        other_new_shape = [d if d in other._shape else "1" for d in self._shape]
        other_value = einops.rearrange(other.value,
                                       f"{' '.join(other._shape)} -> {' '.join(other_new_shape)}")

        return self.edited(self.value + other_value)

    def tokens_to_board(self, dim: Optional[str] = None, fill: float = 0) -> Self:
        """Convert the given dimension to two dimensions, row and col.

        If no dimension is given, use only one that starts with "vocab".
        """
        assert "row" not in self._shape, f"Cannot convert to board, already has a row dimension"
        assert "col" not in self._shape, f"Cannot convert to board, already has a col dimension"

        vocab_dims = [d for d in self._shape if d.startswith("vocab")]
        if dim is None:
            assert len(vocab_dims) == 1, f"Cannot infer which dimension to use: {vocab_dims}"
            dim = vocab_dims[0]
        else:
            assert dim in vocab_dims, f"Dimension {dim} not a vocabulary dimension. Options: {vocab_dims}"

        # move the dimension to the end
        out = self.by(dim, last=True)

        value = logits_to_board(self.value, 'logits', fill_value=fill)
        return out.edited(value, self._shape[:-1] + ['row', 'col'])

    def histogram(self, **kwargs) -> Self:
        return self.plot('histogram', **kwargs)

    def line(self, **kwargs) -> Self:
        return self.plot('line', **kwargs)

    def plot(self, plot_type: Literal['imshow', 'line', 'histogram'] = 'imshow', **kwargs) -> Self:
        """Plot the tensor!

        TODO: Document how the plot is made.
        """

        dim_plot = 2 if plot_type == 'imshow' else 1

        if len(self._shape) > dim_plot:
            facet_dim = (-dim_plot - 1) % len(self._shape)
            facet_name = self._shape[facet_dim]
            nb_plots = self.value.shape[facet_dim]
            kwargs.setdefault('facet_col', facet_dim)
            if 'facet_col_wrap' in kwargs:
                wrap = kwargs['facet_col_wrap']
            elif nb_plots < 4:
                wrap = nb_plots
            elif nb_plots == 4:
                wrap = 2
            else:
                wrap = 3
            kwargs.setdefault('facet_col_wrap', wrap)
            kwargs.setdefault('height', 500 * (nb_plots // wrap))

            if facet_name.startswith('head'):
                kwargs.setdefault('facet_labels', [f"Head {i}" for i in range(nb_plots)])
            elif facet_name.startswith('layer'):
                kwargs.setdefault('facet_labels', [f"Layer {i}" for i in range(nb_plots)])
            else:
                print(f"Warning: Don't know how to label {facet_name}")

        if len(self._shape) == dim_plot + 2:
            kwargs.setdefault('animation_frame', 0)
        elif len(self._shape) > dim_plot + 2:
            raise ValueError(f"Cannot plot {self}: Too many dimensions.")

        kwargs.setdefault('title', str(self))

        if plot_type == 'histogram':
            px.histogram(self.value, **kwargs).show()
            return self

        x_name = self._shape[-1]
        kwargs.setdefault('xaxis_title', x_name)
        if x_name.startswith('vocab'):
            if self.shape[x_name] == len(TOKEN_NAMES):
                kwargs.setdefault('x', TOKEN_NAMES)
            elif self.shape[x_name] == len(CELL_TOKEN_NAMES):
                kwargs.setdefault('x', CELL_TOKEN_NAMES)

        if plot_type == 'imshow':
            y_name = self._shape[-2]
            kwargs.setdefault('yaxis_title', y_name)
            if y_name.startswith('vocab'):
                if self.shape[y_name] == len(TOKEN_NAMES):
                    kwargs.setdefault('y', TOKEN_NAMES)
                elif self.shape[y_name] == len(CELL_TOKEN_NAMES):
                    kwargs.setdefault('y', CELL_TOKEN_NAMES)

            try:
                imshow(self.value, **kwargs)
            except Exception as e:
                print("Error while plotting", self)
                raise

        elif plot_type == 'line':
            # kwargs.setdefault('xaxis', x_name)
            kwargs.setdefault('xaxis_title', x_name)
            line(self.value, **kwargs)
        else:
            raise ValueError(f"Unknown type: {plot_type}")
        return self


if __name__ == "__main__":
    from utils import get_othello_gpt

    cfg, model = get_othello_gpt('cpu')

    c = Kuit(model)
    c.embedding().plot()

    c.unembed().by('dmodel').plot(height=800)

# %%
