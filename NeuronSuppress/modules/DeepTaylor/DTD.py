import abc
from collections import OrderedDict
from typing import Any, Callable, Generic, Optional, Sequence, TypeVar, Union
import dataclasses
import numpy as np
import torch
from torch import nn
from torch.utils import hooks
from diffusers.models.unets import UNet2DConditionModel, UNet2DConditionOutput

# from lrp_relations import local_linear
RULE = Union["Rule", str]
NEURON = Union[int, slice]

REL = TypeVar("REL", bound="Relevance")
REL_FN = TypeVar("REL_FN", bound="RelevanceFn")
NETWORK_LAYER = Union[nn.Module, Sequence[nn.Module], UNet2DConditionOutput, UNet2DConditionModel]

class Rule:
    def __init__(self, name_or_rule: RULE):
        if isinstance(name_or_rule, str):
            self.name = name_or_rule
        else:
            assert isinstance(name_or_rule, Rule)
            self.name = name_or_rule.name

        self.is_layer_rule = self.name in ["pinv"]


def rule_name(rule: RULE) -> str:
    if isinstance(rule, str):
        return rule
    elif isinstance(rule, Rule):
        return rule.name
    else:
        raise ValueError(f"Unknown rule {rule}")
    

class GammaRule(Rule):
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma
        super().__init__("gamma")

class rules:
    pinv = Rule("pinv")
    z_plus = Rule("z+")
    w2 = Rule("w2")
    x = Rule("x")
    zB = Rule("zB")
    zero = Rule("0")

    @staticmethod   #  Independent of the instantiation process
    def gamma(gamma: float) -> GammaRule:
        return GammaRule(gamma=gamma)
    
def is_layer_rule(rule: RULE) -> bool:
    """Return whether the rule is a layer rule."""
    return Rule(rule).name == "pinv"

def calculate_root_for_layer(
        x: torch.Tensor,
        layer: nn.Module,
        rule: RULE = "pinv",
        relevance: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    
    weight, bias = None, None
    if isinstance(layer.state_dict(), OrderedDict):
        for name, param in layer.state_dict().items():
            if 'weight' in name:
                weight = layer.state_dict().get('weight')
            if 'bias' in name:
                bias = layer.state_dict().get('bias')
    if weight.dtype not in [torch.float32, torch.float64]:
        weight = weight.float()
    if bias is not None and bias.dtype not in [torch.float32, torch.float64]:
        bias = bias.float()

    if rule_name(rule) == "pinv":
        if relevance is not None:
            assert len(relevance) == 1
            return (torch.linalg.pinv(weight) @ relevance[0]).unsqueeze(0)
        else:
            if bias is not None:
                return (torch.linalg.pinv(weight) @ bias).unsqueeze(0)
            else:
                return torch.linalg.pinv(weight).unsqueeze(0)
    else:
        raise ValueError(f"Unknow rule {rule}")
    
def calculate_root_for_single_neuron(
        x: torch.Tensor,
        layer: nn.Module,
        j: int,
        rule: RULE = "z+",
        relevance: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Return the DTD root point.

    Args:
        x: Input tensor.
        layer: Layer to compute the root point.
        j: Index of which neuron to compute the root point.
        rule: Rule to compute the root point (supported: `z+`, `w2`, and
            `gamma`).

    Returns:
        A root point `r` with the property `layer(r)[j] == 0`.
    """
    w, b = None, None
    if isinstance(layer.state_dict(), OrderedDict):
        for name, param in layer.state_dict().items():
            if 'weight' in name:
                w = layer.state_dict().get('weight')
            if 'bias' in name:
                b = layer.state_dict().get('bias')

    if w is not None:
        w_j = w[j, :].unsqueeze(0)
    else:
        raise ValueError(f"{layer} weight is None!")
    if b is not None:
        b_j = b[j, :]
    else:
        raise ValueError(f"{layer} bias is None!")
    indicator_w_j_pos = (w_j >= 0).float()

    name = rule_name(rule)
    if name == "z+":
        v = x * indicator_w_j_pos
    elif name == "w2":
        v = w_j
    elif name == "x":
        v = x
    elif name == "zB":
        raise NotImplementedError()
    elif name == "0":
        return 0 * x
    elif name in ["gamma", "γ"]:
        assert isinstance(rule, GammaRule)
        v = x * (1 + rule.gamma * indicator_w_j_pos)
    else:
        raise ValueError()
    
    if isinstance(layer, nn.Linear) and w is not None and b is not None:
        assert torch.allclose(          
                layer(x)[:, j].unsqueeze(1), x @ w_j.t() + b_j, atol=1e-3
            )
    elif isinstance(layer, nn.Linear) and w is not None and b is None:
        assert torch.allclose(          
                layer(x)[:, j].unsqueeze(1), x @ w_j.t(), atol=1e-3
            )
    else:
        raise ValueError(f"Result is not equal to {layer} calculation result")
    
    if (v @ w_j.t()).abs().sum() <= 1e-5:
        return None
    if relevance is None:
        rel_output = (x @ w_j.t() + b_j)[:, 0]
    else:
        rel_output = relevance[:, j]
    
    t = rel_output / (v @ w_j.t()).sum(1)

    t[~t.isfinite()] = 0.0
    root_point = x - t.unsqueeze(1) * v
    return root_point

@dataclasses.dataclass(frozen=True)
class RootPoint:
    root: torch.Tensor
    input: torch.Tensor
    layer: NETWORK_LAYER
    root_finder: "RootFinder"
    explained_neuron: Union[int, slice]
    relevance: torch.Tensor

    def __repr__(self) -> str:
        return (
            f"RootPoint(root={self.root}, input={self.input}",
            f"root_finder={self.root_finder})"
        )


@dataclasses.dataclass(frozen=True)
class RootFinder(abc.ABC):
    @abc.abstractmethod
    def get_root_points_for_layer(
        self,
        for_input_to_layer: NETWORK_LAYER,
        input: torch.Tensor,
        relevance_fn: "RelevanceFn[REL]",
    ) -> list[RootPoint]:
        """Return the root points for the given layer.

        Args:
            for_input_to_layer: The layer to find the root points of.
            input: The input to the layer.
            relevance_fn: A function that maps the output of the layer to a
                relevance score.
        """

@dataclasses.dataclass(frozen=True)
class RecursiveRoot:
    root: torch.Tensor
    input: torch.Tensor
    layer: NETWORK_LAYER
    root_finder: RootFinder
    explained_neuron: Union[int, slice]
    relevance: torch.Tensor

        
@dataclasses.dataclass(frozen=True)
class LocalSegmentCacheKey:
    input_layer: int
    lower_rel_layer: int
    upper_rel_layer: int
    grap_hash: int
    hash_decimals: int

    @staticmethod
    def from_rel_fn(
        rel_fn: "RelevanceFn", graph_hash: int, hash_decimals: int
    ) -> "LocalSegmentCacheKey":
        
        module = rel_fn.module
        def get_index(layer: NETWORK_LAYER) -> int:
            return module.get_layer_index(module.resolve_marker(layer))

        return __class__(
            get_index(rel_fn.get_input_layer()),
            get_index(rel_fn.get_lower_rel_layer()),
            get_index(rel_fn.get_upper_rel_layer()),
            graph_hash,
            hash_decimals,
        )

@dataclasses.dataclass(frozen=True)
class SampleRoots(RootFinder):
    module: nn.Module

    use_cache: bool = True
    cache: dict[
        LocalSegmentCacheKey,
        list[RootPoint],
    ] = dataclasses.field(default_factory=dict, init=False, hash=False)
    cache_hits: int = dataclasses.field(default=0, init=False, hash=False)
    cache_misses: int = dataclasses.field(default=0, init=False, hash=False)

    use_candidate_cache: bool = False
    candidate_cache: dict[
        LocalSegmentCacheKey,
        torch.Tensor,
    ] = dataclasses.field(default_factory=dict, init=False, hash=False)
    cache_grad_hash_decimals: int = 5

    def __post_init__(self):
        pass

    def cache_clear(self) -> None:
        self.candidate_cache.clear()
        self.cache.clear()
    
    def get_cache_key(
        self,
        for_input_to_layer: NETWORK_LAYER,
        input: torch.Tensor,
        relevance_fn: "RelevanceFn",
    ) -> Optional[LocalSegmentCacheKey]:
        assert not isinstance(for_input_to_layer, (UNet2DConditionOutput, UNet2DConditionModel))

        if not self.use_cache:
            return None

        grad = self.model.compute_input_grad(input, for_input_to_layer)
        grad_query = torch.round(grad, decimals=self.cache_grad_hash_decimals)
        grad_bytes = grad_query.detach().cpu().numpy().tobytes()
        grad_hash = hash(grad_bytes)

        return LocalSegmentCacheKey.from_rel_fn(
            relevance_fn, grad_hash, self.cache_grad_hash_decimals
        )
    
    @abc.abstractmethod
    def sample_candidates(
        self, 
        module: nn.Module,
        input: torch.Tensor,
        relevance_fn: "RelevanceFn",
    ) -> torch.Tensor:
        """Return a tensor of candidate root points."""
    
    def get_root_points_for_layer(
        self, 
        for_input_to_layer: NETWORK_LAYER, 
        input: torch.Tensor, 
        relevance_fn: "RelevanceFn",
    ) -> list[RootPoint]:
        assert not isinstance(for_input_to_layer, (UNet2DConditionOutput))
        model = self.module.slice(start=for_input_to_layer)

        cache_key = self.get_cache_key(for_input_to_layer, input, relevance_fn)
        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]
        else:
            print(f"Missed cache@{len(self.cache)} with: ", cache_key)
            self.cache_misses += 1
        
        root_candidates = self.sample_candidates(model, input, relevance_fn)
        if cache_key is not None:
            self.candidate_cache[cache_key] = root_candidates

        candidates_rel = torch.cat(
            [
                relevance_fn(root_candidates.unsqueeze(0)).relevance
                for root_candidate in root_candidates
            ]
        )

        idx = min
        lowest_rel = candidates_rel.argmin(dim=0)
        roots = []
        for j, idx in enumerate(lowest_rel):
            roots.append(
                RootPoint(
                    root=root_candidates[idx].unsqueeze(0),
                    input=input,
                    layer=for_input_to_layer,
                    root_finder=self,
                    explained_neuron=j,
                    relevance=candidates_rel[idx],
                )
            )
        if cache_key is not None:
            self.cache[cache_key] = roots
        return roots

@dataclasses.dataclass(frozen=True)
class NetworkDTDRootFinder(RootFinder):
    """A root finder that uses the network to compute the root points."""

    module: nn.Module # or module: UNet2DConditionModel ???
    explained_output: int
    rule: Rule
    
    def __repr__(self) -> str:
        return (
            f"NetworkDTDRootFinder(explained_output={self.explained_output})"
            f"rule={self.rule}"
        )

    def get_root_points_for_layer(
        self, 
        for_input_to_layer: NETWORK_LAYER, 
        input: torch.Tensor, 
        relevance_fn: "RelevanceFn[REL]",
    ) -> list[RootPoint]:
        assert isinstance(for_input_to_layer, nn.Module)    #  of focus on attention layer
        assert relevance_fn.get_input_layer() == for_input_to_layer

        relevance = relevance_fn(input).relevance
        rule = getattr(for_input_to_layer, "dtd_rule", None) or self.rule
        if is_layer_rule(rule):
            roots: list[Optional[torch.Tensor]] = [
                calculate_root_for_layer(
                    input, for_input_to_layer, rule, relevance
                )
            ]
        else:
            explained_neurons = list(range(for_input_to_layer.out_features))



        return super().get_root_points_for_layer(for_input_to_layer, input, relevance_fn)   


#  --------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Relevance(Generic[REL_FN]):
    """A relevance score.

    Parameters:
        relevance: The relevance score.
        computed_with_fn: The relevance function that was used to compute the
            relevance.

    """
    relevance: torch.Tensor
    computed_with_fn: REL_FN

    def collect_relevance(self) -> Sequence["Relevance"]:
        return [self]




def ensure_2d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.unsqueeze(1)
    return x

class RelevanceFn(Generic[REL], metaclass=abc.ABCMeta):

    def __init__(self, module: nn.Module):
        self.module = module
    
    @abc.abstractmethod
    def get_input_layer(self) -> NETWORK_LAYER:
        """The layer that is used as input to the relevance function."""

    @abc.abstractmethod
    def get_lower_rel_layer(self) -> NETWORK_LAYER:
        """Returns the layer for which the relevance is decomposed."""

    @abc.abstractmethod
    def get_upper_rel_layer(self) -> NETWORK_LAYER:
        """Returns the layer for which **output** relevance is used."""

    @abc.abstractmethod
    def __call__(self, input: torch.Tensor) -> REL:
        """Compute the relevance of the input."""


@dataclasses.dataclass(frozen=True)
class RelevanceFnBase(RelevanceFn[REL], Generic[REL]):

    module: nn.Module
    input_layer: NETWORK_LAYER
    lower_rel_layer: NETWORK_LAYER
    upper_rel_layer: NETWORK_LAYER

    def get_input_layer(self) -> NETWORK_LAYER:
        return self.input_layer

    def get_lower_rel_layer(self) -> NETWORK_LAYER:
        return self.lower_rel_layer

    def get_upper_rel_layer(self) -> NETWORK_LAYER:
        return self.upper_rel_layer

    def __call__(self, input: torch.Tensor) -> REL:
        # raise NotImplementedError()
        return self.compute_relevance(input)

    @abc.abstractmethod
    def compute_relevance(self, input: torch.Tensor) -> REL:
        pass

@dataclasses.dataclass(frozen=True)
class ConstantRelFn(RelevanceFnBase[Relevance]):

    relevance: torch.Tensor

    def __call__(self, input: torch.Tensor) -> Relevance:
        del input
        return Relevance(self.relevance, self)
    
@dataclasses.dataclass(frozen=True)
class NetworkOutputRelecanceFn(RelevanceFnBase[Relevance]):
    """A relevance function that computes the relevance of a network output."""
    module: nn.Module # or module: UNet2DConditionModel ???
    explained_output: NEURON
    input_layer: NETWORK_LAYER
    lower_rel_layer: NETWORK_LAYER = dataclasses.field(init=False)
    upper_rel_layer: NETWORK_LAYER = dataclasses.field(init=False, default_factory=UNet2DConditionOutput)  #  from upper layer's output

    def __post_init__(self):
        object.__setattr__(self, "lower_rel_layer", self.input_layer)
    
    def __call__(self, input: torch.Tensor) -> Relevance:
        output = self.module(input, self.lower_rel_layer)[:, self.explained_output]
        output = ensure_2d(output)
        return Relevance(output, self)

@dataclasses.dataclass(frozen=True)
class OutputRel(RelevanceFnBase[REL_FN], Generic[REL_FN]):
    def __repr__(self) -> str:
        return f"OurputRel(relevance={self.relevance})"
    
@dataclasses.dataclass(frozen=True)
class OutputRelFn(RelevanceFnBase[OutputRel]):
    module: nn.Module # or module: UNet2DConditionModel ???
    explained_output: NEURON

    input_layer: NETWORK_LAYER = dataclasses.field(
        default_factory=UNet2DConditionOutput, init=False
    )
    lower_rel_layer: NETWORK_LAYER = dataclasses.field(
        default_factory=UNet2DConditionOutput, init=False
    )
    upper_rel_layer: NETWORK_LAYER = dataclasses.field(
        default_factory=UNet2DConditionOutput, init=False
    )

    def __repr__(self) -> str:
        return f"OutputRelFn(explained_output={self.explained_output})"
    def __call__(self, input: torch.Tensor) -> "OutputRel[OutputRelFn]":
        rel = input[:, self.explained_output]
        return OutputRel(ensure_2d(rel), self)

@dataclasses.dataclass(frozen=True)
class FullBackwardRel(Relevance):
    roots: list[RootPoint]




REL_FN_FACTORY = Callable[[NETWORK_LAYER, RelevanceFn], RelevanceFn]

def get_decomposed_relevance_fns(
        module: nn.Module,
        explained_output: NEURON,
        relevance_builder: REL_FN_FACTORY,
) -> list[RelevanceFn[Relevance]]:
    rel_fns: list[RelevanceFn] = [
        OutputRelFn(
            module=module,
            explained_output=explained_output,
        )
    ]

    for layer in reversed(list(module.named_modules())):
        rel_fns.append(
            relevance_builder(
                layer=layer,
                rel_fn=rel_fns[-1],
            )
        )

    return rel_fns


#  --------------------------------------------------------

def get_module_by_name(module: nn.Module, name: str) -> nn.Module:
    for n, m in module.named_modules():
        if n == name:
            return m
    raise ValueError(f"Module with name {name} not found in {module}")

def get_relevance_hidden(
        layer: nn.Module,       
        x: torch.Tensor,
        j: int = 0,
        rule: RULE = "z+",
) -> torch.Tensor:
    return get_relevance_hidden_and_root(layer, x, j, rule)[0]

def get_relevance_hidden_and_root(
        layer: nn.Module,
        x: torch.Tensor,
        j: int = 0,
        rule: RULE = "z+",
) -> tuple[torch.Tensor, torch.Tensor]:
    
    with record_all_outputs(layer) as outputs:
        layer(x)
    
    #  outputs

    hidden = outputs[layer][0]

    root = calculate_root_for_single_neuron(x, layer, j, rule)
    return root, layer(root)[:, j]



class record_all_outputs:

    def __init__(self, module: nn.Module):
        self.module = module
        self.outputs: dict[nn.Module, list[torch.Tensor]] = {}
        self.handles: list[hooks.RemovableHandle] = []
    
    def __enter__(self) -> dict[nn.Module, list[torch.Tensor]]:
        def hook(
                module: nn.Module, input: tuple[torch.Tensor], output: torch.Tensor
        ) -> None:
            if isinstance(module.state_dict(), OrderedDict):
                for name, param in module.state_dict().items():
                    if 'weight' in name:
                        print(f"Layer weight: {module.state_dict().get('weight')}")
                    if 'bias' in name:
                        print(f"Layer bias: {module.state_dict().get('bias')}")

            if module not in self.outputs:
                self.outputs[module] = []
            self.outputs[module].append(output)

            self.module.apply(
                lambda module: self.handles.append(
                    module.register_forward_hook(hook)
                )
            )
            return self.outputs
        
        def __exit__(self, *args):
            for handle in self.handles:
                handle.remove()