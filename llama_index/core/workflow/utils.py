# SPDX-License-Identifier: MIT
# Copyright (c) 2025 LlamaIndex Inc.

from __future__ import annotations

import inspect
from typing import (
    Annotated,
    Any,
    Callable,
    Optional,
    get_args,
    get_origin,
    get_type_hints,
)

try:
    from typing import Union
except ImportError:  # pragma: no cover
    from typing_extensions import Union

# handle python version compatibility
try:
    from types import UnionType  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    from typing import Union as UnionType  # type: ignore[assignment]

from pydantic import BaseModel

from .errors import WorkflowValidationError
from .events import Event, EventType
from .resource import ResourceDefinition

BUSY_WAIT_DELAY = 0.01


class StepSignatureSpec(BaseModel):
    """A Pydantic model representing the signature of a step function or method."""

    accepted_events: dict[str, list[EventType]]
    return_types: list[Any]
    context_parameter: str | None
    context_state_type: Any | None
    resources: list[Any]


def inspect_signature(fn: Callable) -> StepSignatureSpec:
    """
    Given a function, ensure the signature is compatible with a workflow step.

    Args:
        fn (Callable): The function to inspect.

    Returns:
        StepSignatureSpec: A specification object containing:
            - accepted_events: Dictionary mapping parameter names to their event types
            - return_types: List of return type annotations
            - context_parameter: Name of the context parameter if present

    Raises:
        TypeError: If fn is not a callable object

    """
    if not callable(fn):
        raise TypeError(f"Expected a callable object, got {type(fn).__name__}")

    sig = inspect.signature(fn)
    type_hints = get_type_hints(fn, include_extras=True)

    accepted_events: dict[str, list[EventType]] = {}
    context_parameter = None
    context_state_type = None
    resources = []

    # Inspect function parameters
    for name, t in sig.parameters.items():
        # Ignore self and cls
        if name in ("self", "cls"):
            continue

        annotation = type_hints.get(name, t.annotation)

        # Handle Context[StateType] annotations
        if get_origin(annotation) is not None:
            origin = get_origin(annotation)
            args = get_args(annotation)

            # Check if this is Context[StateType]
            if hasattr(origin, "__name__") and origin.__name__ == "Context":
                context_parameter = name
                # Extract state type from generic parameter
                if args:
                    context_state_type = args[0]
                continue

        # Handle Annotated types for resources
        if get_origin(annotation) is Annotated:
            _, resource = get_args(annotation)
            resources.append(ResourceDefinition(name=name, resource=resource))
            continue

        # Get name and type of the Context param (without state type)
        if hasattr(annotation, "__name__") and annotation.__name__ == "Context":
            context_parameter = name
            continue

        # Collect name and types of the event param
        param_types = _get_param_types(t, type_hints)
        if all(
            param_t == Event
            or (inspect.isclass(param_t) and issubclass(param_t, Event))
            for param_t in param_types
        ):
            accepted_events[name] = param_types
            continue

    return StepSignatureSpec(
        accepted_events=accepted_events,
        return_types=_get_return_types(fn),
        context_parameter=context_parameter,
        context_state_type=context_state_type,
        resources=resources,
    )


def validate_step_signature(spec: StepSignatureSpec) -> None:
    """
    Validate that a step signature specification meets workflow requirements.

    Args:
        spec (StepSignatureSpec): The signature specification to validate.

    Raises:
        WorkflowValidationError: If the signature is invalid for a workflow step.

    """
    num_of_events = len(spec.accepted_events)
    if num_of_events == 0:
        msg = "Step signature must have at least one parameter annotated as type Event"
        raise WorkflowValidationError(msg)
    elif num_of_events > 1:
        msg = f"Step signature must contain exactly one parameter of type Event but found {num_of_events}."
        raise WorkflowValidationError(msg)

    if not spec.return_types:
        msg = "Return types of workflows step functions must be annotated with their type."
        raise WorkflowValidationError(msg)


def get_steps_from_class(_class: object) -> dict[str, Callable]:
    """
    Given a class, return the list of its methods that were defined as steps.

    Args:
        _class (object): The class to inspect for step methods.

    Returns:
        dict[str, Callable]: A dictionary mapping step names to their corresponding methods.

    """
    step_methods: dict[str, Callable] = {}
    all_methods = inspect.getmembers(_class, predicate=inspect.isfunction)

    for name, method in all_methods:
        if hasattr(method, "__step_config"):
            step_methods[name] = method

    return step_methods


def get_steps_from_instance(workflow: object) -> dict[str, Callable]:
    """
    Given a workflow instance, return the list of its methods that were defined as steps.

    Args:
        workflow (object): The workflow instance to inspect.

    Returns:
        dict[str, Callable]: A dictionary mapping step names to their corresponding methods.

    """
    step_methods: dict[str, Callable] = {}
    all_methods = inspect.getmembers(workflow, predicate=inspect.ismethod)

    for name, method in all_methods:
        if hasattr(method, "__step_config"):
            step_methods[name] = method

    return step_methods


def _get_param_types(param: inspect.Parameter, type_hints: dict) -> list[Any]:
    """
    Extract and process the types of a parameter.

    This helper function handles Union and Optional types, returning a list of the actual types.
    For Union[A, None] (Optional[A]), it returns [A].

    Args:
        param (inspect.Parameter): The parameter to analyze.
        type_hints (dict): The resolved type hints for the function.

    Returns:
        list[Any]: A list of extracted types, excluding None from Unions/Optionals.

    """
    typ = type_hints.get(param.name, param.annotation)
    if typ is inspect.Parameter.empty:
        return [Any]
    if get_origin(typ) in (Union, Optional, UnionType):
        return [t for t in get_args(typ) if t is not type(None)]
    return [typ]


def _get_return_types(func: Callable) -> list[Any]:
    """
    Extract the return type hints from a function.

    Handles Union, Optional, and List types.
    """
    type_hints = get_type_hints(func)
    return_hint = type_hints.get("return")
    if return_hint is None:
        return []

    origin = get_origin(return_hint)
    if origin in (Union, UnionType):
        # Optional is Union[type, None] so it's covered here
        return [t for t in get_args(return_hint) if t is not type(None)]
    else:
        return [return_hint]


def is_free_function(qualname: str) -> bool:
    """
    Determines whether a certain qualified name points to a free function.

    A free function is either a module-level function or a nested function.
    This implementation follows PEP-3155 for handling nested function detection.

    Args:
        qualname (str): The qualified name to analyze.

    Returns:
        bool: True if the name represents a free function, False otherwise.

    Raises:
        ValueError: If the qualified name is empty.

    """
    if not qualname:
        msg = "The qualified name cannot be empty"
        raise ValueError(msg)

    toks = qualname.split(".")
    if len(toks) == 1:
        # e.g. `my_function`
        return True
    elif "<locals>" not in toks:
        # e.g. `MyClass.my_method`
        return False
    else:
        return toks[-2] == "<locals>"
