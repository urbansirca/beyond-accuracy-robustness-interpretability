import types
from typing import Dict

import torch.nn as nn

from interpretability.lrp.backward import LRP_DEBUG


def save_and_replace(module, new_forward):
    """Store original forward and swap in new_forward (bound as method)."""
    module._lrp_orig_forward = module.forward
    module.forward = types.MethodType(new_forward, module)


def restore(module):
    """Undo save_and_replace."""
    if hasattr(module, "_lrp_orig_forward"):
        module.forward = module._lrp_orig_forward
        del module._lrp_orig_forward


def unpatch_all(model):
    """Restore every module that was patched."""
    for module in model.modules():
        restore(module)


def walk_and_patch(model, rules, *, debug_label=None):
    """
    Walk every module and apply the first matching rule's forward.
    Returns the per-rule patch count dict.
    """
    counts: Dict[str, int] = {}
    for module in model.modules():
        for matcher, fn in rules:
            if isinstance(matcher, type):
                if isinstance(module, matcher):
                    save_and_replace(module, fn)
                    counts[matcher.__name__] = counts.get(matcher.__name__, 0) + 1
                    break
            elif callable(matcher) and matcher(module):
                key = getattr(matcher, "__name__", repr(matcher))
                save_and_replace(module, fn)
                counts[key] = counts.get(key, 0) + 1
                break

    if LRP_DEBUG and debug_label is not None:
        print(f"  [{debug_label}] summary: {counts}")
    return counts


def by_classname(*names):
    names_set = set(names)
    pred = lambda m: m.__class__.__name__ in names_set
    pred.__name__ = "/".join(names)
    return pred
