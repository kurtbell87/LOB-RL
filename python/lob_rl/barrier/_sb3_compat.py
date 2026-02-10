"""SB3 compatibility shim for legacy net_arch format.

SB3 <1.8 supported net_arch=[shared_size, ..., dict(pi=[...], vf=[...])].
SB3 >=1.8 removed shared layers from MlpExtractor and only accepts
net_arch=list[int] or net_arch=dict[str, list[int]].

This module patches:
1. MaskableActorCriticPolicy.__init__ — converts the legacy mixed format
   [int, ..., dict(pi=[...], vf=[...])] to dict(pi=[...], vf=[...]).
2. MlpExtractor.__init__ — adds a shared_net attribute for compatibility
   with code that inspects the network architecture.
"""

import torch.nn as nn


def _convert_legacy_net_arch(net_arch):
    """Convert legacy [int, ..., dict(pi=[], vf=[])] to dict format.

    Returns (converted_net_arch, shared_sizes) where shared_sizes is
    the list of shared layer sizes extracted, or None if no conversion.
    """
    if not isinstance(net_arch, list):
        return net_arch, None
    if len(net_arch) == 0:
        return net_arch, None

    # Check for the mixed format: list with ints followed by a dict
    has_ints = any(isinstance(x, int) for x in net_arch)
    has_dict = any(isinstance(x, dict) for x in net_arch)

    if not (has_ints and has_dict):
        return net_arch, None

    # Extract shared layers (ints) and the dict
    shared = [x for x in net_arch if isinstance(x, int)]
    dicts = [x for x in net_arch if isinstance(x, dict)]
    if len(dicts) != 1:
        return net_arch, None

    d = dicts[0]
    pi_layers = shared + d.get("pi", [])
    vf_layers = shared + d.get("vf", [])
    return dict(pi=pi_layers, vf=vf_layers), shared


# Store shared sizes per policy instance for MlpExtractor patching
_pending_shared_sizes = {}


def patch_sb3_net_arch():
    """Monkey-patch MaskableActorCriticPolicy and MlpExtractor."""
    try:
        from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
        from stable_baselines3.common.torch_layers import MlpExtractor

        # --- Patch MaskableActorCriticPolicy ---
        _original_policy_init = MaskableActorCriticPolicy.__init__

        def _patched_policy_init(self, *args, **kwargs):
            if "net_arch" in kwargs and kwargs["net_arch"] is not None:
                converted, shared = _convert_legacy_net_arch(kwargs["net_arch"])
                kwargs["net_arch"] = converted
                if shared is not None:
                    _pending_shared_sizes[id(self)] = shared
            _original_policy_init(self, *args, **kwargs)
            # Clean up
            _pending_shared_sizes.pop(id(self), None)

        if not getattr(MaskableActorCriticPolicy, "_net_arch_patched", False):
            MaskableActorCriticPolicy.__init__ = _patched_policy_init
            MaskableActorCriticPolicy._net_arch_patched = True

        # --- Patch MlpExtractor to add shared_net ---
        _original_mlp_init = MlpExtractor.__init__

        def _patched_mlp_init(self, feature_dim, net_arch, activation_fn, **kwargs):
            _original_mlp_init(self, feature_dim, net_arch, activation_fn, **kwargs)

            # If net_arch is a dict, determine the shared prefix
            if isinstance(net_arch, dict):
                pi_sizes = net_arch.get("pi", [])
                vf_sizes = net_arch.get("vf", [])
                # Find common prefix
                shared_sizes = []
                for p, v in zip(pi_sizes, vf_sizes):
                    if p == v:
                        shared_sizes.append(p)
                    else:
                        break

                if shared_sizes:
                    # Build a shared_net Sequential from the common prefix layers
                    layers = []
                    in_dim = feature_dim
                    for size in shared_sizes:
                        layers.append(nn.Linear(in_dim, size))
                        layers.append(activation_fn())
                        in_dim = size
                    self.shared_net = nn.Sequential(*layers)
                else:
                    self.shared_net = nn.Sequential()
            elif not hasattr(self, "shared_net"):
                self.shared_net = nn.Sequential()

        if not getattr(MlpExtractor, "_shared_net_patched", False):
            MlpExtractor.__init__ = _patched_mlp_init
            MlpExtractor._shared_net_patched = True

    except ImportError:
        pass


# Apply patches on import
patch_sb3_net_arch()
