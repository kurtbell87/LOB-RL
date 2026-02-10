"""Shared configuration utilities for LOB-RL Python wrappers."""


def make_session_config(session_config):
    """Convert session_config (None, dict, or SessionConfig) to C++ SessionConfig.

    Accepts:
        None -> default RTH config
        dict -> construct from keys rth_open_ns, rth_close_ns, warmup_messages
        SessionConfig -> pass through
    """
    import lob_rl_core

    if session_config is None:
        return lob_rl_core.SessionConfig.default_rth()
    elif isinstance(session_config, dict):
        cfg = lob_rl_core.SessionConfig()
        for key in ("rth_open_ns", "rth_close_ns", "warmup_messages"):
            if key in session_config:
                setattr(cfg, key, session_config[key])
        return cfg
    else:
        return session_config
