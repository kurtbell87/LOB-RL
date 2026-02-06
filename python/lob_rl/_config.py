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
        if "rth_open_ns" in session_config:
            cfg.rth_open_ns = session_config["rth_open_ns"]
        if "rth_close_ns" in session_config:
            cfg.rth_close_ns = session_config["rth_close_ns"]
        if "warmup_messages" in session_config:
            cfg.warmup_messages = session_config["warmup_messages"]
        return cfg
    else:
        return session_config
