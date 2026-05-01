from __future__ import annotations

from tau.config import TauConfig
import tau.cli as cli_mod


def test_make_agent_config_sets_code_index_flag():
    cfg = TauConfig()
    agent_cfg = cli_mod._make_agent_config(
        tau_config=cfg,
        provider=None,
        model=None,
        think=None,
        no_confirm=False,
        workspace=".",
        code_index=True,
    )
    assert agent_cfg.code_index_enabled is True

