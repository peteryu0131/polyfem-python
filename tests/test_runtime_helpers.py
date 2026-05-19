from types import SimpleNamespace

from polyfempy.api.runtime import result_output


def test_result_output_uses_generic_default_names():
    cfg = SimpleNamespace(output=None)

    result_output(cfg)

    assert cfg.output.json == "results.json"
    assert cfg.output.paraview.file_name == "results.pvd"
    assert cfg.output.advanced.timestep_prefix == "step_"


def test_result_output_preserves_explicit_output_names():
    cfg = SimpleNamespace(output=None)

    result_output(
        cfg,
        json_name="impact_stats.json",
        pvd_name="impact.pvd",
        timestep_prefix="impact_step_",
    )

    payload = cfg.output.to_dict()
    assert payload["json"] == "impact_stats.json"
    assert payload["paraview"]["file_name"] == "impact.pvd"
    assert payload["advanced"]["timestep_prefix"] == "impact_step_"
