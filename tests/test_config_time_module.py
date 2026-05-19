def test_time_config_blocks_live_in_config_time_module():
    from polyfempy.api import config
    from polyfempy.api import config_time

    names = [
        "BDFIntegrator",
        "ImplicitNewmarkIntegrator",
        "Time",
    ]

    for name in names:
        assert getattr(config, name) is getattr(config_time, name)
