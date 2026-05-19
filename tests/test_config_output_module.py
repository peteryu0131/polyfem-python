def test_output_config_blocks_live_in_config_output_module():
    from polyfempy.api import config
    from polyfempy.api import config_output

    names = [
        "OutputLog",
        "OutputParaviewOptions",
        "ParaviewOutput",
        "OutputDataAdvanced",
        "OutputData",
        "OutputAdvanced",
        "OutputReference",
        "ResultOutput",
        "FallbackOutput",
        "Output",
    ]

    for name in names:
        assert getattr(config, name) is getattr(config_output, name)
