from __future__ import annotations


def test_guided_config_module_owns_translation_helpers():
    from polyfempy.api import _guided_config as config_builder

    expected = [
        "mesh_file",
        "build_material",
        "build_surface_selection",
        "build_space",
        "build_geometry_extra",
        "build_time",
        "build_solver",
        "build_contact",
        "build_output",
        "add_body_from_section",
        "build_config",
    ]

    for name in expected:
        assert hasattr(config_builder, name)


def test_guided_sections_reexports_config_translation_for_compatibility():
    from polyfempy.api import _guided_config as config_builder
    from polyfempy.api import guided_sections

    expected = [
        "build_config",
        "build_material",
        "add_body_from_section",
    ]

    for name in expected:
        assert getattr(guided_sections, name) is getattr(config_builder, name)
