from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cmake_defaults_to_polyfem_data_submodule():
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert '${CMAKE_CURRENT_SOURCE_DIR}/polyfem-data' in cmake
    assert '${CMAKE_CURRENT_SOURCE_DIR}/data/' not in cmake
    assert "file(MAKE_DIRECTORY ${POLYFEMPY_DATA_ROOT})" not in cmake


def test_polyfem_data_recipe_does_not_fetch_data_repo():
    recipe = (ROOT / "cmake" / "recipes" / "polyfem_data.cmake").read_text(
        encoding="utf-8"
    )

    assert "FetchContent" not in recipe
    assert "GIT_REPOSITORY https://github.com/polyfem/polyfem-data" not in recipe
    assert "git submodule update --init --recursive" in recipe
    assert "FATAL_ERROR" in recipe
