################################################################################
# Check required CMake version
set(REQUIRED_CMAKE_VERSION "3.25.0")
cmake_minimum_required(VERSION ${REQUIRED_CMAKE_VERSION})

if(INPUT_POLYFEMPY_DATA_ROOT)
    set(POLYFEMPY_DATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/${INPUT_POLYFEMPY_DATA_ROOT}/")
else()
    set(POLYFEMPY_DATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/data/")
endif()
if(NOT EXISTS ${POLYFEMPY_DATA_ROOT})
    file(MAKE_DIRECTORY ${POLYFEMPY_DATA_ROOT})
endif()

project(polyfempy)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

################################################################################

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/recipes/")

include(polyfem_cpm_cache)

# Color output
include(UseColors)

# Prepend function
include(PrependCurrentPath)

# Extra warnings
include(Warnings)

# Use C++11/14
include(CXXFeatures)

# Sort projects inside the solution
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Generate position independent code by default
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

################################################################################
# Polyfem library
################################################################################

# Use nanobind for Python bindings (pybind11 support removed)
include(CPM)
# Force robin-map to 1.4.1 (or >=1.3.0) BEFORE including polyfem
# This ensures CPM uses the correct version, overriding any indirect dependencies
# (e.g., polyfem's ipc-toolkit may try to use robin-map 1.2.1, which is too old)
# nanobind requires: >= 1.3.0, < 2.0.0
    CPMAddPackage(
        NAME robin-map
        GITHUB_REPOSITORY Tessil/robin-map
        GIT_TAG v1.4.1  # Use 1.4.1 (recommended) or at least >= 1.3.0, < 2.0.0
        OPTIONS
            "ROBIN_MAP_BUILD_TESTS OFF"
            "ROBIN_MAP_BUILD_EXAMPLES OFF"
    )
    if(TARGET tsl::robin_map)
        message(STATUS "Pre-declared robin-map 1.4.1 for nanobind compatibility")
    else()
        message(WARNING "Failed to pre-declare robin-map 1.4.1 - may cause version conflicts")
endif()

# dependencies
include(polyfem)

    # Explicitly disable static library mode BEFORE including nanobind
    # This prevents nanobind from trying to build static library (which requires tsl::robin_map)
    # Must be set before find_package(nanobind) is called
    set(NB_STATIC_LIB OFF CACHE BOOL "Disable nanobind static library" FORCE)
    
    message(STATUS "Using nanobind for Python bindings")
    include(nanobind)

#for testing purpose
include(polyfem_data)

################################################################################
# Subdirectories
################################################################################

    # Use nanobind to create module
    # nanobind provides nanobind_add_module() function
    # Explicitly use NB_SHARED to avoid static library (which requires tsl::robin_map)
    # nanobind defaults to NB_STATIC, so we must explicitly request shared mode
    nanobind_add_module(polyfempy NB_SHARED)
    
    # Verify no static library target was created (safety check)
    if(TARGET nanobind-static)
        message(FATAL_ERROR 
            "nanobind-static target was created. This requires tsl::robin_map.\n"
            "Please ensure NB_STATIC_LIB is OFF and nanobind is used in header-only mode."
        )
    endif()
    
    add_subdirectory(src)
    target_include_directories(polyfempy PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(polyfempy PRIVATE polyfem::polyfem)
    # Define macro for source files
    target_compile_definitions(polyfempy PRIVATE USE_NANOBIND)

# Ensure the produced module has the correct Python extension suffix.
# Prefer FindPython's canonical variable `Python_EXTENSION_MODULE_SUFFIX`.
# Example on Windows: `.cp311-win_amd64.pyd`
if(DEFINED Python_EXTENSION_MODULE_SUFFIX AND NOT "${Python_EXTENSION_MODULE_SUFFIX}" STREQUAL "")
    set_target_properties(polyfempy PROPERTIES PREFIX "" SUFFIX "${Python_EXTENSION_MODULE_SUFFIX}")
elseif(DEFINED Python_EXTENSION_SUFFIX AND NOT "${Python_EXTENSION_SUFFIX}" STREQUAL "")
    # Backwards/alt name in some CMake setups.
    set_target_properties(polyfempy PROPERTIES PREFIX "" SUFFIX "${Python_EXTENSION_SUFFIX}")
elseif(WIN32)
    # Last resort: ensure it is importable as a Python extension on Windows.
    set_target_properties(polyfempy PROPERTIES PREFIX "" SUFFIX ".pyd")
else()
    message(WARNING "Python extension module suffix not defined; leaving default target suffix.")
endif()
