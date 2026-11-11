include("${PROJECT_SOURCE_DIR}/polyfem/cmake/recipes/cli11.cmake")

# Compile the unmodified upstream entry point against the same library.
add_executable(polyfem_reference_cli "${PROJECT_SOURCE_DIR}/polyfem/src/polyfem/main.cpp")
target_link_libraries(polyfem_reference_cli PRIVATE polyfem::polyfem CLI11::CLI11)

add_executable(polyfem_reference_probe reference.cpp)
target_link_libraries(polyfem_reference_probe PRIVATE polyfem::polyfem)
target_compile_features(polyfem_reference_probe PRIVATE cxx_std_17)
set_target_properties(polyfem_reference_cli polyfem_reference_probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_BINARY_DIR}/Release")
