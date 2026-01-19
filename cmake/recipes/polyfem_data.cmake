# data
# License: MIT

message(STATUS "Third-party: fetching 'polyfem data'")

# Check if data directory already exists and contains files
# If it does, skip downloading
if(EXISTS ${POLYFEMPY_DATA_ROOT} AND EXISTS ${POLYFEMPY_DATA_ROOT}/README.md)
    message(STATUS "Using existing polyfem data directory: ${POLYFEMPY_DATA_ROOT}")
    set(polyfem_data_POPULATED TRUE)
else()
    include(FetchContent)
    FetchContent_Declare(
        polyfem_data
        GIT_REPOSITORY https://github.com/polyfem/polyfem-data
        GIT_TAG f2089eb6eaa22071f7490e0f144e10afe85d4eba
        GIT_SHALLOW FALSE
        SOURCE_DIR ${POLYFEMPY_DATA_ROOT}
    )
    FetchContent_GetProperties(polyfem_data)
    if(NOT polyfem_data_POPULATED)
      FetchContent_Populate(polyfem_data)
      # SET(POLYFEM_DATA_DIR ${polyfem_data_SOURCE_DIR})
    endif()
endif()