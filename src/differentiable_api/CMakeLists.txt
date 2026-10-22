set(SOURCES
  session.cpp
  von_mises.cpp
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Source Files" FILES ${SOURCES})
target_sources(polyfempy PRIVATE ${SOURCES})
