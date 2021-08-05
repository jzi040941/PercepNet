set(BINARY ${CMAKE_PROJECT_NAME})

file(GLOB_RECURSE SOURCES LIST_DIRECTORIES true *.h *.cpp)

set(SOURCES ${SOURCES})

add_executable(${BINARY} ${SOURCES})
add_executable(${BINARY}_run ${SOURCES})
set_target_properties(
    ${BINARY}_run
    PROPERTIES
    COMPILE_DEFINITIONS TRAINING=0
)

add_library(${BINARY}_lib STATIC ${SOURCES})