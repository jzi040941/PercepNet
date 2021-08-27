set(BINARY ${CMAKE_PROJECT_NAME})

file(GLOB_RECURSE SOURCES LIST_DIRECTORIES true *.h *.cpp)

set(SOURCES ${SOURCES})

add_executable(${BINARY} ${SOURCES})
set_target_properties(
    ${BINARY}
    PROPERTIES
    COMPILE_DEFINITIONS TRAINING=1
    )
if(EXISTS "${PROJECT_SOURCE_DIR}/src/nnet_data.cpp")
    add_executable(${BINARY}_run ${SOURCES})
else()
    message([WARNING] "nnet_data.cpp is not exist. Do not generate inference executable" ...)
endif()



add_library(${BINARY}_lib STATIC ${SOURCES})
