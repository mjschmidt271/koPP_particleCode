# Precision of floating point numbers.
if (PARPT_PRECISION STREQUAL "single")
  set(PARPT_REAL_TYPE "float")
  set(PARPT_DOUBLE_PRECISION 0)
elseif(PARPT_PRECISION STREQUAL "double")
  set(PARPT_REAL_TYPE "double")
  set(PARPT_DOUBLE_PRECISION 1)
elseif(PARPT_PRECISION)
  message(FATAL_ERROR "Invalid PARPT_PRECISION: ${PARPT_PRECISION} (use 'single' or 'double')")
else()
  message(FATAL_ERROR "PARPT_PRECISION not set (use 'single' or 'double')")
endif()
message(STATUS "Using ${PARPT_PRECISION} precision floating point numbers")

# type of fixed-radius search to use
# Precision of floating point numbers.
if (PARPT_SEARCH_TYPE STREQUAL "tree")
  set(PARPT_SEARCH_POLICY "TreeCRSPolicy")
  set(PARPT_TREE_SEARCH 1)
elseif(PARPT_SEARCH_TYPE STREQUAL "brute_force")
  set(PARPT_SEARCH_POLICY "BruteForceCRSPolicy")
  set(PARPT_TREE_SEARCH 0)
elseif(PARPT_SEARCH_TYPE)
  message(FATAL_ERROR "Invalid PARPT_SEARCH_TYPE: ${PARPT_SEARCH_TYPE} (use 'tree' or 'brute_force')")
else()
  message(FATAL_ERROR "PARPT_SEARCH_TYPE not set (use 'tree' or 'brute_force')")
endif()
message(STATUS "Using ${PARPT_SEARCH_TYPE} for fixed-radius search")

# generate the configure file
configure_file("${PROJECT_SOURCE_DIR}/src/utils/parPT_config.hpp.in"
               "${PROJECT_BINARY_DIR}/src/utils/parPT_config.hpp"
               @ONLY
               )

message(STATUS "PARPT_REAL_TYPE = ${PARPT_REAL_TYPE}")

list(APPEND PARPT_INCDIRS;
     ${PROJECT_SOURCE_DIR}/src/utils;
     ${PROJECT_BINARY_DIR}/src/utils)

# add the utils directory
add_subdirectory(utils)

# Report the installation prefix.
message(STATUS "Installation prefix is ${CMAKE_INSTALL_PREFIX}")

# add the executable and link the libraries
add_executable(parPT parPT.cpp)

if(PARPT_USE_CUDA)
  find_package(CUDA)
  cuda_add_cublas_to_target(parPT)
  list(APPEND LIBRARIES; ${CUDA_cusparse_LIBRARY})
endif()

set(PARPT_LIBS Utils)
set(LIBRARIES ${PARPT_LIBS}; ${LIBRARIES})

# NOTE: in order for this to see the PARPT_LIBS, this has be be after they are
# appended to LIBRARIES
add_subdirectory(tests)

target_link_libraries(parPT ${LIBRARIES})

message(STATUS "Libraries = ${LIBRARIES}")

# add the include directories
target_include_directories(parPT PUBLIC ${PARPT_INCDIRS})

# add the examples subdirectory
add_subdirectory(examples)

install(TARGETS parPT Utils)
