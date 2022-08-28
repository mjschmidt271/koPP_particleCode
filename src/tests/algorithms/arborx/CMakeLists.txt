# we need stuff from the global include dirs
include_directories(${PARPT_INCDIRS})

add_executable(AX_test arborx_test.cpp)
add_executable(AX_compare ax_brute_compare.cpp)
# add the tests to ctest
# NOTE: requires a little extra work since the tests are run by bash scripts
  # and involve python scripts
find_program(BASH_PROGRAM bash)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_ax_compare.sh.in
               ${CMAKE_CURRENT_BINARY_DIR}/run_ax_compare.sh @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_ax_test.sh.in
               ${CMAKE_CURRENT_BINARY_DIR}/run_ax_test.sh @ONLY)
add_test(AX_smoke_test ${BASH_PROGRAM}
         ${CMAKE_CURRENT_BINARY_DIR}/run_ax_test.sh)
add_test(AX_compare_test ${BASH_PROGRAM}
         ${CMAKE_CURRENT_BINARY_DIR}/run_ax_compare.sh)
# copy the required python scripts over
file(COPY utils/verify_arborx.py3 utils/gen_pts_AXtest.py3 DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/utils)

add_dependencies(Utils yaml_cpp kokkos kokkoskernels arborx spdlog)

if(PARPT_USE_CUDA)
  cuda_add_cublas_to_target(AX_compare)
endif()

target_link_libraries(AX_test ${LIBRARIES})
target_link_libraries(AX_compare ${LIBRARIES})

# add the include directories
target_include_directories(AX_test PUBLIC
                           ${PARPT_INCDIRS})
target_include_directories(AX_compare PUBLIC
                           ${PARPT_INCDIRS})
