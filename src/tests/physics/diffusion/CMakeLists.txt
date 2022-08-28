# we need stuff from the global include dirs
include_directories(${PARPT_INCDIRS})

# add the tests to ctest
find_program(BASH_PROGRAM bash)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_mt_test.sh.in
               ${CMAKE_CURRENT_BINARY_DIR}/run_mt_test.sh @ONLY)
add_test(MT_test ${BASH_PROGRAM}
         ${CMAKE_CURRENT_BINARY_DIR}/run_mt_test.sh)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_rw_test.sh.in
               ${CMAKE_CURRENT_BINARY_DIR}/run_rw_test.sh @ONLY)
add_test(RW_test ${BASH_PROGRAM}
         ${CMAKE_CURRENT_BINARY_DIR}/run_rw_test.sh)
# copy the data and utils dirs
file(COPY utils data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${CMAKE_INSTALL_PREFIX}/src/utils/python_scripts/gen_pts.py3
                ${CMAKE_CURRENT_BINARY_DIR}/utils/gen_pts)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_INSTALL_PREFIX}/src/utils/python_scripts/choose_Nchunks.py3
        ${CMAKE_CURRENT_BINARY_DIR}/utils/chunks_chooser)
