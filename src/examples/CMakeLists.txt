# we need stuff from the global include dirs
include_directories(${PARPT_INCDIRS})

set(tests "MT_only" ; "RW_only" ; "RWMT")
foreach(test ${tests})
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${test}/run_${test}.sh.in
               ${CMAKE_CURRENT_BINARY_DIR}/${test}/run_${test}.sh @ONLY)
  install(DIRECTORY ${test}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/${test})
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/${test}/plot_${test}.ipynb DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/${test})
  # add a symlink for the chunking script
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
          ${CMAKE_INSTALL_PREFIX}/src/utils/python_scripts/gen_pts.py3
          ${CMAKE_CURRENT_BINARY_DIR}/${test}/gen_pts)
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
          ${CMAKE_INSTALL_PREFIX}/src/utils/python_scripts/choose_Nchunks.py3
          ${CMAKE_CURRENT_BINARY_DIR}/${test}/chunks_chooser)
endforeach()
