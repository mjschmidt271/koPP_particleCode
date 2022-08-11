# we need stuff from the global include dirs
include_directories(${PARPT_INCDIRS})

# copy the utils dir
file(COPY utils DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set(tests "MT_only" ; "RW_only" ; "RWMT")
foreach(test ${tests})
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${test}/run_${test}.sh.in
               ${CMAKE_CURRENT_BINARY_DIR}/${test}/run_${test}.sh @ONLY)
  install(DIRECTORY ${test}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/${test})
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/${test}/plot_${test}.ipynb DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/${test})
endforeach()
