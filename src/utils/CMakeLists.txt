include_directories(${PARPT_INCDIRS})
# add all the utils libraries to the overall utils object
add_library(Utils parPT_io.cpp
                  mass_transfer.cpp
                  random_walk.cpp
                  diffusion.cpp
                  particles.cpp)
# target_link_libraries(Utils ${PARPT_LIBS})
set(Utils PARENT_SCOPE)
# add_dependencies(Utils yaml_cpp kokkos kokkoskernels arborx spdlog version_info)
# copy the plotting directory to install dir
install(DIRECTORY plotting DESTINATION ${CMAKE_INSTALL_PREFIX})
# note that the type_defs.hpp header is found by adding this directory to the
# target_include_directories for the parPT project
