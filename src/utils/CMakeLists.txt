include_directories(${PARPT_INCDIRS})
# add all the utils libraries to the overall utils object
add_library(Utils parPT_io.cpp
                  mass_transfer.cpp
                  random_walk.cpp
                  diffusion.cpp
                  particles.cpp)
set(Utils PARENT_SCOPE)

file(COPY python_scripts/choose_Nchunks.py3; python_scripts/gen_pts.py3;
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/python_scripts)
