cmake_minimum_required(VERSION 3.17)
CMAKE_POLICY(SET CMP0074 NEW)

# set the project name and version
project(parPT VERSION 0.0.1 LANGUAGES CXX)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_testing()
# this is to account for the stochastic variability with random walks
# we give the tests 3 trys
set(CMAKE_CTEST_ARGUMENTS --repeat until-pass:3)

string(TOUPPER "${PARPT_BUILD_TYPE}" CMAKE_BUILD_TYPE)
if (CMAKE_BUILD_TYPE STREQUAL DEBUG)
  # set compiler debug flag
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
  # set compiler all warnings flag
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
elseif(CMAKE_BUILD_TYPE STREQUAL RELEASE)
  # set compiler debug flag
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -DNDEBUG")
else()
  message(STATUS "invalid build type--options: debug, release")
endif()
message(STATUS "Build type = ${CMAKE_BUILD_TYPE}")

# Are we on Linux?
if (UNIX AND NOT APPLE)
  set(LINUX ON)
endif()

# for some reason, Linux machines (sometimes) need the extra flag
if(LINUX)
  set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl")
endif()

# openmp/cuda flags
if (PARPT_USE_OPENMP)
  set(OMP_FLAGS "-fopenmp")
  message(STATUS "OpenMP flags are: ${OMP_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OMP_FLAGS}")
endif()
if(PARPT_USE_CUDA)
  set(CUDA_FLAGS "--expt-extended-lambda")
  set(CUDA_FLAGS "${CUDA_FLAGS} --expt-relaxed-constexpr")
  set(CUDA_FLAGS "${CUDA_FLAGS} -fopenmp")
  # this is due to some weirdness with the kokkos nvcc_wrapper
  if("$ENV{DEVICE_ARCH}" STREQUAL "TURING75")
    set(CUDA_FLAGS "${CUDA_FLAGS} -arch=sm_75")
  endif()
  message(STATUS "Cuda flags are: ${CUDA_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CUDA_FLAGS} ${OMP_FLAGS}")
endif()

message(STATUS "CMAKE_CXX_FLAGS flags are ${CMAKE_CXX_FLAGS}")

# add the ext directory, with the external TPLs
add_subdirectory(ext)

# add the source files directory
add_subdirectory(src)

# add the profiling directory and make the run script executable
install(DIRECTORY profiling DESTINATION ${CMAKE_INSTALL_PREFIX})
install(PROGRAMS profiling/run_profile_ens.sh; profiling/bg_run_ens.sh DESTINATION ${CMAKE_INSTALL_PREFIX}/profiling)
