#!/bin/bash

# ==============================================================================
export REAL_TYPE="double"
# export REAL_TYPE="single"
export SEARCH_TYPE="tree"
# export SEARCH_TYPE="brute_force"
# ==============================================================================
# ==============================================================================
# NOTE: CMAKE_BUILD_TYPE=RELEASE (set below) defaults to -03 -DNDEBUG
# consider changing this to -O2 if problems arise
export BUILD_TYPE="debug"
# export BUILD_TYPE="release"
# default to installing in the direction the `make` command is run
# (most commonly the 'build' directory)
export INSTALL_LOCATION="./"
# ==============================================================================
# parallel accelerator options
# ==============================================================================
# choose only one of these build options
# (1)
# export USE_OPENMP=false
# export USE_CUDA=false
# (2)
export USE_OPENMP=true
export USE_CUDA=false
# (3) ***NOT CURRENTLY SUPPORTED***
# export USE_OPENMP=false
# export USE_CUDA=true
# (4)
# export USE_OPENMP=true
# export USE_CUDA=true
# ==============================================================================
# if you are using a machine that isn't listed below and you are building
# for GPU, provide the architecture name
# see the "Architecture Keywords" section of:
# https://github.com/kokkos/kokkos/wiki/Compiling
# export GPU_ARCHITECTURE="schmidt27"
# ==============================================================================
# ==============================================================================
# compiler options
# ==============================================================================
# change these to the compilers you plan to use
# note that cuda builds require the CXX compiler to be the
# kokkos-provided nvcc_wrapper
export MAC_SERIAL_CPP="clang++"
export MAC_SERIAL_CC="clang"
export MAC_OMP_CPP="g++-13"
export MAC_OMP_CC="gcc-13"
export LINUX_CPP="g++"
export LINUX_CC="gcc"
# ==============================================================================
# I wouldn't change this unless you have good reason and know what you're doing
# ==============================================================================
export MAC_LIBDIR="lib"
export LINUX_LIBDIR="lib64"
export WIN_LIBDIR="???"
# ==============================================================================
# machine-dependent build options
# ==============================================================================
# these are for the various machines I use.
# I would recommend either adding similar logic for yours, or just
# hard coding for your environment of choice below all this logic
export MACHINE=`hostname`
export HOME_DIR=$HOME
echo "Configuring for ${MACHINE}"
if [ $MACHINE = s1046231 ]; then
    export LIBDIR=$MAC_LIBDIR
    if [ "$USE_OPENMP" = false ]; then
        echo "Building for serial"
        export KO_ROOT="${HOME_DIR}/kokkos/install_clang"
        export KK_ROOT="${HOME_DIR}/kokkos-kernels/install_clang"
        export YCPP_ROOT="${HOME_DIR}/yaml-cpp/install_clang"
        export AX_ROOT="${HOME_DIR}/ArborX/install_clang"
        export SP_ROOT="${HOME_DIR}/spdlog/install_clang_${BUILD_TYPE}"
    elif [ "$USE_OPENMP" = true ]; then
        echo "Building for OpenMP"
        export KO_ROOT="${HOME_DIR}/kokkos/install_gccomp"
        export KK_ROOT="${HOME_DIR}/kokkos-kernels/install_gccomp"
        export YCPP_ROOT="${HOME_DIR}/yaml-cpp/install_gcc11"
        export AX_ROOT="${HOME_DIR}/ArborX/install_gccomp"
        export SP_ROOT="${HOME_DIR}/spdlog/install_gcc_${BUILD_TYPE}"
    else
        echo "ERROR: Unsupported build for this machine"
        exit
    fi
elif [ $MACHINE = s1024454 ]; then
    export LIBDIR=$LINUX_LIBDIR
    export LINUX_CPP=$S102_CPP
    if [ "$USE_OPENMP" = false ] && [ "$USE_CUDA" = false ]; then
        echo "Building for serial"
        echo "ERROR: Unsupported build for this machine"
        exit
        # export KO_ROOT="${HOME_DIR}/kokkos/"
        # export KK_ROOT="${HOME_DIR}/kokkos-kernels/"
        # export YCPP_ROOT="${HOME_DIR}/yaml-cpp/"
        # export AX_ROOT="${HOME_DIR}/ArborX/"
    elif [ "$USE_OPENMP" = true ] && [ "$USE_CUDA" = false ]; then
        echo "Building for OpenMP without CUDA"
        # export KO_ROOT="${HOME_DIR}/kokkos/install_omp_${BUILD_TYPE}"
        # export KK_ROOT="${HOME_DIR}/kokkos-kernels/install_omp_${BUILD_TYPE}"
        # export YCPP_ROOT="${HOME_DIR}/yaml-cpp/install"
        # export AX_ROOT="${HOME_DIR}/ArborX/install_omp_${BUILD_TYPE}"
        # export SP_ROOT="${HOME_DIR}/spdlog/install_${BUILD_TYPE}"
    elif [ "$USE_OPENMP" = false ] && [ "$USE_CUDA" = true ]; then
        echo "Building without OpenMP"
        echo "ERROR: Unsupported build for this machine"
        exit
        # export DEVICE_ARCH="MAXWELL52"
        # export KO_ROOT="${HOME_DIR}/kokkos/"
        # export KK_ROOT="${HOME_DIR}/kokkos-kernels/"
        # export YCPP_ROOT="${HOME_DIR}/yaml-cpp/"
        # export AX_ROOT="${HOME_DIR}/ArborX/"
    elif [ "$USE_OPENMP" = true ] && [ "$USE_CUDA" = true ]; then
        echo "Building for OpenMP and CUDA"
        # export DEVICE_ARCH="MAXWELL52"
        # export KO_ROOT="${HOME_DIR}/kokkos/install_cuda_${BUILD_TYPE}"
        # export KK_ROOT="${HOME_DIR}/kokkos-kernels/install_cuda_${BUILD_TYPE}"
        # export YCPP_ROOT="${HOME_DIR}/yaml-cpp/install"
        # export AX_ROOT="${HOME_DIR}/ArborX/install_cuda_${BUILD_TYPE}"
        # export SP_ROOT="${HOME_DIR}/spdlog/install_${BUILD_TYPE}"
    else
        echo "ERROR: Unsupported build for this machine"
        exit
    fi
elif [ $MACHINE = clamps ]; then
    export HOME_DIR="${HOME}/mjschmidt"
    export LIBDIR=$MAC_LIBDIR
    if [ "$USE_OPENMP" = false ] && [ "$USE_CUDA" = false ]; then
        echo "Building for serial"
        echo "ERROR: Unsupported build for this machine"
        exit
        # export KO_ROOT="${HOME_DIR}/kokkos/"
        # export KK_ROOT="${HOME_DIR}/kokkos-kernels/"
        # export YCPP_ROOT="${HOME_DIR}/yaml-cpp/"
        # export AX_ROOT="${HOME_DIR}/ArborX/"
    elif [ "$USE_OPENMP" = true ] && [ "$USE_CUDA" = false ]; then
        echo "Building for OpenMP without CUDA"
        export KO_ROOT="${HOME_DIR}/kokkos/install_omp"
        export KK_ROOT="${HOME_DIR}/kokkos-kernels/install_omp"
        export YCPP_ROOT="${HOME_DIR}/yaml-cpp/install"
        export AX_ROOT="${HOME_DIR}/ArborX/install_omp"
        export SP_ROOT="${HOME_DIR}/spdlog/install_${BUILD_TYPE}"
    elif [ "$USE_OPENMP" = false ] && [ "$USE_CUDA" = true ]; then
        echo "Building without OpenMP"
        echo "ERROR: Unsupported build for this machine"
        exit
        # export DEVICE_ARCH="TURING75"
        # export KO_ROOT="${HOME_DIR}/kokkos/"
        # export KK_ROOT="${HOME_DIR}/kokkos-kernels/"
        # export YCPP_ROOT="${HOME_DIR}/yaml-cpp/install"
        # export AX_ROOT="${HOME_DIR}/ArborX/"
    elif [ "$USE_OPENMP" = true ] && [ "$USE_CUDA" = true ]; then
        echo "Building for OpenMP and CUDA"
        export DEVICE_ARCH="TURING75"
        export KO_ROOT="${HOME_DIR}/kokkos/install_cuda"
        export KK_ROOT="${HOME_DIR}/kokkos-kernels/install_cuda"
        export YCPP_ROOT="${HOME_DIR}/yaml-cpp/install"
        export AX_ROOT="${HOME_DIR}/ArborX/install_cuda"
        export SP_ROOT="${HOME_DIR}/spdlog/install_${BUILD_TYPE}"
    else
        echo "ERROR: Unsupported build for this machine"
        exit
    fi
else
    if [ "$IN_CONTAINER" = true ]; then
        export BUILD_TYPE="$DOCKER_BUILD_TYPE"
        export REAL_TYPE="$DOCKER_PRECISION"
        export SEARCH_TYPE="$DOCKER_SEARCH_TYPE"
        export HOME_DIR="/ext-docker-${BUILD_TYPE}"
        export LIBDIR=$MAC_LIBDIR
        if [ "$USE_OPENMP" = false ] && [ "$USE_CUDA" = false ]; then
            echo "Building for serial"
            echo "ERROR: Unsupported build for this machine"
            exit
            # export KO_ROOT="${HOME_DIR}/kokkos/"
            # export KK_ROOT="${HOME_DIR}/kokkos-kernels/"
            # export YCPP_ROOT="${HOME_DIR}/yaml-cpp/"
            # export AX_ROOT="${HOME_DIR}/ArborX/"
        elif [ "$USE_OPENMP" = true ] && [ "$USE_CUDA" = false ]; then
            echo "Building for OpenMP without CUDA"
            export KO_ROOT="${HOME_DIR}/kokkos/install"
            export KK_ROOT="${HOME_DIR}/kokkos-kernels/install"
            export YCPP_ROOT="${HOME_DIR}/yaml-cpp/install"
            export AX_ROOT="${HOME_DIR}/ArborX/install"
            export SP_ROOT="${HOME_DIR}/spdlog/install"
        elif [ "$USE_OPENMP" = false ] && [ "$USE_CUDA" = true ]; then
            echo "Building without OpenMP"
            echo "ERROR: Unsupported build for this machine"
            exit
            # export DEVICE_ARCH="TURING75"
            # export KO_ROOT="${HOME_DIR}/kokkos/"
            # export KK_ROOT="${HOME_DIR}/kokkos-kernels/"
            # export YCPP_ROOT="${HOME_DIR}/yaml-cpp/install"
            # export AX_ROOT="${HOME_DIR}/ArborX/"
        elif [ "$USE_OPENMP" = true ] && [ "$USE_CUDA" = true ]; then
            echo "Building for OpenMP and CUDA"
            echo "ERROR: Unsupported build for this machine"
            exit
            # export DEVICE_ARCH="TURING75"
            # export KO_ROOT="${HOME_DIR}/kokkos/install"
            # export KK_ROOT="${HOME_DIR}/kokkos-kernels/install"
            # export YCPP_ROOT="${HOME_DIR}/yaml-cpp/install"
            # export AX_ROOT="${HOME_DIR}/ArborX/install"
            # export SP_ROOT="${HOME_DIR}/spdlog/install"
        else
            echo "ERROR: Unsupported build for this machine"
            exit
        fi
    else
        echo "Unrecognized machine--attempting default build (this may not work)"
        export DEVICE_ARCH=$GPU_ARCHITECTURE
    fi
fi
echo "Home directory is ${HOME_DIR}"

# set these manually to where they are on your machine,
# or just comment out to build them as subprojects if you don't have them built
if [ -z "$KO_ROOT" ]; then
    echo "No local Kokkos provided"
else
    export KOKKOS_LIBDIR="${KO_ROOT}/${LIBDIR}"
    export KOKKOS_INCDIR="${KO_ROOT}/include"
    export KOKKOS_LIBRARY="libkokkoscore.a"
fi
if [ -z "$KK_ROOT" ]; then
    echo "No local Kokkos Kernels provided"
else
    export KOKKOSKERNELS_LIBDIR="${KK_ROOT}/${LIBDIR}"
    export KOKKOSKERNELS_INCDIR="${KK_ROOT}/include"
    export KOKKOSKERNELS_LIBRARY="libkokkoskernels.a"
fi
if [ -z "$YCPP_ROOT" ]; then
    echo "No local yaml-cpp provided"
else
    export YAML_CPP_LIBDIR="${YCPP_ROOT}/lib"
    export YAML_CPP_INCDIR="${YCPP_ROOT}/include"
    export YAML_CPP_LIBRARY="libyaml-cpp.a"
fi
if [ -z "$AX_ROOT" ]; then
    echo "No local ArborX provided"
else
    export ARBORX_INCDIR="${AX_ROOT}/include"
fi
if [ -z "$SP_ROOT" ]; then
    echo "No local spdlog provided"
else
    export SPDLOG_LIBDIR="${SP_ROOT}/${LIBDIR}"
    export SPDLOG_INCDIR="${SP_ROOT}/include"
    if [ "$BUILD_TYPE" == "debug" ]; then
        export SPDLOG_LIBRARY="libspdlogd.a"
        echo "given build type is ${BUILD_TYPE}--good luck."
    elif [ "$BUILD_TYPE" == "release" ]; then
        export SPDLOG_LIBRARY="libspdlog.a"
        echo "given build type is ${BUILD_TYPE}--we are not ready for this(?)."
    else
        echo "given build type is ${BUILD_TYPE}--why you gotta make it weird?"
        exit 1
    fi
fi

# as long as everything above looks good, you should be all set from here down
export OS=`uname -s`
if [ "$OS" == "Linux" ]
then
    if [ "$USE_CUDA" = true ]
    then
        if [ "$BUILD_KOKKOS" = true ]
        then
            export CXX=`pwd`/../ext/kokkos/bin/nvcc_wrapper
            echo $CXX
            export GPU_DEVICE_ARCH_FLAG="-DKokkos_ARCH_"$DEVICE_ARCH"=ON"
            echo $GPU_DEVICE_ARCH_FLAG
        else
            export CXX=${HOME_DIR}/kokkos/bin/nvcc_wrapper
        fi
    else
        export CXX=$LINUX_CPP
        export CC=$LINUX_CC
    fi
else
    if [ "$USE_OPENMP" = true ]
    then
        export CXX=$MAC_OMP_CPP
        export CC=$MAC_OMP_CC
    else
        export CXX=$MAC_SERIAL_CPP
        export CC=$MAC_SERIAL_CC
    fi
fi

rm -rf CMake*

# this configures the build with the following options:
# -D CMAKE_INSTALL_PREFIX="./"\
    # installs in build directory "./"
# -D CMAKE_VERBOSE_MAKEFILE=ON\
    # the build spits out a lot of info
# Below are the compiler and build type options from above
# -D CMAKE_CXX_COMPILER=$CXX\

# -D PARPT_USE_OPENMP=$USE_OPENMP\
# -D PARPT_USE_CUDA=$USE_CUDA
cmake .. \
    -D CMAKE_INSTALL_PREFIX=$INSTALL_LOCATION \
    -D PARPT_BUILD_TYPE=$BUILD_TYPE \
    -D CMAKE_VERBOSE_MAKEFILE=ON \
    -D CMAKE_CXX_COMPILER=$CXX \
    -D CMAKE_C_COMPILER=$CC \
    -D PARPT_USE_OPENMP=$USE_OPENMP \
    -D PARPT_USE_CUDA=$USE_CUDA \
    -D PARPT_PRECISION=$REAL_TYPE \
    -D PARPT_SEARCH_TYPE=$SEARCH_TYPE
