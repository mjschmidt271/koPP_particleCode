# Kokkos-enabled, Performance-portable Particle Code
<!-- use this page when I switch to mkdocs -->
<!-- https://majianglin2003.medium.com/how-to-use-markdown-and-mkdocs-to-write-and-organize-technical-notes-9aad3f3b9c82 -->

This repository contains random walk particle tracking code with SPH-style mass transfers, written in C++ and designed for parallel performance portability, using [Kokkos](https://github.com/kokkos/kokkos).

The only mandatory dependency is **cmake**; however, __python3__ is needed for testing/plotting/verification capability, along with some package dependencies, including __numpy__, __matplotlib__, and a few others.
On Mac, your easiest option is to use [Homebrew](https://docs.brew.sh/Installation) and then
- `brew install cmake`
- `brew install python`
- `pip install numpy`
- `pip install matplotlib`
- etc.

Note that if python codes still don't work after package installation, you may need to use `pip3` in the above.
If you're looking to run unit/verification tests that involve python scripts, then you'll likely have to install more python packages until your code stops crashing.
Also, there are a handful of __Jupyter Notebooks__ that were used for creating the verification tests and may be helpful for visualization or verification of future changes.
This dependency can be taken care of via [Homebrew](https://docs.brew.sh/Installation) using
- `brew install jupyter`
- `brew install notebook`

Then run the notebook with `jupyter-notebook <notebook>.ipynb`, which will open a browser window.

<ins>**Important Notes:**</ins>
1. All of the above dependencies can be avoided by using the [Docker](https://docs.docker.com/get-docker/) build instructions below. However, first you must install Docker :upside_down_face: (`brew install docker`).
1. Simply downloading the zip file from the repository will not include the third-party libraries (Kokkos, Kokkos Kernels, yaml-cpp, ArborX), as they are git submodules.
For that reason, the best bet is to clone the repository, as below in the build instructions.

## Docker Build Instructions
1. Clone the repository:
    - If you use https (this is the case if you haven't set up a [github ssh key](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh)):
        - `git clone --recurse-submodules -j8 https://github.com/mschmidt271/koPP_particleCode.git`
    - If you use ssh:
        - `git clone --recurse-submodules -j8 git@github.com:mschmidt271/koPP_particleCode.git`
    - Note: the `-j8` is a parallel flag, allowing git to fetch up to 8 submodules in parallel.
1. `cd koPP_particleCode`

- To build and run using Docker (recommended for ease):
    1. Start docker.
        - Trust me--you'll forget it eventually and lose 15 minutes of your life debugging :)
    1. `docker build -t ko_pt .`
        - Default build is `debug` for now. However, if you want to explicitly choose the build mode, use:
            - `docker build --build-arg BUILD_TYPE_IN=<debug, release> -t ko_pt .`
    1. `docker run -it --rm ko_pt bash`
        - This will enter a bash terminal in a Docker container running Ubuntu.
            - To exit, type: `exit` or <kbd>ctrl</kbd> + <kbd>d</kbd>.

    ### Some Notes on the Docker Build

    1. If you wish to run Jupyter notebooks from within the container, run the container using:
        - `docker run -it -p 8888:8888 ko_pt bash`
            - Note that the `-p <host-port>:<container-port>` flag maps a port from inside the container to one on your host machine (well, technically publishes internal ports), in this case 8888 to 8888.
        - Once you've generated the results, run the Jupyter notebook using:
            1. If you chose the 8888 port mapping above, you can use a macro that I've defined to run the notebook:
                - `djupyter <notebook-name>.ipynb`
            1. Otherwise:
                - `jupyter notebook --ip 0.0.0.0 --no-browser --allow-root <notebook-name>.ipynb`
        - Now, in your machine's web browser, either,
            - Copy and paste one of the URLs containing a token into the browser (the final one works for me most reliably).
            - Go to `localhost:<host-port>`, where `host-port` is 8888 if you're using my macro. You will be prompted for a password or token that is given in the container's terminal at the end of one of the above-referenced URLs.
        - You will be presented with a directory structure in the browser where you can select the desired notebook and run it.
    1. If you make changes to the source code from within the container, they will not transmit to the actual source code nor persist after exiting and restarting the container. If you want to make persistent changes, I would recommend:
        - Modify/test within the container.
        - Once you've got things working, modify the external source code.
        - Rebuild and restart the container, which will now contain these changes (`docker build -t ko_pt . && docker run -it --rm ko_pt bash`).

## Building from Source
1. Clone the repository, as given [above](#docker-build-instructions).
1. Edit the `config.sh` script to accommodate  your build goals and development environment.
1. `mkdir build && cd build`
1. Run the config script
    - `../config.sh`
    - _**Advanced Maneuver:**_ If you know what you're doing and want to cut down on build times by pre-installing the libraries that are referenced in the **"machine-dependent build options"** section of `config.sh`, add a logic block for your machine to that section, with the full knowledge that you are on from here :construction_worker:.
1. Run the `cmake`-generated makefiles and install the project.
    - `make -j install`
    - **Note:** similarly to above, the `-j` flag executes the `make` using the max number of cores available, and `make -jN` will use `N` cores.
    - **Note:** the current behavior, as given in `config.sh` is to install to the directory in which the config script is run (probably `build`).
    As such, you can re-build/install/run from the same place without changing directories repeatedly.
        - If you want to customize this behavior, you can change the line near the top of `config.sh` that reads `export INSTALL_LOCATION="./"` to indicate a directory other than `build` (`./` = this directory where we run `config.sh`).

## Testing and Running
1. Run the unit/verification tests to ensure the code is running properly.
    - `make test`
    - If tests fail, check out `<install location>/Testing/Temporary/LastTest.log` to get some info, and if it's not a straightforward issue, like a missing python package, then reach out or file a GitHub issue.
1. To run a basic example, execute `./run.sh` from the `install (build)` directory.
    - The example can be modified by editing the input file `src/data/particleParams.yaml`.
        - Note that if you edit the input file or run script in the build directory, it will be overwritten by the original (in the source tree--e.g. `koPP_particleCode/src`) after doing another `make install`.
        - Some additional examples can be found in the `tests` directory, containing `MT_only`, `RWMT`, and `RW_only` (MT = mass-transfer, RW = random walk).

## Building for OpenMP (CPU)
- Building with OpenMP does work on my personal Mac running Monterey (earlier versions also worked on Mojave) with `g++` versions 12 and 13 and `libomp` versions 11 and 14, as well as on a couple of Linux workstations, using various versions of the `g++` compiler.
    - **Note:**  If OpenMP was installed via Homebrew `brew info libomp` will give the version.
- In order to build for OpenMP, (un)comment the relevant lines in `config.sh` so that `export USE_OPENMP=True` and ensure the proper compiler variable is set in the **"compiler options"** section, namely `MAC_OMP_CPP`.
    - In the current iteration of the config file, this is option #2 in the **"parallel accelerator options"** section.
- If you are on an Apple machine, it is recommended to use the __g++__ compiler from the GNU compiler collection (GCC) and compatible OpenMP (Apple's default `clang++` doesn't have OpenMP).
The easiest way to achieve this is on Mac is with Homebrew, by running
    - `brew install gcc`
    - `brew install libomp`
- Finally, know that OpenMP functionality can be finicky on Mac, so reach out if you have issues.

## Building for CUDA (GPU)
- In principle, this should be as easy as (un)commenting the relevant lines in the **"parallel accelerator options"** section of `config.sh` so that `export USE_CUDA=True` (option #4) and setting the `GPU_ARCHITECTURE` variable (see the ___Architecture Keywords___ section of the [Kokkos Wiki's build guide](https://github.com/kokkos/kokkos/wiki/Compiling) for more info)
- However, everything gets harder once GPUs are involved, so this may not be for the faint of heart and could involve some pain if you're using a machine I haven't already tested on.
- If you have a non-NVIDIA GPU, I have no idea, but... maybe it'll work. ¯\\\_(ツ)\_/¯

## Issues?
Please feel free to reach out or file an issue if you run into problems!
