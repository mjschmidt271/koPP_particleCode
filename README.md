# Kokkos-enabled, Performance-portable Particle Code
<!-- use this page when I switch to mkdocs -->
<!-- https://majianglin2003.medium.com/how-to-use-markdown-and-mkdocs-to-write-and-organize-technical-notes-9aad3f3b9c82 -->

This repository contains random walk particle tracking code with SPH-style mass transfers, written in C++ and designed for parallel performance portability, using [Kokkos](https://github.com/kokkos/kokkos).

The only current dependencies are **cmake** and __python3__ (with some package dependencies, including __numpy__ and __matplotlib__, if you want to use some of the plotting utilities). On Mac, your easiest option is to use [Homebrew](https://docs.brew.sh/Installation) and then `brew install cmake`, `brew install python`, `pip install numpy`, and `pip install matplotlib`. Note that if python scripts aren't running properly, you may need to use `pip3` in the above. If you're looking to run unit tests that involve python scripts, then you'll likely have to install more python packages until your code stops crashing. Also, some of the tests are verified using __Jupyter Notebooks__, which can be taken care of via [Homebrew](https://docs.brew.sh/Installation) via `brew install jupyter`, `brew install notebook`, and then run the notebook with `jupyter-notebook <notebook>.ipynb` which will open a browser window.

<ins>**Note:**</ins> All of the above dependencies can be avoided by using the [Docker](https://docs.docker.com/get-docker/) build instructions below. However, first you must install Docker :upside_down_face: (`brew install docker`).

Note that simply downloading the repository will not include the third-party libraries (Kokkos, Kokkos Kernels, yaml-cpp, ArborX), as they are git submodules.
For that reason, the best bet is to clone the repository.
For example, in whichever directory you want to put the code:

## Build Instructions

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
    1. `./run.sh` runs a simple example or `make test` to run the verification/unit tests.
        - The example launched by `./run.sh` can be modified by editing the input file `src/data/particleParams.yaml`.
            - Note that if you edit the one in the build directory, it will be overwritten by the original after a `make install`. Similarly, if you edit the one in the top-level `src` directory, `make install` will be required before you can run the new simulation.
        - Some different examples can be found in the `tests` directory, containing `MT_only`, `RWMT`, and `RW_only` (MT = mass-transfer, RW = random walk).

    ### Some Notes

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

- To build from source (and run):
    1. Edit the `config.sh` script to describe your build goals and development environment.
    1. `mkdir build && cd build`
    1. `../config.sh`
        - If you're savvy and want to pre-install the libraries that are commented in `config.sh` to cut down on build times, follow the given templates to specify their paths.
    1. `make -j install`
        - Note: similarly to above, the `-j` flag executes the make using the max number of cores available.
        - Note: the current behavior, as given in `config.sh` is to install to the build directory. As such, you can re-build/install/run from the same place without changing directories repeatedly.
            - If you want to customize this behavior, you can change the line toward the bottom that reads `CMAKE_INSTALL_PREFIX="./"` to indicate a directory other than `build` (e.g., `./` = this directory).
    1. `./run.sh` to run a basic example.
        - The example can be modified by editing the input file `src/data/particleParams.yaml`.
            - Note that if you edit the one in the build directory, it will be overwritten by the original after a `make install`. Similarly, if you edit the one in the top-level `src` directory, `make install` will be required before you can run the new simulation.

## Building for CPU (OpenMP)

Building with OpenMP does work on my personal Mac running Mojave with `gcc v12.0.0` and `libomp v11.0.0`, as well as on a couple of Linux workstations, using various gcc compilers.
<!-- "brew info libomp" will give the openmp version if it was installed via Homebrew -->
This is achieved by uncommenting the relevant lines in `config.sh` so that `export USE_OPENMP=True` and ensuring the proper compiler is pointed to with the environment variables below.
If you are on an Apple machine, it is recommended to use the __gcc__ compilers with compatible OpenMP. The easiest way to achieve this is with Homebrew, by running
```
brew install gcc
brew install libomp
```
However, OpenMP functionality can be finicky on Mac, so reach out if you have issues.

## Building for GPU (CUDA)

In principle, this should be as easy as uncommenting the relevant lines in `config.sh` so that `export USE_CUDA=True` and providing the GPU architecture in the compilers section of the config script (see the ___Architecture Keywords___ section of the [Kokkos Wiki's build guide](https://github.com/kokkos/kokkos/wiki/Compiling) for more info)
However, everything gets harder once GPUs are involved, so this may not be for the faint of heart and could involve some pain if you're using a machine I haven't already tested on.
If you have a non-Nvidia GPU, I have no idea, but... maybe it'll work. ¯\\\_(ツ)\_/¯

Otherwise, feel free to reach out or file an issue if you run into problems!
