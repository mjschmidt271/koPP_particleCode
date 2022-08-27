#include <iostream>
#include <vector>

#include "particles.hpp"

using namespace particles;

namespace {

// Print usage information and exit.
void usage(const char* exe) {
  fmt::print(stderr, "ERROR: Too few inputs to {}--usage:\n", exe);
  fmt::print(stderr, "{} <input.yml>\n", exe);
  exit(1);
}
}  // namespace

int main(int argc, char* argv[]) {
  // make sure the input file is specified, or print usage and exit
  if (argc < 2) usage(argv[0]);

  // this is essentially kokkos initialization, but supposedly
  // takes care of scoping issues proactively
  ko::ScopeGuard guard(argc, argv);

  {  // Kokkos scope
    fmt::print("Kokkos execution space is: {}\n",
               typeid(ExecutionSpace).name());
    ko::print_configuration(std::cout, true);

    // get the input file name from command line argument
    std::string input_file(argv[1]);
    // create the particles object
    ko::Profiling::pushRegion("particle constructor");
    Particles parts(input_file);
    ko::Profiling::popRegion();

    int tStep = 0;

    ko::Profiling::pushRegion("write initial positions");
    // write initial positions/masses to file
    parts.particleIO.write(parts.params, parts.X, parts.mass, tStep);
    ko::Profiling::popRegion();

    ko::Profiling::pushRegion("timestepping");
    // begin time stepping
    for (int tStep = 1; tStep <= parts.params.nSteps; ++tStep) {
      ko::Profiling::pushRegion("tstep loop diffuse");
      parts.diffusion.diffuse();
      ko::Profiling::popRegion();
      ko::Profiling::pushRegion("tstep_write");
      // write updated particle info to file
      if (parts.params.write_plot) {
        parts.particleIO.write(parts.params, parts.X, parts.mass, tStep);
      }
      ko::Profiling::popRegion();
    }
    // still write final condition, even if not writing time steps
    if (not parts.params.write_plot) {
      parts.particleIO.write(parts.params, parts.X, parts.mass, tStep);
    }
    ko::Profiling::popRegion();

  }  // end Kokkos scope

  return 0;
}
