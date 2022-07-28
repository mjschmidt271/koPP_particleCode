#ifndef PARTICLES_HPP
#define PARTICLES_HPP

#include "diffusion.hpp"
#include "Kokkos_Core.hpp"
#include "mass_transfer.hpp"
#include "parPT_io.hpp"
#include "random_walk.hpp"

namespace particles {

// class for particles and associated methods
class Particles {
 public:
  // real-valued position
  ko::View<Real**> X;
  // FIXME: keep a mirror (private?) of these for writing out every xx time
  // steps mass carried by particles (FIXME: units, )
  ko::View<Real*> mass;
  // host version of params
  // FIXME(?): create a device version?
  Params params;
  // diffusion object contains random-walk and mass-transfer
  Diffusion diffusion;
  ParticleIO particleIO;
  // constructor that gets the random number seed from the input yaml, clock time
  // or uses a standard one that leads to identical, deterministic output
  Particles(const std::string& input_file);
  void initialize_positions();
  void initialize_masses();
};

}  // namespace particles

#endif
