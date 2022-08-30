#include "particles.hpp"

namespace particles {

Particles::Particles(const std::string& _input_file) {
  ko::Profiling::pushRegion("construct particleIO");
  const std::string yaml_name = _input_file;
  particleIO = ParticleIO(params, yaml_name);
  ko::Profiling::popRegion();
  ko::Profiling::pushRegion("constructor print");
  particleIO.print_params_summary(params);
  ko::Profiling::popRegion();
  ko::Profiling::pushRegion("ctor initialize position");
  // initialize the X view
  // FIXME: this is layoutleft
  X = ko::View<Real**>("X", params.dim, params.Np);
  particleIO.set_positions(params, yaml_name, X);
  // initialize the mass view
  ko::Profiling::popRegion();
  ko::Profiling::pushRegion("ctor initialize mass");
  mass = ko::View<Real*>("mass", params.Np);
  initialize_masses();
  ko::Profiling::popRegion();
  ko::Profiling::pushRegion("ctor initialize diffusion");
  diffusion = Diffusion(X, mass, params);
  ko::Profiling::popRegion();
}

// FIXME: this should be passed in
void Particles::initialize_masses() {
  switch (params.IC_type_mass) {
    case point_mass: {
      auto hmass = ko::create_mirror_view(mass);
      int mid = (params.Np + 2 - 1) / 2 - 1;
      for (int i = 0; i < params.Np; ++i) {
        hmass(i) = 0.0;
      }
      hmass(mid) = 1.0;
      ko::deep_copy(mass, hmass);
    }
    case heaviside: {
      break;
    }
    case gaussian: {
      break;
    }
  }
}

}  // namespace particles
