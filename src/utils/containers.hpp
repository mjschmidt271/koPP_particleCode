#ifndef CONTAINERS_HPP
#define CONTAINERS_HPP

#include <chrono>

#include "Kokkos_Core.hpp"
#include "spdlog/formatter.h"
#include "type_defs.hpp"
#include "yaml-cpp/yaml.h"

namespace particles {

// NOTE: this may be obsolete, now that we read from file
// initial condition enums, one for positions of particles, the other for mass
// carried by particles
enum IC_enum_space { point_loc, uniform, equi, hat };
enum IC_enum_mass { point_mass, heaviside, gaussian };

// enum specifying how the random-number generator seed is set
enum rand_seed_enum { clock_rand, specified_rand, default_rand, missing };

// class holding simulation parameters
class Params {
 public:
  // total number of particles
  int Np;
  // number of spatial dimensions
  int dim;
  // number of chunks of the particle array that we serialize over for
  // substepping
  int n_chunks;
  // number of time steps
  int nSteps;
  Real X0_mass, X0_space, maxT, dt, D, pctRW, denom, cdist_coeff, cutdist,
      hat_pct;
  std::string IC_str_space, IC_str_mass;
  std::vector<Real> omega;
  std::string pFile;
  IC_enum_space IC_type_space;
  IC_enum_mass IC_type_mass;
  rand_seed_enum seed_type;
  uint64_t seed_val;
  bool write_plot;
  Params() = default;
  Params(const std::string& yaml_name);
  void enumerate_IC(std::string& IC_str, const YAML::Node& yml,
                    const bool& space);
  void enumerate_seed_type(std::string& seed_str, const YAML::Node& yml);
};

struct SparseMatViews {
  ko::View<int*> row;
  ko::View<int*> col;
  ko::View<Real*> val;
  ko::View<int*> rowmap;
};

}  // namespace particles

#endif
