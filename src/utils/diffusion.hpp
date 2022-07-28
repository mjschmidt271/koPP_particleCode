#ifndef DIFFUSION_HPP
#define DIFFUSION_HPP

#include "ArborX_LinearBVH.hpp"
#include "Kokkos_Core.hpp"
#include "Kokkos_Random.hpp"
#include "containers.hpp"
#include "mass_transfer.hpp"
#include "parPT_io.hpp"
#include "random_walk.hpp"

#include "brute_force_crs_policy.hpp"
#include "tree_crs_policy.hpp"

namespace particles {

// this comes from the parPT_config.hpp via type_defs.hpp (which is included in
// a few lower-level headers)
using CRSViewPolicy = crs_policy_name;

class Diffusion {
 public:
  // real-valued position
  ko::View<Real**> X;
  // FIXME: keep a mirror (private?) of these for writing out every xx time
  // steps mass carried by particles (FIXME: units, )
  ko::View<Real*> mass;
  // host version of params
  // FIXME(?): create a device version?
  Params params;
  MassTransfer<CRSViewPolicy> mass_trans;
  RandPoolType rand_pool;
  Diffusion() = default;
  Diffusion(ko::View<Real**>& X, ko::View<Real*>& mass, const Params& params);
  void diffuse();
};

}  // namespace particles

#endif
