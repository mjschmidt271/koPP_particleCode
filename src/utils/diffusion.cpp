#include "diffusion.hpp"

namespace particles {

Diffusion::Diffusion(ko::View<Real**>& _X,
                     ko::View<Real*>& _mass, const Params& _params)
    : X(_X), mass(_mass), params(_params) {
  rand_pool = RandPoolType(params.seed_val);
  mass_trans = MassTransfer<CRSViewPolicy>(params, X, mass);
}

void Diffusion::diffuse() {
  if (params.pctRW > 0.0) {
    particles::random_walk(X, params, rand_pool);
  }
  if (params.pctRW < 1.0) {
    mass_trans.transfer_mass();
  }
}

}  // namespace particles
