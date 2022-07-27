#include "random_walk.hpp"

namespace particles {

// parallelized random walk function that calls the RandomWalk functor
void random_walk(ko::View<Real**>& X, const Params& params,
                 RandPoolType& rand_pool) {
  ko::Profiling::pushRegion("RW_fxn_pre-if");
  ko::Profiling::pushRegion("RW_fxn");
  ko::parallel_for(
      "RW_kernel",
      ko::MDRangePolicy<ko::Rank<2>>({0, 0}, {params.dim, params.Np}),
      RandomWalk<RandPoolType>(
          X, rand_pool, 0.0,
          sqrt(2.0 * params.pctRW * params.D * params.dt)));
  ko::Profiling::popRegion();
  ko::Profiling::popRegion();
}

}  // namespace particles
