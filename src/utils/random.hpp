#ifndef RANDOM_HPP
#define RANDOM_HPP

#include "Kokkos_Core.hpp"
#include "Kokkos_Random.hpp"
#include "type_defs.hpp"

namespace particles {

// functor for generating uniformly-distributed random Reals
// in the range [start, end]
// Note: RandPoolType is currently hard-coded in the particle class
// GeneratorPool type
template <class RandPool>
struct RandomUniform {
  // Output View for the random numbers
  ko::View<Real**> vals;

  // The GeneratorPool
  RandPool rand_pool;

  typedef Real Scalar;
  typedef typename RandPool::generator_type gen_type;

  Scalar start, end;

  KOKKOS_INLINE_FUNCTION
  // TODO: determine which way this loop should go
  // *not that important right now, as it's only used in initialization
  void operator()(int j, int i) const {
    // Get a random number state from the pool for the active thread
    gen_type rgen = rand_pool.get_state();

    // size_t dim = vals.extent(0);
      // draw random normal numbers, with mean and variance provided
      vals(j, i) = rgen.drand(start, end);

    // Give the state back, which will allow another thread to acquire it
    rand_pool.free_state(rgen);
  }

  // Constructor, initialize all members
  RandomUniform(ko::View<Real**> vals_, const RandPool& rand_pool_,
                const Scalar& start_, const Scalar& end_)
      : vals(vals_), rand_pool(rand_pool_), start(start_), end(end_) {}

};  // end RandomUniform functor

}  // namespace particles

#endif
