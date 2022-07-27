#ifndef TYPE_DEFS_HPP
#define TYPE_DEFS_HPP

#include "KokkosKernels_default_types.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_spadd.hpp"
#include "Kokkos_Random.hpp"
#include "parPT_config.hpp"

namespace particles {

// alias the kokkos namespace to make life easier
namespace ko = Kokkos;

// Note: there's also a 1024-bit generator, but that is probably overkill
using RandPoolType = typename ko::Random_XorShift64_Pool<>;

// some other kokkos aliases to keep things consistent/easier
using Scalar = default_scalar;
using Ordinal = default_lno_t;
using Offset = default_size_type;
using DeviceType =
    typename ko::Device<ko::DefaultExecutionSpace,
                        typename ko::DefaultExecutionSpace::memory_space>;
using ExecutionSpace = typename DeviceType::execution_space;
using MemorySpace = typename DeviceType::memory_space;
using SpmatType =
    typename KokkosSparse::CrsMatrix<Scalar, Ordinal, DeviceType, void, Offset>;
using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<
    Offset, Ordinal, Scalar, ExecutionSpace, MemorySpace, MemorySpace>;

}  // namespace particles

#endif
