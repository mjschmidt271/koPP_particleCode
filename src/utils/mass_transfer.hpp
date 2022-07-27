#ifndef MASS_TRANSFER_HPP
#define MASS_TRANSFER_HPP

#include "KokkosSparse_spadd.hpp"
#include "KokkosSparse_spmv.hpp"
#include "Kokkos_Core.hpp"
#include "containers.hpp"
#include "type_defs.hpp"

namespace particles {

template <typename CRSViewPolicy>
class MassTransfer {
 public:
  // copy of the Particles' Params
  Params params;
  ko::View<Real**> X;
  ko::View<Real*> mass;
  // temporary mass view, used for computing the spmv
  ko::View<Real*> tmpmass;
  // number of particles in a chunk
  // NOTE: unless Nc evenly divides Np, Nc = Np / n_substeps + 1, since the
  // remainders are distributed over the first Np1 chunks, and the remaining
  // chunks are of size Nc - 1
  int Nc;
  // the number of chunks that get an extra particle from the remainder
  int Np1;
  // beginning and end of the subsets of particles, when doing the chunk method
  int substart, subend;
  // the views containing the data to build a kokkos CRS sparse matrix
  SparseMatViews spmat_views;
  // the actual CRS sparse transfer matrix
  SpmatType transfer_mat;
  MassTransfer<CRSViewPolicy>() = default;
  MassTransfer<CRSViewPolicy>(const Params& params, const ko::View<Real**>& X,
                              ko::View<Real*>& mass);
  void transfer_mass();
  SpmatType build_sparse_transfer_mat();
  SparseMatViews get_crs_views(int& nnz);
};

}  // namespace particles

#endif
