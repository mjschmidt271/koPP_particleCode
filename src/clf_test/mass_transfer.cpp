#include "mass_transfer.hpp"

namespace particles {

template <typename CRSViewPolicy>
MassTransfer<CRSViewPolicy>::MassTransfer(const Params& _params,
                                          const ko::View<Real**>& _X,
                                          ko::View<Real*>& _mass)
    : params(_params) {
  X = _X;
  mass = _mass;
}

template <typename CRSViewPolicy>
void MassTransfer<CRSViewPolicy>::transfer_mass() {
  if (params.pctRW < 1.0) {
    ko::Profiling::pushRegion("begin substep_fxn");
    // number of particles to be considered in a substep and the remainder
    // FIXME: this should all be done much earlier
    Nc = params.Np / params.n_chunks;
    // NOTE: the number of initial chunks that get +1 particle is the remainder
    // of the integer division
    Np1 = params.Np % params.n_chunks;
    // make a temporary copy of the mass and positions views, as we will be
    // filling in the updated mass values into the original mass view, while
    // keeping tmpmass static, and the subX view just makes life easier
    tmpmass = ko::View<Real*>("tmpmass", params.Np);
    ko::deep_copy(tmpmass, mass);
    substart = 0;
    subend = Nc - 1;
    // if the remainder is > 0, add one to the initial chunks
    if (Np1 > 0) {
      // this chunk has 1 extra particle, until we've done transfers for
      // Np1 chunks
      subend += 1;
      Nc += 1;
    }
    // NOTE: we should never enter this loop in the case that n_chunks evenly
    // divides Np, though if things crash here, maybe put some logic in
    for (int i = 0; i < Np1; ++i) {
      ko::Profiling::pushRegion("transfer subloop 1");
      transfer_mat = build_sparse_transfer_mat();
      // get a subview of mass, so we only update the particles under
      // consideration
      auto submass = ko::subview(mass, ko::make_pair(substart, subend + 1));
      KokkosSparse::spmv("N", 1.0, transfer_mat, tmpmass, 0.0, submass);
      substart += Nc;
      subend += Nc;
      ko::Profiling::popRegion();
    }
    // subtract 1 from Nc for the remaining chunks
    if (Np1 > 0) {
      Nc -= 1;
      subend -= 1;
    }
    for (int i = Np1 + 1; i <= params.n_chunks; ++i) {
      ko::Profiling::pushRegion("transfer subloop 2");
      transfer_mat = build_sparse_transfer_mat();
      // get a subview of mass, so we only update the particles under
      // consideration
      auto submass = ko::subview(mass, ko::make_pair(substart, subend + 1));
      // update only a portion of mass, keeping tmpmass static
      KokkosSparse::spmv("N", 1.0, transfer_mat, tmpmass, 0.0, submass);
      substart += Nc;
      subend += Nc;
      ko::Profiling::popRegion();
    }
    ko::Profiling::popRegion();
  }
}

template <typename CRSViewPolicy>
SpmatType MassTransfer<CRSViewPolicy>::build_sparse_transfer_mat() {
  // number of nonzero entries in the sparse distance matrix
  int nnz = 0;
  // local copies of external variables
  // NOTE: Nc is the size of the current chunk of particles
  auto lNc = Nc;
  auto lNp = params.Np;
  auto rowsum = ko::View<Real*>("rowsum", lNc);
  spmat_views = get_crs_views(nnz);
  // local shallow copy for use in parallel kernels
  auto lspmat = spmat_views;
  // construct the original sparse kernel matrix
  SpmatType kmat("sparse_transfer_mat", lNc, lNp, nnz, spmat_views.val,
                 spmat_views.rowmap, spmat_views.col);
  ko::deep_copy(rowsum, 0.0);
  ko::parallel_for(
      "compute rowsum", lNc, KOKKOS_LAMBDA(const int& i) {
        auto rowi = kmat.row(i);
        for (int j = 0; j < rowi.length; ++j) {
          rowsum(i) += rowi.value(j);
        }
      });
  // NOTE: this normalization assumes the matrix will always be symmetric
  // we may have to change things if that fails to be true
  ko::parallel_for(
      "normalize_mat", nnz, KOKKOS_LAMBDA(const int& i) {
        lspmat.val(i) = lspmat.val(i) / rowsum(lspmat.row(i));
      });
  kmat = SpmatType("sparse_transfer_mat", lNc, lNp, nnz, spmat_views.val,
                   spmat_views.rowmap, spmat_views.col);
  ko::deep_copy(rowsum, 0.0);
  ko::parallel_for(
      "compute rowsum", lNc, KOKKOS_LAMBDA(const int& i) {
        auto rowi = kmat.row(i);
        for (int j = 0; j < rowi.length; ++j) {
          rowsum(i) += rowi.value(j);
        }
      });
  // ultimately, kmat = W_N + I - diag(rowsum(W_N)), where W_N is the kernel
  // matrix, normalized by the row and column sums, as above
  // this is I - diag(rowsum(W_N))
  auto Iminusrow_val = ko::View<Real*>("Iminusrow_val", lNc);
  auto Iminusrow_rowmap = ko::View<int*>("Iminusrow_rowmap", lNc + 1);
  auto Iminusrow_col = ko::View<int*>("Iminusrow_col", lNc);
  ko::parallel_for(
      "subtract I - rowsum, and create rowmap/col", lNc,
      KOKKOS_LAMBDA(const int& i) {
        Iminusrow_val(i) = 1.0 - rowsum(i);
        Iminusrow_rowmap(i) = i;
        Iminusrow_col(i) = substart + i;
      });
  Iminusrow_rowmap(lNc) = lNc;
  SpmatType Iminusrow =
      SpmatType("sparse_Iminusrow_mat", lNc, lNp, lNc, Iminusrow_val,
                Iminusrow_rowmap, Iminusrow_col);
  // Create KokkosKernelHandle for spadd()... so many steps
  KernelHandle kh;
  kh.create_spadd_handle(false);
  SpmatType finalkmat = SpmatType();
  KokkosSparse::spadd_symbolic(&kh, kmat, Iminusrow, finalkmat);
  KokkosSparse::spadd_numeric(&kh, 1.0, kmat, 1.0, Iminusrow, finalkmat);
  kh.destroy_spadd_handle();
  return finalkmat;
}

template <typename CRSViewPolicy>
SparseMatViews MassTransfer<CRSViewPolicy>::get_crs_views(int& nnz) {
  auto lspmat_views =
      CRSViewPolicy::get_views(X, params, nnz, Nc, substart, subend);
  return lspmat_views;
}

}  // namespace particles

#include "brute_force_crs_policy.hpp"
template class particles::MassTransfer<particles::BruteForceCRSPolicy>;

#include "tree_crs_policy.hpp"
template class particles::MassTransfer<particles::TreeCRSPolicy>;
