#include "ArborX_LinearBVH.hpp"
#include "Kokkos_Core.hpp"
#include "Kokkos_Sort.hpp"
#include "brute_force_crs_policy.hpp"
#include "containers.hpp"
#include "mass_transfer.hpp"
#include "spdlog/formatter.h"
#include "tree_crs_policy.hpp"
#include "type_defs.hpp"
#include "yaml-cpp/yaml.h"

using namespace particles;

namespace {

// Print usage information and exit.
void usage(const char* exe) {
  fmt::print(stderr, "ERROR: Too few inputs to {}--usage:\n", exe);
  fmt::print(stderr, "{} <input_pts.yml>\n", exe);
  exit(1);
}

}  // end namespace

int main(int argc, char* argv[]) {
  // // print usage if to few args are provided
  if (argc < 2) usage(argv[0]);

  // this is essentially kokkos initialization, but supposedly
  // takes care of scoping issues proactively
  ko::ScopeGuard guard(argc, argv);

  fmt::print("Kokkos execution space is: {}\n", typeid(ExecutionSpace).name());

  ko::Profiling::pushRegion("setup");
  // set the string giving the input file
  std::string infile = argv[1];

  // host parameters
  // note that radius is a real here and then recast as float for arborx, since
  // that is the way it will be for the actual code
  int xlen, dim;
  Real radius;

  // get the parameters from the input file
  ko::Profiling::pushRegion("read params from yaml");
  auto root = YAML::LoadFile(infile);
  xlen = root["N"].as<int>();
  dim = root["dim"].as<int>();
  radius = root["dist"].as<Real>();
  ko::Profiling::popRegion();

  // create the particle location views
  ko::Profiling::pushRegion("create X views and read from yaml");
  auto X = ko::View<Real**>("X", dim, xlen);
  ko::deep_copy(X, 0.0);
  auto hX = ko::create_mirror_view(X);
  ko::deep_copy(hX, 0.0);

  // for indexing the following
  std::vector<std::string> coords{"x", "y", "z"};

  // loop over dimension and particles to fill X view
  for (int j = 0; j < dim; ++j) {
    auto pts = root["pts"];
    auto node = pts[coords[j]];
    int i = 0;
    for (auto iter : node) {
      hX(j, i) = iter.as<Real>();
      ++i;
    }
  }
  ko::deep_copy(X, hX);
  ko::Profiling::popRegion();

  ko::Profiling::popRegion();
  ko::Profiling::pushRegion("***brute_force***");

  Params params;
  params.Np = xlen;
  params.cutdist = radius;
  params.pctRW = 0.5;
  params.denom = 6.384;
  params.dim = dim;
  ko::View<Real*> mass;
  auto mass_trans = MassTransfer<BruteForceCRSPolicy>(params, X, mass);
  mass_trans.Nc = params.Np;
  mass_trans.Np1 = params.Np;
  mass_trans.substart = 0;
  mass_trans.subend = params.Np;
  auto temp = mass_trans.build_sparse_transfer_mat();

  auto hcol = ko::create_mirror_view(mass_trans.spmat_views.col);
  auto hrow = ko::create_mirror_view(mass_trans.spmat_views.row);
  auto hrowmap = ko::create_mirror_view(mass_trans.spmat_views.rowmap);
  auto hval = ko::create_mirror_view(mass_trans.spmat_views.val);
  ko::deep_copy(hcol, mass_trans.spmat_views.col);
  ko::deep_copy(hrow, mass_trans.spmat_views.row);
  ko::deep_copy(hrowmap, mass_trans.spmat_views.rowmap);
  ko::deep_copy(hval, mass_trans.spmat_views.val);

  ko::Profiling::popRegion();
  ko::Profiling::pushRegion("***tree***");

  ko::Profiling::pushRegion("create MassTrans object");
  auto mass_trans_tree = MassTransfer<TreeCRSPolicy>(params, X, mass);
  ko::Profiling::popRegion();
  mass_trans_tree.Nc = params.Np;
  mass_trans_tree.Np1 = params.Np;
  mass_trans_tree.substart = 0;
  mass_trans_tree.subend = params.Np - 1;
  ko::Profiling::pushRegion("build sparsemats");
  auto temp2 = mass_trans_tree.build_sparse_transfer_mat();
  ko::Profiling::popRegion();

  auto hcol_tree = ko::create_mirror_view(mass_trans_tree.spmat_views.col);
  auto hrow_tree = ko::create_mirror_view(mass_trans_tree.spmat_views.row);
  auto hrowmap_tree =
      ko::create_mirror_view(mass_trans_tree.spmat_views.rowmap);
  auto hval_tree = ko::create_mirror_view(mass_trans_tree.spmat_views.val);

  // the make_pair below needs this info on host
  ko::deep_copy(hrowmap_tree, mass_trans_tree.spmat_views.rowmap);

  auto col_tree = mass_trans_tree.spmat_views.col;
  auto val_tree = mass_trans_tree.spmat_views.val;

  ko::Profiling::popRegion();
  ko::Profiling::pushRegion("***SORT***");

  // sort the tree-generated columns, so as to match the naturally ordered brute
  // force ones. then apply the same permutation to the val view
  ko::Profiling::pushRegion("***LOOP***");
  for (int i = 0; i < xlen; ++i) {
    // pair for indexing the current row in the sparse structure
    auto rpair = ko::make_pair(hrowmap_tree(i), hrowmap_tree(i + 1));
    ko::Profiling::pushRegion("sort-subview");
    // the subview of the column view that will be sorted in this loop iteration
    auto sort_view = ko::subview(col_tree, rpair);
    ko::Profiling::popRegion();
    ko::Profiling::pushRegion("permutation");
    // sort the current row of column entries and save the permutation view
    auto permutation =
        ArborX::Details::sortObjects(ExecutionSpace{}, sort_view);
    ko::Profiling::popRegion();
    ko::Profiling::pushRegion("val-subview");
    // subview of vals to be sorted
    auto val_sort_view = ko::subview(val_tree, rpair);
    ko::Profiling::popRegion();
    ko::Profiling::pushRegion("apply perm");
    // create a temp view since it appears the original and target view must be
    // different
    auto temp = ko::View<Real*>("permute_temp", val_sort_view.size());
    ko::deep_copy(temp, val_sort_view);
    // apply the permutation to the vals view
    ArborX::Details::applyPermutation(ExecutionSpace{}, permutation, temp,
                                      val_sort_view);
    ko::Profiling::popRegion();
  }
  ko::Profiling::popRegion();

  ko::Profiling::popRegion();
  ko::Profiling::pushRegion("***copy***");

  ko::deep_copy(hcol_tree, mass_trans_tree.spmat_views.col);
  ko::deep_copy(hrow_tree, mass_trans_tree.spmat_views.row);
  ko::deep_copy(hrowmap_tree, mass_trans_tree.spmat_views.rowmap);
  ko::deep_copy(hval_tree, mass_trans_tree.spmat_views.val);

  for (int i = 0; i < xlen + 1; ++i) {
    assert(hrowmap(i) - hrowmap_tree(i) == 0);
  }
  fmt::print(stdout, "Rowmaps match!\n");

  int nnz = hcol_tree.size();
  for (int i = 0; i < nnz; ++i) {
    assert(hrow(i) - hrow_tree(i) == 0);
  }
  fmt::print(stdout, "Rows match!\n");

  for (int i = 0; i < nnz; ++i) {
    assert(hcol(i) - hcol_tree(i) == 0);
  }
  fmt::print(stdout, "Columns match!\n");
  for (int i = 0; i < nnz; ++i) {
    assert(hval(i) - hval_tree(i) < 1.0e-13);
  }
  fmt::print(stdout, "Values match!\n");

  fmt::print(
      stdout,
      "SUCCESS: brute force and tree search produce identical results.\n");

  ko::Profiling::popRegion();
  return 0;
}
