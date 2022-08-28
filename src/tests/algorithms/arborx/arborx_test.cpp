#include "ArborX_LinearBVH.hpp"
#include "Kokkos_Core.hpp"
#include "spdlog/formatter.h"
#include "type_defs.hpp"
#include "yaml-cpp/yaml.h"

using namespace particles;

// these are some structs that are necessary for the ArborX fixed-radius search
// this was provided by the helpful ArborX/Kokkos developer dalg24
template <class MemorySpace>
struct Spheres {
  Kokkos::View<float**, MemorySpace> points;
  float radius;
};
template <class MemorySpace>
struct ArborX::AccessTraits<Spheres<MemorySpace>, ArborX::PredicatesTag> {
  using memory_space = MemorySpace;
  using size_type = std::size_t;
  static KOKKOS_FUNCTION size_type size(Spheres<MemorySpace> const& x) {
    return x.points.extent(0);
  }
  static KOKKOS_FUNCTION auto get(Spheres<MemorySpace> const& x, size_type i) {
    return ArborX::attach(
        ArborX::intersects(ArborX::Sphere{
            ArborX::Point{x.points(i, 0), x.points(i, 1), x.points(i, 2)},
            x.radius}),
        (int)i);
  }
};

namespace {

// Print usage information and exit.
void usage(const char* exe) {
  fmt::print(stderr, "ERROR: Too few inputs to {}--usage:\n", exe);
  fmt::print(stderr, "{} <input_pts.yml> <outfile.txt>\n", exe);
  exit(1);
}

}  // end namespace

int main(int argc, char* argv[]) {
  // print usage if to few args are provided
  if (argc < 3) usage(argv[0]);

  // this is essentially kokkos initialization, but supposedly
  // takes care of scoping issues proactively
  ko::ScopeGuard guard(argc, argv);

  fmt::print("Kokkos execution space is: {}\n", typeid(ExecutionSpace).name());

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

  // NOTE: These X (fX) views are the transpose of how they are in the actual
  // code

  // create the particle location views
  ko::Profiling::pushRegion("create X views and read from yaml");
  auto X = ko::View<Real**>("X", xlen, dim);
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
      hX(i, j) = iter.as<Real>();
      ++i;
    }
  }
  ko::deep_copy(X, hX);
  ko::Profiling::popRegion();

  // make a float version of radius and X, as required by arborx, being careful
  // because the X view passed to arborx must have dimension 3
  ko::Profiling::pushRegion("create device/float views");
  auto fX = ko::View<float* [3], MemorySpace>("float X", xlen);
  ko::deep_copy(fX, 0.0);
  ko::deep_copy(ko::subview(fX, ko::ALL(), ko::make_pair(0, dim)),
                ko::subview(X, ko::ALL(), ko::make_pair(0, dim)));
  float frad = float(radius);
  ko::Profiling::popRegion();

  // create the bounding value hierarchy and query it for the fixed-radius
  // search
  ko::Profiling::pushRegion("create BVH");
  ArborX::BVH<MemorySpace> bvh{ExecutionSpace(), fX};
  ko::Profiling::popRegion();
  ko::Profiling::pushRegion("create offset/indices views");
  ko::View<int*, MemorySpace> offsets("offsets", 0);
  ko::View<int*, MemorySpace> indices("indices", 0);
  ko::Profiling::popRegion();
  ko::Profiling::pushRegion("query BVH");
  ArborX::query(bvh, ExecutionSpace{}, Spheres<MemorySpace>{fX, frad}, indices,
                offsets);
  ko::Profiling::popRegion();

  ko::Profiling::pushRegion("deep copy results to host");
  auto hoff = ko::create_mirror_view(offsets);
  auto hind = ko::create_mirror_view(indices);
  ko::deep_copy(hoff, offsets);
  ko::deep_copy(hind, indices);
  ko::Profiling::popRegion();

  // write the results out to a text file, with name provided as CLA
  ko::Profiling::pushRegion("write to file");
  FILE* outFile;
  std::string fname = argv[2];
  outFile = fopen(fname.c_str(), "w");

  // this goofy looking loop is because the results are stored in CRS format
  for (int i = 0; i < xlen; ++i) {
    for (int j = hoff(i); j < hoff(i + 1); ++j) {
      fmt::print(outFile, "{}", hind(j) + 1);
      if (j < hoff(i + 1) - 1) {
        fmt::print(outFile, ", ");
      }
    }
    fmt::print(outFile, "\n");
  }
  fclose(outFile);
  ko::Profiling::popRegion();

  return 0;
}
