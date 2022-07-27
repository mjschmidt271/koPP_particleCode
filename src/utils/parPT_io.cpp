#include "parPT_io.hpp"

namespace particles {

ParticleIO::ParticleIO(Params& _params, const std::string& yaml_name) {
  YAML::Node file_params = YAML::LoadFile(yaml_name);

  _params.Np = file_params["Np"].as<int>();
  _params.dim = file_params["dimension"].as<int>();
  _params.n_chunks = file_params["n_chunks"].as<int>();
  _params.IC_str_space =
      file_params["initial_condition"]["space"]["type"].as<std::string>();
  _params.IC_str_mass =
      file_params["initial_condition"]["mass"]["type"].as<std::string>();
  for (auto iter : file_params["omega"]) {
    _params.omega.push_back(iter.as<Real>());
  }
  std::string rand_seed_str;
  // the parser will crash is the type string is empty, so check first
  // Note: for whatever reason, IsScalar() seems to do the best job of
  // determining whether or not the value is empty when the key is in the yaml
  if (file_params["rand_seed_type"] and
      file_params["rand_seed_type"].IsScalar()) {
    rand_seed_str = file_params["rand_seed_type"].as<std::string>();
  } else {
    rand_seed_str = "missing";
  }
  enumerate_seed_type(_params, rand_seed_str, file_params);
  enumerate_IC(_params, _params.IC_str_space, file_params, true);
  enumerate_IC(_params, _params.IC_str_mass, file_params, false);
  set_seed_val(_params, file_params);

  _params.maxT = file_params["maxT"].as<Real>();
  _params.dt = file_params["dt"].as<Real>();
  _params.D = file_params["D"].as<Real>();
  _params.pctRW = file_params["pctRW"].as<Real>();
  _params.denom = 4 * _params.D * (1.0 - _params.pctRW) * _params.dt;
  _params.cdist_coeff = file_params["cdist_coeff"].as<Real>();
  _params.cutdist = _params.cdist_coeff * sqrt(_params.denom);
  _params.pFile = file_params["pFile"].as<std::string>();
  if (file_params["write_plot"]) {
    _params.write_plot = file_params["write_plot"].as<bool>();
  } else {
    _params.write_plot = false;
  }
  _params.nSteps = ceil(_params.maxT / _params.dt);

  if (_params.write_plot) {
    outfile = fopen(_params.pFile.c_str(), "w");
  }
}

void ParticleIO::enumerate_IC(Params& params, std::string& IC_str,
                              const YAML::Node& yml, const bool& space) {
  transform(IC_str.begin(), IC_str.end(), IC_str.begin(), ::tolower);
  if (space) {
    if (IC_str.compare("point") == 0) {
      params.IC_type_space = point_loc;
      params.X0_space = yml["initial_condition"]["space"]["X0"].as<Real>();
    } else if (IC_str.compare("uniform") == 0) {
      params.IC_type_space = uniform;
    } else if (IC_str.compare("equi") == 0) {
      params.IC_type_space = equi;
    } else if (IC_str.compare("hat") == 0) {
      params.IC_type_space = hat;
      params.hat_pct = yml["initial_condition"]["space"]["hat_pct"].as<Real>();
    } else {
      fmt::print(stderr, "ERROR: Unsupported initial condition type: {}\n",
                 &IC_str[0]);
      exit(1);
    }
  } else {
    if (IC_str.compare("point") == 0) {
      params.IC_type_mass = point_mass;
      params.X0_mass = yml["initial_condition"]["mass"]["X0"].as<Real>();
    } else if (IC_str.compare("heaviside") == 0) {
      params.IC_type_mass = heaviside;
      params.X0_mass = yml["initial_condition"]["mass"]["X0"].as<Real>();
    } else if (IC_str.compare("gaussian") == 0) {
      params.IC_type_mass = gaussian;
      params.X0_mass = yml["initial_condition"]["mass"]["X0"].as<Real>();
    } else {
      fmt::print(stderr, "ERROR: Unsupported initial condition type: {}\n",
                 &IC_str[0]);
      exit(1);
    }
  }
}

void ParticleIO::enumerate_seed_type(Params& params, std::string& seed_str,
                                     const YAML::Node& yml) {
  if (!seed_str.empty()) {
    transform(seed_str.begin(), seed_str.end(), seed_str.begin(), ::tolower);
  }
  if (seed_str.compare("default") == 0) {
    params.seed_type = default_rand;
  } else if (seed_str.compare("clock") == 0) {
    params.seed_type = clock_rand;
  } else if (seed_str.compare("specified") == 0) {
    params.seed_type = specified_rand;
  } else if (seed_str.compare("missing") == 0) {
    params.seed_type = missing;
  } else {
    params.seed_type = default_rand;
  }
}

void ParticleIO::set_seed_val(Params& params, const YAML::Node& yml) {
  switch (params.seed_type) {
    case clock_rand: {
      params.seed_val =
          std::chrono::high_resolution_clock::now().time_since_epoch().count();
      if (params.seed_val % 2 == 0) {
        params.seed_val = params.seed_val + 1;
      }
      break;
    }
    case specified_rand: {
      if (yml["rand_seed_value"] and yml["rand_seed_value"].IsScalar()) {
        params.seed_val = yml["rand_seed_value"].as<uint64_t>();
      } else {
        params.seed_type = missing;
        params.seed_val = 5374857;
      }
      break;
    }
    case default_rand: {
      params.seed_val = 5374857;
      break;
    }
    case missing: {
      fmt::print(
          "Seed type or value not provided--using default seed 5374857.\n");
      params.seed_val = 5374857;
      break;
    }
    default: {
      // this should not be possible
      fmt::print(
          "Seed type or value is invalid--using default seed 5374857.\n");
      params.seed_val = 5374857;
      break;
    }
  }
}

void ParticleIO::set_positions(const Params& params,
                               const std::string& yaml_name,
                               ko::View<Real**>& X) {
  YAML::Node file_params = YAML::LoadFile(yaml_name);

  std::vector<std::string> dims{"x", "y", "z"};

  auto hX = ko::create_mirror_view(X);
  auto points = file_params["points"];
  for (auto j = 0; j < params.dim; ++j) {
    auto dim_pt = points[dims[j]];
    int i = 0;
    for (const auto& pt : dim_pt) {
      hX(j, i) = pt.as<Real>();
      ++i;
    }
  }
  ko::deep_copy(X, hX);
}

void print_version_info() {
  // note: the format statement means the text will be centered in 60 total
  // spaces
  fmt::print("{:^60}\n",
             "1D Particle Tracking With Random Walks and Mass Transfer");
  fmt::print("{:^60}\n", version_str);
}

void ParticleIO::print_params_summary(const Params& params) {
  fmt::print("{}\n",
             "************************************************************");
  print_version_info();
  fmt::print("{}\n",
             "************************************************************");
  fmt::print("Np = {}\n", params.Np);
  fmt::print("maxT = {}\n", params.maxT);
  fmt::print("dt = {}\n", params.dt);
  fmt::print("nSteps = {}\n", params.nSteps);
  fmt::print("dimension = {}\n", params.dim);
  fmt::print("N tree decomp. chunks = {}\n", params.n_chunks);
  fmt::print("omega = [{}, {}]\n", params.omega[0], params.omega[1]);
  fmt::print("IC type (space) = {}\n", params.IC_str_space);
  fmt::print("IC type (mass) = {}\n", params.IC_str_mass);
  if (params.IC_type_space == point_loc) {
    fmt::print("X0 space = {}\n", params.X0_space);
  } else if (params.IC_type_space == hat) {
    fmt::print("hat_pct = {}\n", params.hat_pct);
  }
  fmt::print("X0 mass = {}\n", params.X0_mass);
  fmt::print("D = {}\n", params.D);
  fmt::print("pctRW = {}\n", params.pctRW);
  fmt::print("denom = {}\n", params.denom);
  fmt::print("cdist_coeff = {}\n", params.cdist_coeff);
  fmt::print("cutdist = {}\n", params.cutdist);
  fmt::print("seed_type = {}\n", params.seed_type);
  fmt::print("seed_val = {}\n", params.seed_val);
  fmt::print("pFile = {}\n", params.pFile);
  fmt::print("write plot info (position/mass) = {}\n", params.write_plot);
  fmt::print("{}\n\n",
             "************************************************************");
}

void ParticleIO::write(const Params& params, const ko::View<Real**>& X,
                       const ko::View<Real*>& mass, const int& tStep) {
  ko::Profiling::pushRegion("write_deep_copy");
  auto hX = ko::create_mirror_view(X);
  auto hmass = ko::create_mirror_view(mass);
  ko::deep_copy(hX, X);
  ko::deep_copy(hmass, mass);
  ko::Profiling::popRegion();
  if (tStep == 0) {
    ko::Profiling::pushRegion("print_meta_info");
    fmt::print(outfile, "{} {}\n", params.Np, params.nSteps);
    fmt::print(outfile, "{}\n", params.dim);
    fmt::print(outfile, "{} ", params.IC_type_space);
    fmt::print(outfile, "{} ", params.IC_type_mass);
    for (auto i : params.omega) {
      fmt::print(outfile, "{} ", i);
    }
    if (params.IC_type_space == point_loc) {
      fmt::print(outfile, "{} -999 ", params.X0_space);
    } else if (params.IC_type_space == hat) {
      fmt::print(outfile, "{} -999 ", params.hat_pct);
    } else if (params.IC_type_space == equi) {
      fmt::print(outfile, "-999 -999 ");
    }
    fmt::print(outfile, "{} {} {} {} {} {} {}\n", params.X0_mass, params.maxT,
               params.dt, params.D, params.pctRW, params.cdist_coeff,
               params.cutdist);
    ko::Profiling::popRegion();
    ko::Profiling::pushRegion("print_position and mass");
    for (size_t i = 0; i < hX.extent(1); ++i) {
      for (size_t j = 0; j < hX.extent(0); ++j) {
        // ko::Profiling::pushRegion("print_position");
        fmt::print(outfile, "{} ", hX(j, i));
        // ko::Profiling::popRegion();
      }
      // ko::Profiling::pushRegion("print_mass");
      fmt::print(outfile, "{}\n", hmass(i));
      // ko::Profiling::popRegion();
    }
    ko::Profiling::popRegion();
  } else {
    for (size_t i = 0; i < hX.extent(1); ++i) {
      for (size_t j = 0; j < hX.extent(0); ++j) {
        fmt::print(outfile, "{} ", hX(j, i));
      }
      fmt::print(outfile, "{}\n", hmass(i));
    }
  }
}

}  // namespace particles
