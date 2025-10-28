#define GLOG_USE_GLOG_EXPORT
#include <algorithm>
#include <assert.h>
#include <memory.h>
#include <mkl_pardiso.h>
#include <mkl_types.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <vector>

#define BR_V 1
#define BR_I 2
#define BR_G 3
#define BR_R 4
#define BR_C 5
#define BR_L 6
#define BR_XC 7
#define BR_XL 8
#define PI 3.1415926

struct Triplet {
  int row, col;
  double val;
  bool operator<(const Triplet &other) const {
    return row < other.row || (row == other.row && col < other.col);
  }
};

void coalesce(std::vector<Triplet> &ts) {
  if (ts.empty()) {
    return;
  }
  std::sort(ts.begin(), ts.end());
  auto result = ts.begin();
  for (auto it = ts.begin() + 1; it != ts.end(); ++it) {
    if (it->row == result->row && it->col == result->col) {
      result->val += it->val;
    } else {
      ++result;
      *result = *it;
    }
  }
  ts.erase(result + 1, ts.end());
}

void initTriplets(std::vector<Triplet> &ts, size_t m, const int *typs,
                  const int *us, const int *vs, const double *vals) {
  ts.clear();

  int i_begin = std::lower_bound(typs, typs + m, BR_I) - typs;
  int g_begin = std::lower_bound(typs, typs + m, BR_G) - typs;
  int r_begin = std::lower_bound(typs, typs + m, BR_R) - typs;
  int num_nonzero_nodes =
      std::max(*std::max_element(us, us + m), *std::max_element(vs, vs + m));

  auto make = [&](int i, double g) {
    int u = std::min(us[i], vs[i]) - 1;
    int v = std::max(us[i], vs[i]) - 1;
    if (u >= 0)
      ts.push_back({u, u, g});
    if (v >= 0)
      ts.push_back({v, v, g});
    if (u >= 0 && v >= 0)
      ts.push_back({u, v, -g});
  };
  for (int i = g_begin; i < r_begin; i++)
    make(i, vals[i]);
  for (int i = r_begin; i < m; i++)
    make(i, 1. / vals[i]);

  // voltage source
  for (int i = 0; i < i_begin; i++) {
    int u = us[i] - 1;
    int v = vs[i] - 1;
    if (u >= 0)
      ts.push_back({u, i + num_nonzero_nodes, 1.});
    if (v >= 0)
      ts.push_back({v, i + num_nonzero_nodes, -1.});
  }

  coalesce(ts);
}

void initRhs(std::vector<double> &rhs, size_t m, const int *typs, const int *us,
             const int *vs, const double *vals) {
  int num_nonzero_nodes =
      std::max(*std::max_element(us, us + m), *std::max_element(vs, vs + m));
  int i_begin = std::lower_bound(typs, typs + m, BR_I) - typs;
  int g_begin = std::lower_bound(typs, typs + m, BR_G) - typs;

  rhs.assign(num_nonzero_nodes + i_begin, 0.);
  // Current Source
  for (int i = i_begin; i < g_begin; i++) {
    int u = us[i] - 1;
    int v = vs[i] - 1;
    if (u >= 0)
      rhs[u] -= vals[i];
    if (v >= 0)
      rhs[v] += vals[i];
  }
  // Voltage Source
  for (int i = 0; i < i_begin; i++) {
    rhs[i + num_nonzero_nodes] = vals[i];
  }
}

struct OpPardisoHandle {
  void *pt[64];
  MKL_INT iparm[64];
  MKL_INT mtype;
  MKL_INT n;
  MKL_INT *ia;
  MKL_INT *ja;
  MKL_INT phase;
  MKL_INT nrhs;
  MKL_INT error;
  MKL_INT msglvl;
  MKL_INT maxfct;
  MKL_INT mnum;
  double *a;
  double *b;
  double *x;

  std::vector<Triplet> ts;
  std::vector<double> rhs;
};

void opPardisoHandleInit(OpPardisoHandle &h, const at::Tensor &branch_typ,
                         const at::Tensor &branch_u, const at::Tensor &branch_v,
                         const at::Tensor &branch_val) {
  size_t m = branch_typ.size(0);
  const int *typs = branch_typ.data_ptr<int>();
  const int *us = branch_u.data_ptr<int>();
  const int *vs = branch_v.data_ptr<int>();
  const double *vals = branch_val.data_ptr<double>();

  initTriplets(h.ts, m, typs, us, vs, vals);
  initRhs(h.rhs, m, typs, us, vs, vals);

  h.n = static_cast<MKL_INT>(h.rhs.size());
  h.ia = (MKL_INT *)malloc((h.n + 1) * sizeof(MKL_INT));
  h.ja = (MKL_INT *)malloc(h.ts.size() * sizeof(MKL_INT));
  h.a = (double *)malloc(h.ts.size() * sizeof(double));
  h.b = (double *)malloc(h.n * sizeof(double));
  h.x = (double *)malloc(h.n * sizeof(double));
  h.ia[0] = 0;
  MKL_INT r = 0;
  for (size_t i = 0; i < h.ts.size(); i++) {
    while (h.ts[i].row > r) {
      h.ia[++r] = i;
    }
    h.ja[i] = h.ts[i].col;
  }
  while (h.n > r) {
    h.ia[++r] = static_cast<MKL_INT>(h.ts.size());
  }
  memset(h.pt, 0, sizeof(h.pt));
  memset(h.iparm, 0, sizeof(h.iparm));
  h.iparm[0] = 1;
  h.iparm[1] = 2; // metis
  h.iparm[7] = 2;
  h.iparm[9] = 13;
  h.iparm[10] = 1;
  h.iparm[12] = 0;
  h.iparm[34] = 1;
  h.mtype = -2; // real symmetric matrix
  h.nrhs = 1;
  h.maxfct = 1;
  h.mnum = 1;
  h.error = 0;
  h.msglvl = 0;

  // Analysis
  h.phase = 11;
  PARDISO(h.pt, &h.maxfct, &h.mnum, &h.mtype, &h.phase, &h.n, NULL, h.ia, h.ja,
          NULL, &h.nrhs, h.iparm, &h.msglvl, NULL, NULL, &h.error);
  TORCH_CHECK(h.error == 0, "pardiso error ", h.error);
}

void opPardisoHandleFactorize(OpPardisoHandle &h, const at::Tensor &branch_typ,
                              const at::Tensor &branch_u,
                              const at::Tensor &branch_v,
                              const at::Tensor &branch_val) {
  size_t m = branch_typ.size(0);
  const int *typs = branch_typ.data_ptr<int>();
  const int *us = branch_u.data_ptr<int>();
  const int *vs = branch_v.data_ptr<int>();
  const double *vals = branch_val.data_ptr<double>();
  initTriplets(h.ts, m, typs, us, vs, vals);
  for (size_t i = 0; i < h.ts.size(); i++) {
    h.a[i] = h.ts[i].val;
  }

  // Factorization
  h.phase = 22;
  PARDISO(h.pt, &h.maxfct, &h.mnum, &h.mtype, &h.phase, &h.n, h.a, h.ia, h.ja,
          NULL, &h.nrhs, h.iparm, &h.msglvl, NULL, NULL, &h.error);
  TORCH_CHECK(h.error == 0, "pardiso error ", h.error);
}

void opPardisoHandleSolve(OpPardisoHandle &h, const at::Tensor &branch_typ,
                          const at::Tensor &branch_u,
                          const at::Tensor &branch_v,
                          const at::Tensor &branch_val) {
  size_t m = branch_typ.size(0);
  const int *typs = branch_typ.data_ptr<int>();
  const int *us = branch_u.data_ptr<int>();
  const int *vs = branch_v.data_ptr<int>();
  const double *vals = branch_val.data_ptr<double>();
  initRhs(h.rhs, m, typs, us, vs, vals);
  memcpy(h.b, h.rhs.data(), h.n * sizeof(double));

  // Solve
  h.phase = 33;
  PARDISO(h.pt, &h.maxfct, &h.mnum, &h.mtype, &h.phase, &h.n, h.a, h.ia, h.ja,
          NULL, &h.nrhs, h.iparm, &h.msglvl, h.b, h.x, &h.error);
  TORCH_CHECK(h.error == 0, "pardiso error ", h.error);
}

void opPardisoHandleTerminate(OpPardisoHandle &h) {
  // Terminate
  h.phase = -1;
  PARDISO(h.pt, &h.maxfct, &h.mnum, &h.mtype, &h.phase, &h.n, NULL, h.ia, h.ja,
          NULL, &h.nrhs, h.iparm, &h.msglvl, NULL, NULL, &h.error);
  TORCH_CHECK(h.error == 0, "pardiso error ", h.error);

  free(h.ia);
  free(h.ja);
  free(h.a);
  free(h.b);
  free(h.x);
}

at::Tensor opPardisoHandleGetSolution(OpPardisoHandle &h) {
  at::Tensor sln = at::empty({h.n}, at::TensorOptions().dtype(at::kDouble));
  double *sln_ptr = sln.data_ptr<double>();
  memcpy(sln_ptr, h.x, h.n * sizeof(double));
  return sln;
}

std::vector<at::Tensor> opPardisoHandleExport(OpPardisoHandle &h,
                                              const at::Tensor &branch_typ,
                                              const at::Tensor &branch_u,
                                              const at::Tensor &branch_v,
                                              const at::Tensor &branch_val) {
  size_t m = branch_typ.size(0);
  const int *typs = branch_typ.data_ptr<int>();
  const int *us = branch_u.data_ptr<int>();
  const int *vs = branch_v.data_ptr<int>();
  const double *vals = branch_val.data_ptr<double>();
  initTriplets(h.ts, m, typs, us, vs, vals);
  initRhs(h.rhs, m, typs, us, vs, vals);

  int64_t n = h.rhs.size();
  int64_t nnz = h.ts.size();
  std::vector<int64_t> row_indices;
  std::vector<int64_t> col_indices;
  std::vector<double> values;
  for (int64_t i = 0; i < nnz; i++) {
    int64_t u = h.ts[i].row;
    int64_t v = h.ts[i].col;
    double d = h.ts[i].val;
    if (u != v) {
      row_indices.push_back(u);
      col_indices.push_back(v);
      values.push_back(d);

      row_indices.push_back(v);
      col_indices.push_back(u);
      values.push_back(d);
    } else {
      row_indices.push_back(u);
      col_indices.push_back(u);
      values.push_back(d);
    }
  }
  at::Tensor indices_tensor = torch::stack(
      {
          torch::tensor(row_indices, at::kLong),
          torch::tensor(col_indices, at::kLong),
      },
      0);
  at::Tensor values_tensor = torch::tensor(values, torch::kDouble);
  at::Tensor mat =
      torch::sparse_coo_tensor(indices_tensor, values_tensor, {n, n});
  at::Tensor rhs = torch::tensor(h.rhs, torch::kDouble);
  return {mat, rhs};
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<OpPardisoHandle>(m, "OpPardisoHandle").def(py::init<>());
  m.def("opPardisoHandleInit", &opPardisoHandleInit, "opPardisoHandleInit");
  m.def("opPardisoHandleFactorize", &opPardisoHandleFactorize,
        "opPardisoHandleFactorize");
  m.def("opPardisoHandleSolve", &opPardisoHandleSolve, "opPardisoHandleSolve");
  m.def("opPardisoHandleTerminate", &opPardisoHandleTerminate,
        "opPardisoHandleTerminate");
  m.def("opPardisoHandleGetSolution", &opPardisoHandleGetSolution,
        "opPardisoHandleGetSolution");
  m.def("opPardisoHandleExport", &opPardisoHandleExport,
        "opPardisoHandleExport");
}
