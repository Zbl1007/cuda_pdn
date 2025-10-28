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
  std::complex<double> val;
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
                  const int *us, const int *vs,
                  const std::complex<double> *vals, double freq) {
  ts.clear();

  int i_begin = std::lower_bound(typs, typs + m, BR_I) - typs;
  int g_begin = std::lower_bound(typs, typs + m, BR_G) - typs;
  int r_begin = std::lower_bound(typs, typs + m, BR_R) - typs;
  int c_begin = std::lower_bound(typs, typs + m, BR_C) - typs;
  int l_begin = std::lower_bound(typs, typs + m, BR_L) - typs;
  int xc_begin = std::lower_bound(typs, typs + m, BR_XC) - typs;
  int xl_begin = std::lower_bound(typs, typs + m, BR_XL) - typs;
  int num_nonzero_nodes =
      std::max(*std::max_element(us, us + m), *std::max_element(vs, vs + m));

  auto make = [&](int i, std::complex<double> g) {
    int u = std::min(us[i], vs[i]) - 1;
    int v = std::max(us[i], vs[i]) - 1;
    if (u >= 0)
      ts.push_back({u, u, g});
    if (v >= 0)
      ts.push_back({v, v, g});
    if (u >= 0 && v >= 0)
      ts.push_back({u, v, -g});
  };
  std::complex jomega = std::complex<double>{0., 2 * PI * freq};
  for (int i = g_begin; i < r_begin; i++)
    make(i, vals[i]);
  for (int i = r_begin; i < c_begin; i++)
    make(i, 1. / vals[i]);

  for (int i = c_begin; i < l_begin; i++)
    make(i, jomega * vals[i]);
  for (int i = l_begin; i < xc_begin; i++)
    make(i, 1. / jomega / vals[i]);
  for (int i = xc_begin; i < xl_begin; i++)
    make(i, jomega / vals[i]);
  for (int i = xl_begin; i < m; i++)
    make(i, 1. / jomega * vals[i]);

  // Voltage Source
  for (int i = 0; i < i_begin; i++) {
    int u = us[i] - 1;
    int v = vs[i] - 1;
    if (u >= 0)
      ts.push_back({u, i + num_nonzero_nodes, 1.});
    if (v >= 0)
      ts.push_back({v, i + num_nonzero_nodes, -1.});
  }
  for (size_t i = 0; i < ts.size(); i++)
  {
    /* code */
    std::cout<<ts[i].row<<"  "<<ts[i].col<<"  "<<ts[i].val.real()<<"  "<<ts[i].val.imag()<<std::endl;
  }
  coalesce(ts);
  for (size_t i = 0; i < ts.size(); i++)
  {
    /* code */
    std::cout<<ts[i].row<<"  "<<ts[i].col<<"  "<<ts[i].val.real()<<"  "<<ts[i].val.imag()<<std::endl;
  }
  std::cout<<std::endl;
}

void initRhs(std::vector<std::complex<double>> &rhs, size_t m, const int *typs,
             const int *us, const int *vs, const std::complex<double> *vals) {
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

struct AcPardisoHandle {
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
  std::complex<double> *a;
  std::complex<double> *b;
  std::complex<double> *x;

  std::vector<Triplet> ts;
  std::vector<std::complex<double>> rhs;
};

void acPardisoHandleInit(AcPardisoHandle &h, const at::Tensor &branch_typ,
                         const at::Tensor &branch_u, const at::Tensor &branch_v,
                         const at::Tensor &branch_val) {
  size_t m = branch_typ.size(0);
  const int *typs = branch_typ.data_ptr<int>();
  const int *us = branch_u.data_ptr<int>();
  const int *vs = branch_v.data_ptr<int>();
  const c10::complex<double> *vals_c10 =
      branch_val.data_ptr<c10::complex<double>>();
  const std::complex<double> *vals =
      reinterpret_cast<const std::complex<double> *>(vals_c10);

  initTriplets(h.ts, m, typs, us, vs, vals, 1.);
  initRhs(h.rhs, m, typs, us, vs, vals);

  h.n = static_cast<MKL_INT>(h.rhs.size());
  h.ia = (MKL_INT *)malloc((h.n + 1) * sizeof(MKL_INT));
  h.ja = (MKL_INT *)malloc(h.ts.size() * sizeof(MKL_INT));
  h.a = (std::complex<double> *)malloc(
      h.ts.size() * sizeof(std::complex<double>)); // 修正为复数类型
  h.b = (std::complex<double> *)malloc(h.n * sizeof(std::complex<double>));
  h.x = (std::complex<double> *)malloc(h.n * sizeof(std::complex<double>));

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
  h.mtype = 6;
  h.nrhs = 1;
  h.maxfct = 1;
  h.mnum = 1;
  h.error = 0;
  h.msglvl = 0;

    // for (size_t i = 0; i < h.ts.size(); i++) {
    //     std::cout << "h_a["<<i<<"]:" << h.ts[i].val.real()<< ", " << h.ts[i].val.imag() << std::endl;
    // }
    // for (int i = 0; i < h.ts.size(); i++) {
    //     std::cout << "h_ia[" << i << "] = " << h.ia[i] << std::endl;
    // }
    // for (size_t i = 0; i < h.n + 1; i++) {
    //     std::cout << "h_ja[" << i << "] = " << h.ja[i] << std::endl;
    // }
    // exit(0);

  // Analysis
  h.phase = 11;
  PARDISO(h.pt, &h.maxfct, &h.mnum, &h.mtype, &h.phase, &h.n, NULL, h.ia, h.ja,
          NULL, &h.nrhs, h.iparm, &h.msglvl, NULL, NULL, &h.error);
  TORCH_CHECK(h.error == 0, "pardiso error ", h.error);
}

void acPardisoHandleFactorize(AcPardisoHandle &h, const at::Tensor &branch_typ,
                              const at::Tensor &branch_u,
                              const at::Tensor &branch_v,
                              const at::Tensor &branch_val, double freq) {
  size_t m = branch_typ.size(0);
  const int *typs = branch_typ.data_ptr<int>();
  const int *us = branch_u.data_ptr<int>();
  const int *vs = branch_v.data_ptr<int>();
  const c10::complex<double> *vals_c10 =
      branch_val.data_ptr<c10::complex<double>>();
  const std::complex<double> *vals =
      reinterpret_cast<const std::complex<double> *>(vals_c10);

  initTriplets(h.ts, m, typs, us, vs, vals, freq);
  for (size_t i = 0; i < h.ts.size(); i++) {
    h.a[i] = h.ts[i].val; // 赋值
  }
  // exit(0);
  h.phase = 22;
  PARDISO(h.pt, &h.maxfct, &h.mnum, &h.mtype, &h.phase, &h.n, h.a, h.ia, h.ja,
          NULL, &h.nrhs, h.iparm, &h.msglvl, NULL, NULL, &h.error);
  TORCH_CHECK(h.error == 0, "pardiso error ", h.error);
}

void acPardisoHandleSolve(AcPardisoHandle &h, const at::Tensor &branch_typ,
                          const at::Tensor &branch_u,
                          const at::Tensor &branch_v,
                          const at::Tensor &branch_val) {
  size_t m = branch_typ.size(0);
  const int *typs = branch_typ.data_ptr<int>();
  const int *us = branch_u.data_ptr<int>();
  const int *vs = branch_v.data_ptr<int>();
  const c10::complex<double> *vals_c10 =
      branch_val.data_ptr<c10::complex<double>>();
  const std::complex<double> *vals =
      reinterpret_cast<const std::complex<double> *>(vals_c10);

  initRhs(h.rhs, m, typs, us, vs, vals);
  memcpy(h.b, h.rhs.data(), h.n * sizeof(std::complex<double>));

  // Solve
  h.phase = 33;
  PARDISO(h.pt, &h.maxfct, &h.mnum, &h.mtype, &h.phase, &h.n, h.a, h.ia, h.ja,
          NULL, &h.nrhs, h.iparm, &h.msglvl, h.b, h.x, &h.error);
  TORCH_CHECK(h.error == 0, "pardiso error ", h.error);
}

void acPardisoHandleTerminate(AcPardisoHandle &h) {
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

at::Tensor acPardisoHandleGetSolution(AcPardisoHandle &h) {
  at::Tensor sln =
      at::empty({h.n}, at::TensorOptions().dtype(at::kComplexDouble));
  std::complex<double> *sln_ptr = reinterpret_cast<std::complex<double> *>(
      sln.data_ptr<c10::complex<double>>());

  memcpy(sln_ptr, h.x, h.n * sizeof(std::complex<double>));

  return sln;
}

std::vector<at::Tensor>
acPardisoHandleExport(AcPardisoHandle &h, const at::Tensor &branch_typ,
                      const at::Tensor &branch_u, const at::Tensor &branch_v,
                      const at::Tensor &branch_val, double freq) {
  size_t m = branch_typ.size(0);
  const int *typs = branch_typ.data_ptr<int>();
  const int *us = branch_u.data_ptr<int>();
  const int *vs = branch_v.data_ptr<int>();
  const c10::complex<double> *vals_c10 =
      branch_val.data_ptr<c10::complex<double>>();
  const std::complex<double> *vals =
      reinterpret_cast<const std::complex<double> *>(vals_c10);
  initTriplets(h.ts, m, typs, us, vs, vals, freq);
  initRhs(h.rhs, m, typs, us, vs, vals);

  int64_t n = h.rhs.size();
  int64_t nnz = h.ts.size();
  std::vector<int64_t> row_indices;
  std::vector<int64_t> col_indices;
  std::vector<std::complex<double>> values;
  for (int64_t i = 0; i < nnz; i++) {
    int64_t u = h.ts[i].row;
    int64_t v = h.ts[i].col;
    std::complex<double> d = h.ts[i].val;
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

  // 1. 创建复数 Tensor (values)
  at::Tensor values_tensor =
      at::empty({static_cast<int64_t>(values.size())},
                at::TensorOptions().dtype(torch::kComplexDouble));
  memcpy(values_tensor.data_ptr<c10::complex<double>>(), values.data(),
         values.size() * sizeof(std::complex<double>));

  // 2. 创建稀疏张量
  at::Tensor mat =
      torch::sparse_coo_tensor(indices_tensor, values_tensor, {n, n});

  // 3. 创建复数 Tensor (rhs)
  at::Tensor rhs = at::empty({static_cast<int64_t>(h.rhs.size())},
                             at::TensorOptions().dtype(torch::kComplexDouble));
  memcpy(rhs.data_ptr<c10::complex<double>>(), h.rhs.data(),
         h.rhs.size() * sizeof(std::complex<double>));

  return {mat, rhs};
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<AcPardisoHandle>(m, "AcPardisoHandle").def(py::init<>());
  m.def("acPardisoHandleInit", &acPardisoHandleInit, "acPardisoHandleInit");
  m.def("acPardisoHandleFactorize", &acPardisoHandleFactorize,
        "acPardisoHandleFactorize");
  m.def("acPardisoHandleSolve", &acPardisoHandleSolve, "acPardisoHandleSolve");
  m.def("acPardisoHandleTerminate", &acPardisoHandleTerminate,
        "acPardisoHandleTerminate");
  m.def("acPardisoHandleGetSolution", &acPardisoHandleGetSolution,
        "acPardisoHandleGetSolution");
  m.def("acPardisoHandleExport", &acPardisoHandleExport,
        "acPardisoHandleExport");
}
