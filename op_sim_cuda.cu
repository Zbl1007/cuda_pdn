#define GLOG_USE_GLOG_EXPORT
#include <algorithm>
#include <assert.h>
#include <memory.h>
#include <mkl_pardiso.h>
#include <mkl_types.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <vector>


// CUDA 和 cudss 头文件
#include <cuda_runtime.h>
#include "cudss.h"
#include <cuComplex.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>

#define BR_V 1
#define BR_I 2
#define BR_G 3
#define BR_R 4
#define BR_C 5
#define BR_L 6
#define BR_XC 7
#define BR_XL 8
#define PI 3.1415926


// 用于检查所有 CUDA/cuSOLVER API 调用的宏

#define CUDA_CHECK(call) \
    do { \
        cudaError_t cuda_error = call; \
        if (cuda_error != cudaSuccess) { \
            throw std::runtime_error("CUDA error at " __FILE__ ":" + std::to_string(__LINE__) + \
                                   " - " + std::string(cudaGetErrorString(cuda_error))); \
        } \
    } while(0)

#define CUDSS_CHECK(call) \
    do { \
        cudssStatus_t cudss_status = call; \
        if (cudss_status != CUDSS_STATUS_SUCCESS) { \
            throw std::runtime_error("cuDSS error at " __FILE__ ":" + std::to_string(__LINE__) + \
                                   " - status code: " + std::to_string(cudss_status)); \
        } \
    } while(0)

struct Triplet {
  int row, col;
  double val;
  bool operator<(const Triplet &other) const {
    return row < other.row || (row == other.row && col < other.col);
  }
};

struct Triplet_GPU {
  int row, col;
  double val;
  int idx; // 新增：记录这个 triplet 来自哪个 branch
};


// -----------------------------------------------------------------
// ----------------- 内核 1: 计数 (第 1 趟) ------------------------
// -----------------------------------------------------------------
// 这个内核不写入三元组，只计算每个 branch 将会生成多少个三元组
__global__ void count_triplets_kernel(
    int m,
    const int* typs,
    const int* us,
    const int* vs,
    int* out_counts) // 输出：每个 branch 的三元组数量 (0, 1, 2, 或 3)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    int local_count = 0;
    int typ = typs[i];

    // (这个 'if' 结构必须与 build_triplets_kernel 中的逻辑完全一致)
    if (typ == 1) { // 电压源
        int u = us[i] - 1;
        int v = vs[i] - 1;
        if (u >= 0) local_count++;
        if (v >= 0) local_count++;
    } else if (typ >= 3) { // 无源元件 (G, R, C, L, ...)
        int u = min(us[i], vs[i]) - 1;
        int v = max(us[i], vs[i]) - 1;
        if (u >= 0) local_count++;
        if (v >= 0) local_count++;
        if (u >= 0 && v >= 0) local_count++;
    }
    // typ == 2 (电流源) 或其他未知类型，local_count 保持为 0

    out_counts[i] = local_count;
}

// -----------------------------------------------------------------
// ----------------- 内核 2: 写入 (第 2 趟) ------------------------
// -----------------------------------------------------------------
// 这个内核现在没有 atomicAdd，而是使用 d_offsets 来精确定位
__global__ void build_triplets_kernel(
    int m,
    const int* typs,
    const int* us,
    const int* vs,
    const double* vals,
    Triplet_GPU* out_triplets, // 输出：紧凑的三元组数组
    const int* d_offsets, // 输入：每个 branch 的写入偏移量
    int num_nonzero_nodes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    // 1. 获取这个线程的“基地址”
    int my_base_offset = d_offsets[i];
    int local_write_count = 0; // 本地（私有）计数器

    // 计算电导g
    double g = 0.0;
    int typ = typs[i];

    if (typ == BR_G) { g = vals[i]; }
    else if (typ == BR_R) { g = 1.0 / vals[i]; }
    else if (typ == BR_V) { g = 1.0; }
    else { return; } // 跳过 BR_I (typ=2) 和其他

    
    if (typ != BR_V) { // 无源元件: produce up to 3 entries
        int u = min(us[i], vs[i]) - 1;
        int v = max(us[i], vs[i]) - 1;

        if (u >= 0) {
            int pos = my_base_offset + local_write_count; // 精确索引
            out_triplets[pos].row = u;
            out_triplets[pos].col = u;
            out_triplets[pos].val = g;
            local_write_count++;
        }
        if (v >= 0) {
            int pos = my_base_offset + local_write_count;
            out_triplets[pos].row = v;
            out_triplets[pos].col = v;
            out_triplets[pos].val = g;
            local_write_count++;
        }
        if (u >= 0 && v >= 0) {
            int pos = my_base_offset + local_write_count;
            out_triplets[pos].row = u;
            out_triplets[pos].col = v;
            out_triplets[pos].val = -g;
            local_write_count++;
        }
    } else { // 电压源
        int u = us[i] - 1;
        int v = vs[i] - 1;
        if (u >= 0) {
            int pos = my_base_offset + local_write_count;
            out_triplets[pos].row = u;
            out_triplets[pos].col = i + num_nonzero_nodes;
            out_triplets[pos].val = 1.0;
            local_write_count++;
        }
        if (v >= 0) {
            int pos = my_base_offset + local_write_count;
            out_triplets[pos].row = v;
            out_triplets[pos].col = i + num_nonzero_nodes;
            out_triplets[pos].val = -1.0;
            local_write_count++;
        }
    }
}

// --- (pack_key, unpack_key, ComplexAdd 保持不变) ---
__host__ __device__ inline uint64_t pack_key(int row, int col) {
    return ( (uint64_t)( (uint32_t)row ) << 32 ) | (uint64_t)( (uint32_t)col );
}
__host__ __device__ inline void unpack_key(uint64_t key, int &row, int &col) {
    row = (int)( (key >> 32) & 0xFFFFFFFFu );
    col = (int)( key & 0xFFFFFFFFu );
}
struct RealAdd  {
    __host__ __device__ double operator()(const double &a, const double &b) const {
        return a + b;  // 普通实数加法
    }
};


// -----------------------------------------------------------------
// ----------------- 主机端函数 (已重构) ---------------------------
// -----------------------------------------------------------------
void initTripletsGPU_thrust(
    int m,
    const int* d_typs,
    const int* d_us,
    const int* d_vs,
    const double* d_vals,
    int h_maxnode, // <-- 从 Factorize 函数传入
    double* d_out_vals_ptr // <-- 目标 GPU 指针 (即 h.d_a)
    )
{
    // ---------------------------------------------------
    // --- 步骤 1: 在 GPU 上计算 h_maxnode ---
    // ---------------------------------------------------
    // int h_maxnode = 0;
    if (m > 0) {
        // 1. 创建 device_ptr (零拷贝)
        thrust::device_ptr<const int> d_us_ptr(d_us);
        thrust::device_ptr<const int> d_vs_ptr(d_vs);
        
        // 2. 在 GPU 上并行查找最大值
        int max_u = *thrust::max_element(d_us_ptr, d_us_ptr + m);
        int max_v = *thrust::max_element(d_vs_ptr, d_vs_ptr + m);
        
        // 3. 在 CPU 上比较两个结果
        h_maxnode = max(max_u, max_v);
    }
    
    int threads = 1024;
    int blocks = (m + threads - 1)/threads;

    // ---------------------------------------------------
    // --- 步骤 2: 第 1 趟 - 计数内核 ---
    // ---------------------------------------------------
    thrust::device_vector<int> d_counts(m);
    count_triplets_kernel<<<blocks, threads>>>(
        m, d_typs, d_us, d_vs, thrust::raw_pointer_cast(d_counts.data())
    );
    cudaDeviceSynchronize();

    // ---------------------------------------------------
    // --- 步骤 3: 并行前缀和 (Scan) ---
    // ---------------------------------------------------
    thrust::device_vector<int> d_offsets(m);
    thrust::exclusive_scan(d_counts.begin(), d_counts.end(), d_offsets.begin());
    
    // --- 计算总大小 h_count ---
    int h_count = 0;
    if (m > 0) {
        // 总数 = 最后一个偏移量 + 最后一个计数
        // (这只会从 GPU 拷贝两个整数，非常快)
        int last_offset = d_offsets.back();
        int last_count = d_counts.back();
        h_count = last_offset + last_count;
    }

    if (h_count == 0) {
        // 没有三元组，提前退出
        // out_rows.clear(); out_cols.clear(); out_vals.clear();
        return;
    }

    // ---------------------------------------------------
    // --- 步骤 4: 分配 *精确* 内存并执行第 2 趟 (写入) ---
    // ---------------------------------------------------
    Triplet_GPU* d_raw = nullptr;
    cudaMalloc(&d_raw, sizeof(Triplet_GPU) * h_count); // <-- 分配紧凑的内存

    build_triplets_kernel<<<blocks, threads>>>(
        m, d_typs, d_us, d_vs, d_vals,
        d_raw, // 写入紧凑的数组
        thrust::raw_pointer_cast(d_offsets.data()), // 传入偏移量
        h_maxnode
    );
    cudaDeviceSynchronize();

    // ---------------------------------------------------
    // --- 步骤 5: 打包、排序、归约 (与之前相同) ---
    // ---------------------------------------------------
    
    // 2) 创建 device vectors (大小为 h_count)
    thrust::device_vector<uint64_t> d_keys(h_count);
    thrust::device_vector<double> d_vals_vec(h_count);

    // 3) 启动打包内核 (大小为 h_count)
    int b2 = (h_count + threads - 1)/threads;
    extern __global__ void fill_triplet_arrays_kernel(const Triplet_GPU* raw, uint64_t* keys, double* vals, int n);
    fill_triplet_arrays_kernel<<<b2, threads>>>(d_raw, thrust::raw_pointer_cast(d_keys.data()), thrust::raw_pointer_cast(d_vals_vec.data()), h_count);
    cudaDeviceSynchronize();

    // 4) 排序
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_vals_vec.begin());

    // 5) 归约
    thrust::device_vector<uint64_t> d_out_keys(h_count);
    thrust::device_vector<double> d_out_vals(h_count);
    auto new_end = thrust::reduce_by_key(
        d_keys.begin(), d_keys.end(),
        d_vals_vec.begin(),
        d_out_keys.begin(),
        d_out_vals.begin(),
        thrust::equal_to<uint64_t>(),
        RealAdd()
    );
    int out_n = new_end.first - d_out_keys.begin();
    thrust::copy(
        d_out_vals.begin(),                 // 源 (临时 device_vector)
        d_out_vals.begin() + out_n,         // 源结束
        thrust::device_pointer_cast(d_out_vals_ptr) // 目标 (h.d_a)
    );

    // ---------------------------------------------------
    // --- 步骤 7: 清理 ---
    // ---------------------------------------------------
    cudaFree(d_raw);
    // return out_n;

}


// -----------------------------------------------------------------
// ----------------- 打包内核 (已简化) -----------------------------
// -----------------------------------------------------------------
// 这个内核现在变得非常简单，因为它不需要再检查 -1 了
__global__ void fill_triplet_arrays_kernel(const Triplet_GPU* raw, uint64_t* keys, double* vals, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Triplet_GPU t = raw[idx];
    
    // 所有三元组都是有效的，直接打包
    keys[idx] = pack_key(t.row, t.col);
    vals[idx] = t.val;
}


__global__ void update_rhs_kernel(
    double* d_b,                 // 要更新的 RHS 向量 (输出)
    int n,                                // RHS 向量长度
    const int* d_branch_typs,             // branch 类型
    const int* d_branch_us,               // branch u 节点
    const int* d_branch_vs,               // branch v 节点
    const double* d_branch_vals, // branch 值
    int m,                                // branch 总数
    int num_nonzero_nodes                // 非零节点数
) {
    // 每个线程处理一个 branch
    int branch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (branch_idx >= m) return;

    int type = d_branch_typs[branch_idx];
    
    if (type == BR_I) { // 电流源
        int u = d_branch_us[branch_idx] - 1;
        int v = d_branch_vs[branch_idx] - 1;
        double val = d_branch_vals[branch_idx];
        if (u >= 0) atomicAdd(&d_b[u], -val);
        if (v >= 0) atomicAdd(&d_b[v],  val);
    } else if (type == BR_V) { // 电压源
        // 假设电压源的 branch_idx 从 0 开始
        d_b[num_nonzero_nodes + branch_idx] = d_branch_vals[branch_idx];
    }
}

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

struct OpCuDssHandle {
  cudssHandle_t handle = nullptr;
  cudssConfig_t config = nullptr;
  cudssData_t   data   = nullptr;
  
  // 描述符来自 CUSPARSE 库
  cudssMatrix_t matA = nullptr;
  cudssMatrix_t vecX = nullptr;  // Solution vector x
  cudssMatrix_t vecB = nullptr;  // RHS vector b


  // CPU 上的矩阵/向量数据 (临时)
  int n;    // Matrix dimension
  int nnz;  // Number of non-zero elements
  std::vector<double> h_x; // Solution

  // --- GPU 上的 CSR 矩阵和向量 ---
  int *d_ia = nullptr;
  int *d_ja = nullptr;
  double *d_a = nullptr;
  double *d_x = nullptr;
  double *d_b = nullptr;

  // === 新增：在 GPU 上存储电路基础信息 ===
  // 映射关系：h.ts 中第 k 个元素的值，应该由哪个 branch 计算得到
  // int* d_branch_indices_for_nnz = nullptr;
  double* d_signs_for_nnz = nullptr;

  
  // 电路 branch 的基础信息
  int   m; // branch 的总数
  int* d_branch_typs = nullptr;
  int* d_branch_us = nullptr;
  int* d_branch_vs = nullptr;
  double *d_branch_vals = nullptr;
  
  // 用于 RHS 计算的额外信息
  int num_nonzero_nodes;
  int num_v_sources;
};

void opCuDssHandleInit(OpCuDssHandle &h, const at::Tensor &branch_typ,
                         const at::Tensor &branch_u, const at::Tensor &branch_v,
                         const at::Tensor &branch_val) {

  CUDSS_CHECK(cudssCreate(&h.handle));
  CUDSS_CHECK(cudssConfigCreate(&h.config));
  CUDSS_CHECK(cudssDataCreate(h.handle, &h.data));

  size_t m = branch_typ.size(0);
  const int *typs = branch_typ.data_ptr<int>();
  const int *us = branch_u.data_ptr<int>();
  const int *vs = branch_v.data_ptr<int>();
  const double *vals = branch_val.data_ptr<double>();
  std::vector<Triplet> ts; 
  std::vector<double> rhs;

  initTriplets(ts, m, typs, us, vs, vals);
  initRhs(rhs, m, typs, us, vs, vals);

  h.n = rhs.size();
  h.nnz = ts.size();

  h.m = m;
  if (h.n == 0) return;

  // === 1. 在CPU上构建CSR格式 ===
  std::vector<int> h_ia(h.n + 1);
  std::vector<int> h_ja(h.nnz);
  std::vector<int> h_branch_indices_for_nnz(h.nnz); // 新的映射向量
  std::vector<double> h_signs_for_nnz(h.nnz); // <--- 新增 CPU 上的符号向量

  std::vector<double> h_a(h.nnz);

  h_ia[0] = 0;
  int r = 0;
  for (size_t i = 0; i < ts.size(); i++) {
    while (ts[i].row > r) {
      h_ia[++r] = i;
    }
    h_ja[i] = ts[i].col;
  }
  while (h.n > r) {
    h_ia[++r] = static_cast<int>(ts.size());
  }

  // --- 分配GPU内存并上传结构 ---
  CUDA_CHECK(cudaMalloc((void **)&h.d_ia, (h.n + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&h.d_ja, h.nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&h.d_a,  h.nnz * sizeof(double))); // 为数值预留空间
  CUDA_CHECK(cudaMalloc((void **)&h.d_b,  h.n * sizeof(double)));   // 为RHS预留空间
  CUDA_CHECK(cudaMalloc((void **)&h.d_x,  h.n * sizeof(double)));   // 为解预留空间

  // CUDA_CHECK(cudaMalloc((void **)&h.d_branch_indices_for_nnz, h.nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&h.d_signs_for_nnz, h.nnz * sizeof(double))); // <--- 为符号数组分配显存
  CUDA_CHECK(cudaMalloc((void **)&h.d_branch_typs, h.m * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&h.d_branch_us,   h.m * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&h.d_branch_vs,   h.m * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&h.d_branch_vals, h.m * sizeof(double)));
    

  // --- 拷贝数据到GPU ---
  CUDA_CHECK(cudaMemcpy(h.d_ia, h_ia.data(), (h.n + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(h.d_ja, h_ja.data(), h.nnz * sizeof(int), cudaMemcpyHostToDevice));
  // CUDA_CHECK(cudaMemcpy(h.d_branch_indices_for_nnz, h_branch_indices_for_nnz.data(), h.nnz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(h.d_signs_for_nnz, h_signs_for_nnz.data(), h.nnz * sizeof(double), cudaMemcpyHostToDevice)); // <--- 拷贝符号数组


  CUDA_CHECK(cudaMemcpy(h.d_branch_typs, branch_typ.data_ptr<int>(), h.m * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(h.d_branch_us,   branch_u.data_ptr<int>(), h.m * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(h.d_branch_vs,   branch_v.data_ptr<int>(), h.m * sizeof(int), cudaMemcpyHostToDevice));

  std::vector<double> h_branch_vals(h.m);
  CUDA_CHECK(cudaMemcpy(h.d_branch_vals, branch_val.data_ptr<double>(), h.m * sizeof(double), cudaMemcpyHostToDevice));
  // --- 创建 cuDSS 矩阵描述符 ---
  CUDSS_CHECK(cudssMatrixCreateCsr(&h.matA, h.n, h.n, h.nnz, h.d_ia, NULL, h.d_ja, h.d_a, CUDA_R_32I, CUDA_C_64F, CUDSS_MTYPE_SPD, CUDSS_MVIEW_UPPER,CUDSS_BASE_ZERO));
  CUDSS_CHECK(cudssMatrixCreateDn(&h.vecB, h.n, 1, h.n, h.d_b, CUDA_C_64F, CUDSS_LAYOUT_COL_MAJOR));
  CUDSS_CHECK(cudssMatrixCreateDn(&h.vecX, h.n, 1, h.n, h.d_x, CUDA_C_64F, CUDSS_LAYOUT_COL_MAJOR));
  // --- 执行分析阶段 ---
  CUDSS_CHECK(cudssExecute(h.handle, CUDSS_PHASE_ANALYSIS, h.config, h.data, h.matA, h.vecX, h.vecB));
  // 【新增】保存 RHS Kernel 需要的参数
  h.num_nonzero_nodes = std::max(*std::max_element(us, us + m), *std::max_element(vs, vs + m));  
}

void opCuDssHandleFactorize(OpCuDssHandle &h, 
                              // const at::Tensor &branch_typ,
                              // const at::Tensor &branch_u,
                              // const at::Tensor &branch_v,
                              const at::Tensor &branch_val) {
  // const int *typs = branch_typ.data_ptr<int>();
  // const int *us = branch_u.data_ptr<int>();
  // const int *vs = branch_v.data_ptr<int>();
  size_t m = branch_val.size(0);

  const double *vals = branch_val.data_ptr<double>();
  thrust::device_vector<double> d_vals(reinterpret_cast<const double*>(vals), reinterpret_cast<const double*>(vals) + m);
  const int* d_typs_ptr = h.d_branch_typs;
  const int* d_us_ptr = h.d_branch_us;
  const int* d_vs_ptr = h.d_branch_vs;

  initTripletsGPU_thrust(
      m,
      d_typs_ptr,
      d_us_ptr,
      d_vs_ptr,
      thrust::raw_pointer_cast(d_vals.data()),
      h.num_nonzero_nodes, // <-- 从 h 中传入
      h.d_a   // <-- 获取计算出的 nnz
  );

  // Factorization
  CUDSS_CHECK(cudssExecute(h.handle, CUDSS_PHASE_FACTORIZATION, h.config, h.data, h.matA, h.vecX, h.vecB));

}

void opCuDssHandleSolve(OpCuDssHandle &h) {
    if (h.n == 0) return;
    // --- 1. 定义 CUDA Kernel 的启动配置 ---
    int threads_per_block = 1024;
    // RHS kernel 需要清零和原子加，网格大小可以设置得大一些以保证并行度
    int blocks_per_grid = (h.m + threads_per_block - 1) / threads_per_block;
    // --- 2. 启动 Kernel 直接在 GPU 上更新 RHS 向量 ---
    CUDA_CHECK(cudaMemset(h.d_b, 0, h.n * sizeof(double)));

    update_rhs_kernel<<<blocks_per_grid, threads_per_block>>>(
        h.d_b,
        h.n,
        h.d_branch_typs,
        h.d_branch_us,
        h.d_branch_vs,
        h.d_branch_vals,
        h.m,
        h.num_nonzero_nodes
    );
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    // // 1. 在 CPU (Host) 上创建一个 std::vector 来接收数据
    // std::vector<double> h_a_debug_vector(h.nnz);

    // // 2. 将数据从 GPU (h.d_b) 拷贝到 CPU (h_b_debug_vector)
    // CUDA_CHECK(cudaMemcpy(h_a_debug_vector.data(),    // 目标 (CPU)
    //                       h.d_a,                      // 源 (GPU)
    //                       h.nnz * sizeof(double), // 拷贝的字节数
    //                       cudaMemcpyDeviceToHost));   // 拷贝方向：Device -> Host

    // // 3. 打印一些值来检查
    // std::cout << "--- [DEBUG] 检查 RHS (b 向量) ---" << std::endl;
    // std::cout << h.nnz << std::endl;
    // int print_count = (h.nnz > 100000) ? 10000 : h.nnz; // 最多打印前 20 个
    // double norm_b = 0.0; // 计算 L2 范数，这是最好的检查方法

    // for (int i = 0; i < h.nnz; ++i) {
    //     double val = h_a_debug_vector[i];
    //     if (i < print_count) {
    //         std::cout << "h.d_a[" << i << "] = " << val << std::endl;
    //     }
    //     norm_b += val;
    // }
    
    // std::cout << "RHS (b 向量) 的 L2 范数: " << norm_b << std::endl;
    // std::cout << h.nnz << std::endl;
    // std::cout << "--- [DEBUG] 检查完毕 ---" << std::endl;
    // sleep(5);

    // // ==============================================================
    // // =============         调试代码结束           =================
    // // ==============================================================
    // --- 3. 调用 cuDSS Solve ---
    CUDSS_CHECK(cudssExecute(h.handle, CUDSS_PHASE_SOLVE, h.config, h.data, h.matA, h.vecX, h.vecB));

    // --- 4. 从 GPU 拷贝解到 CPU (这步是必须的，因为Python代码需要结果) ---
    h.h_x.resize(h.n);
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<double*>(h.h_x.data()), h.d_x, 
                          h.n * sizeof(double), cudaMemcpyDeviceToHost));

}

void opCuDssHandleTerminate(OpCuDssHandle &h) {
  // Terminate
  if (!h.handle) return; // 如果未创建，则不执行任何操作

  // Free device memory
  if (h.d_ia) cudaFree(h.d_ia);
  if (h.d_ja) cudaFree(h.d_ja);
  if (h.d_a)  cudaFree(h.d_a);
  if (h.d_b)  cudaFree(h.d_b);
  if (h.d_x)  cudaFree(h.d_x);

  // Destroy cuDSS objects
  if (h.matA) cudssMatrixDestroy(h.matA);
  if (h.vecB) cudssMatrixDestroy(h.vecB);
  if (h.vecX) cudssMatrixDestroy(h.vecX);
  if (h.data) cudssDataDestroy(h.handle, h.data);
  if (h.config) cudssConfigDestroy(h.config);
  if (h.handle) cudssDestroy(h.handle);
}

at::Tensor opCuDssHandleGetSolution(OpCuDssHandle &h) {
  at::Tensor sln = at::empty({h.n}, at::TensorOptions().dtype(at::kDouble));
  double *sln_ptr = sln.data_ptr<double>();
  memcpy(sln_ptr, h.h_x.data(), h.n * sizeof(double));

  return sln;
}

std::vector<at::Tensor> opCuDssHandleExport(OpCuDssHandle &h,
                                              const at::Tensor &branch_typ,
                                              const at::Tensor &branch_u,
                                              const at::Tensor &branch_v,
                                              const at::Tensor &branch_val) {
  size_t m = branch_typ.size(0);
  const int *typs = branch_typ.data_ptr<int>();
  const int *us = branch_u.data_ptr<int>();
  const int *vs = branch_v.data_ptr<int>();
  const double *vals = branch_val.data_ptr<double>();
  std::vector<Triplet> ts; 
  std::vector<double> rhs;
  initTriplets(ts, m, typs, us, vs, vals);
  initRhs(rhs, m, typs, us, vs, vals);

  int64_t n = rhs.size();
  int64_t nnz = ts.size();
  std::vector<int64_t> row_indices;
  std::vector<int64_t> col_indices;
  std::vector<double> values;
  for (int64_t i = 0; i < nnz; i++) {
    int64_t u = ts[i].row;
    int64_t v = ts[i].col;
    double d = ts[i].val;
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
  at::Tensor T_rhs = torch::tensor(rhs, torch::kDouble);
  return {mat, T_rhs};
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<OpCuDssHandle>(m, "OpCuDssHandle").def(py::init<>());
  m.def("opCuDssHandleInit", &opCuDssHandleInit, "opCuDssHandleInit");
  m.def("opCuDssHandleFactorize", &opCuDssHandleFactorize,
        "opCuDssHandleFactorize");
  m.def("opCuDssHandleSolve", &opCuDssHandleSolve, "opCuDssHandleSolve");
  m.def("opCuDssHandleTerminate", &opCuDssHandleTerminate,
        "opCuDssHandleTerminate");
  m.def("opCuDssHandleGetSolution", &opCuDssHandleGetSolution,
        "opCuDssHandleGetSolution");
  m.def("opCuDssHandleExport", &opCuDssHandleExport,
        "opCuDssHandleExport");
}
