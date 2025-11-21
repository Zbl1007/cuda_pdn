// 核心头文件
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <vector>
#include <complex>
#include <algorithm>
#include <memory.h>

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


// 宏定义部分保持不变
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

// =========================================================================
// ==              【新增】GPU 调试专用打印函数                          ==
// =========================================================================
void print_gpu_array(const char* name, const cuDoubleComplex* d_data, int size) {
    if (size == 0) {
        std::cout << "Debug print: Array '" << name << "' is empty." << std::endl;
        return;
    }

    // 1. 在 CPU (Host) 上创建一个临时 vector
    std::vector<cuDoubleComplex> h_data(size);

    // 2. 将数据从 GPU (Device) 拷贝到 CPU (Host)
    CUDA_CHECK(cudaMemcpy(
        h_data.data(),
        d_data,
        size * sizeof(cuDoubleComplex),
        cudaMemcpyDeviceToHost
    ));

    // 3. 打印内容
    std::cout << "--- Debug Print: Contents of " << name << " (size=" << size << ") ---" << std::endl;
    // 为了避免刷屏，我们只打印前 10 个和后 10 个元素
    int limit = 10;
    for (int i = 0; i < size; ++i) {
        if (i < limit || i >= size - limit) {
            std::cout << name << "[" << i << "] = ("
                      << h_data[i].x << ", " << h_data[i].y << "j)" << std::endl;
        }
        if (i == limit && size > limit * 2) {
            std::cout << "..." << std::endl;
        }
    }
    std::cout << "--- End of Debug Print ---" << std::endl;
}

// Triplet 和 coalesce 函数保持不变
struct Triplet {
  int row, col;
  std::complex<double> val;
  int idx; // 新增：记录这个 triplet 来自哪个 branch
  double sign; // <--- 新增：用于存储 +1.0 或 -1.0

  bool operator<(const Triplet &other) const {
    return row < other.row || (row == other.row && col < other.col);
  }
};


struct Triplet_GPU {
  int row, col;
  cuDoubleComplex val;
  int idx; // 新增：记录这个 triplet 来自哪个 branch
  double sign; // <--- 新增：用于存储 +1.0 或 -1.0

  // bool operator<(const Triplet &other) const {
  //   return row < other.row || (row == other.row && col < other.col);
  // }
};



//定义一个与cuDoubleComplex类似的类型,atomicAdd不能直接用于cuDoubleComplex
struct complex_double_pod {
    double x;
    double y;
};

//自定义累加器
struct complex_sum_atomic {
    __device__ void operator()(complex_double_pod* value, complex_double_pod const& new_value){
        atomicAdd(&(value->x), new_value.x);
        atomicAdd(&(value->y), new_value.y);
    }
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
    if (typ == BR_V) { // 电压源
        int u = us[i] - 1;
        int v = vs[i] - 1;
        if (u >= 0) local_count++;
        if (v >= 0) local_count++;
    } else if (typ == BR_G || typ == BR_R || typ == BR_C || typ == BR_L ||
             typ == BR_XC || typ == BR_XL) { // 无源元件 (G, R, C, L, ...)
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
    const cuDoubleComplex* vals,
    double freq,
    Triplet_GPU* out_triplets, // 输出：紧凑的三元组数组
    const int* d_offsets, // 输入：每个 branch 的写入偏移量
    int num_nonzero_nodes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    // 1. 获取这个线程的“基地址”
    int my_base_offset = d_offsets[i];
    int local_write_count = 0; // 本地（私有）计数器

    // ... (复数 g 的计算逻辑保持不变) ...
    cuDoubleComplex jomega = make_cuDoubleComplex(0.0, 2.0 * PI * freq);
    cuDoubleComplex g = make_cuDoubleComplex(0.0, 0.0);
    int typ = typs[i];

    if (typ == BR_G) { g = vals[i]; }
    else if (typ == BR_R) { g = cuCdiv(make_cuDoubleComplex(1.0, 0.0), vals[i]); }
    else if (typ == BR_C) { g = cuCmul(jomega, vals[i]); }
    else if (typ == BR_L) { g = cuCdiv(make_cuDoubleComplex(1.0, 0.0), cuCmul(jomega, vals[i])); }
    else if (typ == BR_XC) { g = cuCdiv(jomega, vals[i]); }
    else if (typ == BR_XL) { g = cuCmul(cuCdiv(make_cuDoubleComplex(1.0, 0.0), jomega), vals[i]); }
    else if (typ == BR_V) { g = make_cuDoubleComplex(1.0, 0.0); }
    else { return; } // 跳过 BR_I (typ=2) 和其他

    
    if (typ != 1) { // 无源元件: produce up to 3 entries
        int u = min(us[i], vs[i]) - 1;
        int v = max(us[i], vs[i]) - 1;

        if (u >= 0) {
            int pos = my_base_offset + local_write_count; // 精确索引
            out_triplets[pos].row = u;
            out_triplets[pos].col = u;
            out_triplets[pos].val = g;
            out_triplets[pos].idx = i;
            out_triplets[pos].sign = +1.0;
            local_write_count++;
        }
        if (v >= 0) {
            int pos = my_base_offset + local_write_count;
            out_triplets[pos].row = v;
            out_triplets[pos].col = v;
            out_triplets[pos].val = g;
            out_triplets[pos].idx = i;
            out_triplets[pos].sign = +1.0;
            local_write_count++;
        }
        if (u >= 0 && v >= 0) {
            int pos = my_base_offset + local_write_count;
            out_triplets[pos].row = u;
            out_triplets[pos].col = v;
            out_triplets[pos].val = make_cuDoubleComplex(-g.x, -g.y);
            out_triplets[pos].idx = i;
            out_triplets[pos].sign = -1.0;
            local_write_count++;
        }
    } else { // 电压源
        int u = us[i] - 1;
        int v = vs[i] - 1;
        if (u >= 0) {
            int pos = my_base_offset + local_write_count;
            out_triplets[pos].row = u;
            out_triplets[pos].col = i + num_nonzero_nodes;
            out_triplets[pos].val = make_cuDoubleComplex(1.0, 0.0);
            out_triplets[pos].idx = i;
            out_triplets[pos].sign = +1.0;
            local_write_count++;
        }
        if (v >= 0) {
            int pos = my_base_offset + local_write_count;
            out_triplets[pos].row = v;
            out_triplets[pos].col = i + num_nonzero_nodes;
            out_triplets[pos].val = make_cuDoubleComplex(-1.0, 0.0);
            out_triplets[pos].idx = i;
            out_triplets[pos].sign = -1.0;
            local_write_count++;
        }
    }
    // (不再需要 'for' 循环来写 -1)
}


// --- (pack_key, unpack_key, ComplexAdd 保持不变) ---
__host__ __device__ inline uint64_t pack_key(int row, int col) {
    return ( (uint64_t)( (uint32_t)row ) << 32 ) | (uint64_t)( (uint32_t)col );
}
__host__ __device__ inline void unpack_key(uint64_t key, int &row, int &col) {
    row = (int)( (key >> 32) & 0xFFFFFFFFu );
    col = (int)( key & 0xFFFFFFFFu );
}
struct ComplexAdd {
    __host__ __device__ cuDoubleComplex operator()(const cuDoubleComplex &a, const cuDoubleComplex &b) const {
        return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
    }
};


// -----------------------------------------------------------------
// ----------------- 主机端函数 (已重构) ---------------------------
// -----------------------------------------------------------------
int initTripletsGPU_thrust(
    int m,
    const int* d_typs,
    const int* d_us,
    const int* d_vs,
    const cuDoubleComplex* d_vals,
    double freq,
    int h_maxnode, // <-- 从 Factorize 函数传入
    cuDoubleComplex* d_out_vals_ptr // <-- 目标 GPU 指针 (即 h.d_a)
    )
{
    // ---------------------------------------------------
    // --- 步骤 1: 【已修复】在 GPU 上计算 h_maxnode ---
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
    
    int threads = 512;
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

    // if (h_count == 0) {
    //     // 没有三元组，提前退出
    //     out_rows.clear(); out_cols.clear(); out_vals.clear();
    //     return;
    // }

    // ---------------------------------------------------
    // --- 步骤 4: 分配 *精确* 内存并执行第 2 趟 (写入) ---
    // ---------------------------------------------------
    Triplet_GPU* d_raw = nullptr;
    cudaMalloc(&d_raw, sizeof(Triplet_GPU) * h_count); // <-- 分配紧凑的内存

    build_triplets_kernel<<<blocks, threads>>>(
        m, d_typs, d_us, d_vs, d_vals, freq, 
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
    thrust::device_vector<cuDoubleComplex> d_vals_vec(h_count);

    // 3) 启动打包内核 (大小为 h_count)
    int b2 = (h_count + threads - 1)/threads;
    extern __global__ void fill_triplet_arrays_kernel(const Triplet_GPU* raw, uint64_t* keys, cuDoubleComplex* vals, int n);
    fill_triplet_arrays_kernel<<<b2, threads>>>(d_raw, thrust::raw_pointer_cast(d_keys.data()), thrust::raw_pointer_cast(d_vals_vec.data()), h_count);
    cudaDeviceSynchronize();

    // 4) 排序
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_vals_vec.begin());

    // 5) 归约
    thrust::device_vector<uint64_t> d_out_keys(h_count);
    thrust::device_vector<cuDoubleComplex> d_out_vals(h_count);
    auto new_end = thrust::reduce_by_key(
        d_keys.begin(), d_keys.end(),
        d_vals_vec.begin(),
        d_out_keys.begin(),
        d_out_vals.begin(),
        thrust::equal_to<uint64_t>(),
        ComplexAdd()
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
    return out_n;

    // (d_counts, d_offsets, d_keys 等都是 device_vector，会自动释放)
}

// -----------------------------------------------------------------
// ----------------- 打包内核 (已简化) -----------------------------
// -----------------------------------------------------------------
// 这个内核现在变得非常简单，因为它不需要再检查 -1 了
__global__ void fill_triplet_arrays_kernel(const Triplet_GPU* raw, uint64_t* keys, cuDoubleComplex* vals, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    Triplet_GPU t = raw[idx];
    
    // 所有三元组都是有效的，直接打包
    keys[idx] = pack_key(t.row, t.col);
    vals[idx] = t.val;
}

// =========================================================================
// ==              CUDA KERNELS (直接在 GPU 上计算)                     ==
// =========================================================================

// // GPU版initTriplets
// __global__ void build_triplets_kernel(
//     Triplet *triplets,
//     int *counter,
//     const int *typs,
//     const int *us,
//     const int *vs,
//     const cuDoubleComplex *vals,
//     int m,
//     int num_nonzero_nodes,
//     double freq
// ){
//     const int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= m) return;
//     int branch_idx = d_branch_indices_for_nnz[idx];

//     auto insert_or_add = [&](int row, int col, complex_double_pod val){
//         if(row < 0 || col < 0) return;
//         int64_t key = (static_cast<int16_t>(row) << 32) | static_cast<int64_t>(col);

//         //这里会自动处理,如果不存在则插入,存在则调用complex_sum_atomic 累加
//         map_view.insert_or_assign(key, val);
//     }

//     int branch_type = d_branch_typs[branch_idx];
//     int u = std::min(d_us[branch_idx], d_vs[branch_idx]) - 1;
//     int v = std::max(d_us[branch_idx], d_vs[branch_idx]) - 1;

//     cuDoubleComplex base_val = d_branch_vals[branch_idx];
    
//     cuDoubleComplex jomega = make_cuDoubleComplex(0.0, 2.0 * PI * freq);
//     cuDoubleComplex val_out = make_cuDoubleComplex(0.0, 0.0);

//     int typ = typs[i];
// }

__global__ void update_matrix_values_kernel(
    cuDoubleComplex* d_a,                  // 要更新的矩阵数值 (输出)
    int nnz,                               // 矩阵非零元数量
    const int* d_branch_indices_for_nnz,   // 映射：每个非零元来自哪个 branch
    double freq,                           // 当前频率
    const int* d_branch_typs,              // branch 类型
    const int* d_us,              // 节点u
    const int* d_vs,              // 节点v
    const cuDoubleComplex* d_branch_vals,   // branch 基础值 (电容值/电感值等)
    const double* d_signs_for_nnz           // <--- 新增参数：符号指令单
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;
    int branch_idx = d_branch_indices_for_nnz[idx];



    int branch_type = d_branch_typs[branch_idx];
    int u = std::min(d_us[branch_idx], d_vs[branch_idx]) - 1;
    int v = std::max(d_us[branch_idx], d_vs[branch_idx]) - 1;

    cuDoubleComplex base_val = d_branch_vals[branch_idx];
    
    cuDoubleComplex jomega = make_cuDoubleComplex(0.0, 2.0 * PI * freq);
    cuDoubleComplex val_out = make_cuDoubleComplex(0.0, 0.0);

    switch (branch_type) {
        case BR_V:
            val_out = make_cuDoubleComplex(1.0, 0.0); // 假设电压源相关值为 +/- 1
            // d_a[idx] = make_cuDoubleComplex(d_signs_for_nnz[idx], 0.0);
            // return;
            break;
        case BR_G:
            val_out = base_val;
            break;
        case BR_R:
            val_out = make_cuDoubleComplex(1.0 / base_val.x, 0.0);
            printf("base_val.x:%.12lf  base_val.y:%.12lf",base_val.x, base_val.y);
            break;
        case BR_C:
            // val_out = jomega * base_val
            val_out = cuCmul(jomega, base_val);
            break;
        case BR_L:
             // val_out = 1.0 / (jomega * base_val)
            val_out = cuCdiv(make_cuDoubleComplex(1.0, 0.0), cuCmul(jomega, base_val));
            break;
        case BR_XC:
             // val_out = jomega / base_val
            val_out = cuCdiv(jomega, base_val);
            break;
        case BR_XL:
             // val_out = 1.0 / jomega * base_val
            val_out = cuCmul(cuCdiv(make_cuDoubleComplex(1.0, 0.0), jomega), base_val);
            break;
    }
    
    // 注意：您需要在 initTriplets 中处理好 +/- 符号
    // 这个 kernel 只计算导纳的绝对值
    // d_a[idx] = val_out;
    printf("branch_type:%d  val_out = (%.12lf, %.12lf)\n",branch_type, val_out.x, val_out.y);

    double sign = d_signs_for_nnz[idx];
    d_a[idx] = make_cuDoubleComplex(val_out.x * sign, val_out.y * sign);
}

__global__ void update_rhs_kernel(
    cuDoubleComplex* d_b,                 // 要更新的 RHS 向量 (输出)
    int n,                                // RHS 向量长度
    const int* d_branch_typs,             // branch 类型
    const int* d_branch_us,               // branch u 节点
    const int* d_branch_vs,               // branch v 节点
    const cuDoubleComplex* d_branch_vals, // branch 值
    int m,                                // branch 总数
    int num_nonzero_nodes,                // 非零节点数
    int num_v_sources                     // 电压源数量
) {
    // // 首先将向量清零
    // for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
    //     d_b[i] = make_cuDoubleComplex(0.0, 0.0);
    // }
    // __syncthreads(); // 确保所有线程都完成了清零

    // 每个线程处理一个 branch
    int branch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (branch_idx >= m) return;

    int type = d_branch_typs[branch_idx];
    
    if (type == BR_I) { // 电流源
        int u = d_branch_us[branch_idx] - 1;
        int v = d_branch_vs[branch_idx] - 1;
        cuDoubleComplex val = d_branch_vals[branch_idx];
        if (u >= 0) atomicAdd(&d_b[u].x, -val.x), atomicAdd(&d_b[u].y, -val.y);
        if (v >= 0) atomicAdd(&d_b[v].x,  val.x), atomicAdd(&d_b[v].y,  val.y);
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


// initTriplets 和 initRhs 函数也保持不变，因为它们在 CPU 上构建矩阵
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
      ts.push_back({u, u, g, i, +1.0});  //对角线值为正，非对角线为负
    if (v >= 0)
      ts.push_back({v, v, g, i, +1.0});
    if (u >= 0 && v >= 0)
      ts.push_back({u, v, -g, i, -1.0});
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
  // 修正后的电抗公式
  for (int i = xc_begin; i < xl_begin; i++)
    make(i, jomega / vals[i]);
    // make(i, std::complex<double>{0., 1.} / vals[i]);
  for (int i = xl_begin; i < m; i++)
    make(i, 1. / jomega * vals[i]);
    // make(i, std::complex<double>{0., -1.} / vals[i]);

  // Voltage Source
  for (int i = 0; i < i_begin; i++) {
    int u = us[i] - 1;
    int v = vs[i] - 1;
    if (u >= 0)
      ts.push_back({u, i + num_nonzero_nodes, 1., i, +1.0});
    if (v >= 0)
      ts.push_back({v, i + num_nonzero_nodes, -1., i, -1.0});
    // // 为非对称矩阵添加另一侧
    // if (u >= 0)
    //     ts.push_back({i + num_nonzero_nodes, u, 1.});
    // if (v >= 0)
    //     ts.push_back({i + num_nonzero_nodes, v, -1.});
  }
  coalesce(ts);
}


void initRhs(std::vector<std::complex<double>> &rhs, size_t m, const int *typs,
             const int *us, const int *vs, const std::complex<double> *vals) {
  int num_nonzero_nodes =
      std::max(*std::max_element(us, us + m), *std::max_element(vs, vs + m));
  int i_begin = std::lower_bound(typs, typs + m, BR_I) - typs;
  int g_begin = std::lower_bound(typs, typs + m, BR_G) - typs;
  rhs.assign(num_nonzero_nodes + i_begin, 0.);

  for (int i = i_begin; i < g_begin; i++) {
    int u = us[i] - 1;
    int v = vs[i] - 1;
    if (u >= 0)
      rhs[u] -= vals[i];
    if (v >= 0)
      rhs[v] += vals[i];
  }
  for (int i = 0; i < i_begin; i++) {
    rhs[i + num_nonzero_nodes] = vals[i];
  }
}

// =========================================================================
// ==              新的 cuSOLVER 句柄和函数                              ==
// =========================================================================

struct AcCuDssHandle {
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
  std::vector<std::complex<double>> h_x; // Solution


  // --- GPU 上的 CSR 矩阵和向量 ---
  int *d_ia = nullptr;
  int *d_ja = nullptr;
  cuDoubleComplex *d_a = nullptr;
  cuDoubleComplex *d_x = nullptr;
  cuDoubleComplex *d_b = nullptr;

  // === 新增：在 GPU 上存储电路基础信息 ===
  // 映射关系：h.ts 中第 k 个元素的值，应该由哪个 branch 计算得到
  int* d_branch_indices_for_nnz = nullptr;
  double* d_signs_for_nnz = nullptr;

  
  // 电路 branch 的基础信息
  int   m; // branch 的总数
  int* d_branch_typs = nullptr;
  int* d_branch_us = nullptr;
  int* d_branch_vs = nullptr;
  cuDoubleComplex *d_branch_vals = nullptr;
  
  // 用于 RHS 计算的额外信息
  int num_nonzero_nodes;
  int num_v_sources;

  // std::vector<Triplet> ts;
  // std::vector<std::complex<double>> rhs;
};

void acCuDssHandleInit(AcCuDssHandle &h, const at::Tensor &branch_typ,
                         const at::Tensor &branch_u, const at::Tensor &branch_v,
                         const at::Tensor &branch_val) {
  CUDSS_CHECK(cudssCreate(&h.handle));
  CUDSS_CHECK(cudssConfigCreate(&h.config));
  CUDSS_CHECK(cudssDataCreate(h.handle, &h.data));

  size_t m = branch_typ.size(0);
  const int *typs = branch_typ.data_ptr<int>();
  const int *us = branch_u.data_ptr<int>();
  const int *vs = branch_v.data_ptr<int>();
  const std::complex<double> *vals = reinterpret_cast<const std::complex<double> *>(branch_val.data_ptr<c10::complex<double>>());

  std::vector<Triplet> ts; 
  std::vector<std::complex<double>> rhs;

  initTriplets(ts, m, typs, us, vs, vals, 1.);
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

  std::vector<std::complex<double>> h_a(h.nnz);
  
  h_ia[0] = 0;
  int r = 0;
  for (size_t i = 0; i < ts.size(); i++) {
    while (ts[i].row > r) {
      h_ia[++r] = i;
    }
    h_ja[i] = ts[i].col;
    h_branch_indices_for_nnz[i] = ts[i].idx; // 记录映射
    h_signs_for_nnz[i] = ts[i].sign; // <--- 从三元组中提取符号

  }
  while (h.n > r) {
    h_ia[++r] = static_cast<int>(ts.size());
  }
    // for (size_t i = 0; i < ts.size(); i++) {
    //     std::cout << "h_a["<<i<<"]:" << ts[i].val.real()<< ", " << ts[i].val.imag() << std::endl;
    // }
    // for (int i = 0; i < ts.size(); i++) {
    //     std::cout << "h_ia[" << i << "] = " << h_ia[i] << std::endl;
    // }
    // for (size_t i = 0; i < h.n + 1; i++) {
    //     std::cout << "h_ja[" << i << "] = " << h_ja[i] << std::endl;
    // }
    // exit(0);
  

  // --- 分配GPU内存并上传结构 ---
  CUDA_CHECK(cudaMalloc((void **)&h.d_ia, (h.n + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&h.d_ja, h.nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&h.d_a,  h.nnz * sizeof(cuDoubleComplex))); // 为数值预留空间
  CUDA_CHECK(cudaMalloc((void **)&h.d_b,  h.n * sizeof(cuDoubleComplex)));   // 为RHS预留空间
  CUDA_CHECK(cudaMalloc((void **)&h.d_x,  h.n * sizeof(cuDoubleComplex)));   // 为解预留空间

  CUDA_CHECK(cudaMalloc((void **)&h.d_branch_indices_for_nnz, h.nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&h.d_signs_for_nnz, h.nnz * sizeof(double))); // <--- 为符号数组分配显存
  CUDA_CHECK(cudaMalloc((void **)&h.d_branch_typs, h.m * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&h.d_branch_us,   h.m * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&h.d_branch_vs,   h.m * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&h.d_branch_vals, h.m * sizeof(cuDoubleComplex)));
    

  // --- 拷贝数据到GPU ---
  CUDA_CHECK(cudaMemcpy(h.d_ia, h_ia.data(), (h.n + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(h.d_ja, h_ja.data(), h.nnz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(h.d_branch_indices_for_nnz, h_branch_indices_for_nnz.data(), h.nnz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(h.d_signs_for_nnz, h_signs_for_nnz.data(), h.nnz * sizeof(double), cudaMemcpyHostToDevice)); // <--- 拷贝符号数组


  CUDA_CHECK(cudaMemcpy(h.d_branch_typs, branch_typ.data_ptr<int>(), h.m * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(h.d_branch_us,   branch_u.data_ptr<int>(), h.m * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(h.d_branch_vs,   branch_v.data_ptr<int>(), h.m * sizeof(int), cudaMemcpyHostToDevice));

  std::vector<cuDoubleComplex> h_branch_vals(h.m);
    auto* src = branch_val.data_ptr<c10::complex<double>>();

    // 手动转换每个元素
    //  std::cout << h.m << std::endl;
    // for (int i = 0; i < h.m; i++) {
    //     h_branch_vals[i] = make_cuDoubleComplex(src[i].real(), src[i].imag());
    //     std::cout << "h_branch_vals["<<i<<"]:" << src[i].real()<< ", " << src[i].imag() << std::endl;
    // }

    // 然后再拷贝
    // CUDA_CHECK(cudaMemcpy(
    //     h.d_branch_vals,
    //     h_branch_vals.data(),
    //     h.m * sizeof(cuDoubleComplex),
    //     cudaMemcpyHostToDevice
    // ));
  CUDA_CHECK(cudaMemcpy(h.d_branch_vals, branch_val.data_ptr<c10::complex<double>>(), h.m * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
  // --- 创建 cuDSS 矩阵描述符 ---
  CUDSS_CHECK(cudssMatrixCreateCsr(&h.matA, h.n, h.n, h.nnz, h.d_ia, NULL, h.d_ja, h.d_a, CUDA_R_32I, CUDA_C_64F, CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_UPPER,CUDSS_BASE_ZERO));
  CUDSS_CHECK(cudssMatrixCreateDn(&h.vecB, h.n, 1, h.n, h.d_b, CUDA_C_64F, CUDSS_LAYOUT_COL_MAJOR));
  CUDSS_CHECK(cudssMatrixCreateDn(&h.vecX, h.n, 1, h.n, h.d_x, CUDA_C_64F, CUDSS_LAYOUT_COL_MAJOR));

  // --- 执行分析阶段 ---
  CUDSS_CHECK(cudssExecute(h.handle, CUDSS_PHASE_ANALYSIS, h.config, h.data, h.matA, h.vecX, h.vecB));
  // 【新增】保存 RHS Kernel 需要的参数
  h.num_nonzero_nodes = std::max(*std::max_element(us, us + m), *std::max_element(vs, vs + m));
  h.num_v_sources = std::lower_bound(typs, typs + m, BR_I) - typs;

}

void acCuDssHandleFactorize(AcCuDssHandle &h, 
                              //  const at::Tensor &branch_typ,
                              //  const at::Tensor &branch_u,
                              //  const at::Tensor &branch_v,
                               const at::Tensor &branch_val, double freq) {
    size_t m = branch_val.size(0);
    // const int *typs = branch_typ.data_ptr<int>();
    // const int *us = branch_u.data_ptr<int>();
    // const int *vs = branch_v.data_ptr<int>();
    const std::complex<double> *vals = reinterpret_cast<const std::complex<double> *>(branch_val.data_ptr<c10::complex<double>>());

    // std::vector<Triplet> ts; 
    // initTriplets(ts, m, typs, us, vs, vals, freq);
    

    // std::vector<int> h_ia(h.n + 1, 0);
    // std::vector<int> h_ja(h.nnz);
    // std::vector<std::complex<double>> h_a(h.nnz);
    // for (size_t i = 0; i < ts.size(); i++) {
    //     h_a[i] = ts[i].val; // 赋值
    // }

    // for (size_t i = 0; i < ts.size(); i++) {
    //     h_a[i] = {ts[i].val.real(), ts[i].val.imag()};
    //     std::cout << "h_a["<<i<<"]:" << ts[i].val.real()<< ", " << ts[i].val.imag() << std::endl;
    // }
    // for (int i = 0; i < ts.size(); i++) {
    //    std::cout << "h_ia[" << i << "] = " << h_ia[i] << std::endl;
    // }
    // for (size_t i = 0; i < h.n + 1; i++) {
    //     std::cout << "h_ja[" << i << "] = " << h_ja[i] << std::endl;
    // }
    // exit(0);

    // 拷贝到GPU
    // CUDA_CHECK(cudaMemcpy(h.d_a, reinterpret_cast<const cuDoubleComplex*>(h_a.data()), 
    //                     h.nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    
    // 查一下gpu上的数据



    thrust::device_vector<cuDoubleComplex> d_vals(reinterpret_cast<const cuDoubleComplex*>(vals), reinterpret_cast<const cuDoubleComplex*>(vals) + m);
    // const int* d_typs_ptr = h.d_branch_typs;
    // const int* d_us_ptr = h.d_branch_us;
    // const int* d_vs_ptr = h.d_branch_vs;
    // std::vector<int> out_rows, out_cols;
    // std::vector<cuDoubleComplex> out_vals;
    // std::cout << "Coalesced triplet count: " <<  std::endl;
    // int nnz_computed = 0; 
    // CUDA_CHECK(cudaMemcpy(h.d_branch_vals, branch_val.data_ptr<c10::complex<double>>(), h.m * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    int nnz_computed = initTripletsGPU_thrust(
        m,
        h.d_branch_typs,
        h.d_branch_us,
        h.d_branch_vs,
        thrust::raw_pointer_cast(d_vals.data()),
        freq, // freq
        h.num_nonzero_nodes, // <-- 从 h 中传入
        h.d_a   // <-- 获取计算出的 nnz
    );
    // std::cout << "Coalesced triplet count: " << out_rows.size() << std::endl;
    // for (size_t i=0;i<out_rows.size();++i) {
    //     std::cout << "(" << out_rows[i] << "," << out_cols[i] << ") = "
    //               << out_vals[i].x << " + " << out_vals[i].y << "i\n";
    // }


    // --- 步骤 3: 安全检查 ---
    if (h.nnz != nnz_computed) {
      std::cout << "FATAL ERROR: CPU nnz (" << h.nnz << ") != GPU nnz ("
                << nnz_computed << ")" << std::endl;
    }
    // if (h.nnz != nnz_computed) {
    //     throw std::runtime_error("FATAL: Matrix structure (NNZ) changed.");
    // }
    // exit(0);

    // if (h.n == 0) return;
    //  // --- 1. 定义 CUDA Kernel 的启动配置 ---
    // int threads  = 256;
    // int blocks  = (h.nnz + threads - 1) / threads;
    // // std::cout<<"ok  success!"<<std::endl;

    // // --- 2. 启动 Kernel 直接在 GPU 上更新矩阵数值 ---
    // update_matrix_values_kernel<<<blocks, threads>>>(
    //     h.d_a,
    //     h.nnz,
    //     h.d_branch_indices_for_nnz,
    //     freq,
    //     h.d_branch_typs,
    //     h.d_branch_us,
    //     h.d_branch_vs,
    //     h.d_branch_vals,
    //     h.d_signs_for_nnz
    // );


    // // 等待 Kernel 执行完成，确保数据已写入
    // cudaDeviceSynchronize(); 
    
    // // 检查 Kernel 启动和执行是否有错
    // CUDA_CHECK(cudaGetLastError()); 
    // // std::cout << "Kernel launch: Finished." << std::endl;
    // // std::cout<<"ok  success!"<<std::endl;
    // // for (size_t i = 0; i < 4; i++) {
    // //     // std::cout << "h.d_a[" << i << "] = " << h.d_a[i] << std::endl;
    // //     std::cout << "h.d_a["<<i<<"]:" << h.d_a[i].x<< ", " << h.d_a[i].y << std::endl;
    // // }
    // // exit(0);


    // // --- 2. 【核心调试步骤】调用我们的打印函数 ---
    // print_gpu_array("h.d_a", h.d_a, h.nnz);
    // exit(0);
    // // std::cout<<"okok success!"<<std::endl;
    // print_gpu_array("h.d_a", h.d_a, h.nnz);
    // exit(0);

    // CUDA_CHECK(cudaGetLastError()); // 检查 Kernel 启动是否有错
    // --- 3. 调用 cuDSS Factorization ---
    CUDSS_CHECK(cudssExecute(h.handle, CUDSS_PHASE_FACTORIZATION, h.config, h.data, h.matA, h.vecX, h.vecB));

    // CUDSS_CHECK(cudssExecute(h.handle, CUDSS_PHASE_FACTORIZATION, h.config, h.data, h.matA, h.vecX, h.vecB));
}

void acCuDssHandleSolve(AcCuDssHandle &h,
                        // const at::Tensor &branch_typ,
                        // const at::Tensor &branch_u,
                        // const at::Tensor &branch_v,
                        const at::Tensor &branch_val
    ) {
    if (h.n == 0) return;
    size_t m = branch_val.size(0);
    // const int *typs = branch_typ.data_ptr<int>();
    // const int *us = branch_u.data_ptr<int>();
    // const int *vs = branch_v.data_ptr<int>();
    const std::complex<double> *vals = reinterpret_cast<const std::complex<double> *>(branch_val.data_ptr<c10::complex<double>>());

    // initRhs(h.rhs, m, typs, us, vs, vals);
    // std::vector<cuDoubleComplex> h_rhs(h.n);
    // for(size_t i = 0; i < h.n; ++i) {
    //     h_rhs[i] = {h.rhs[i].real(), h.rhs[i].imag()};
    // }

    // // 拷贝RHS到GPU
    // CUDA_CHECK(cudaMemcpy(h.d_b, h_rhs.data(), 
    //                     h.n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    
    // CUDSS_CHECK(cudssExecute(h.handle, CUDSS_PHASE_SOLVE, h.config, h.data, h.matA, h.vecX, h.vecB));

    // //从GPU拷贝解到CPU
    // h.h_x.resize(h.n);
    // CUDA_CHECK(cudaMemcpy(reinterpret_cast<cuDoubleComplex*>(h.h_x.data()), h.d_x, 
    //                       h.n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    // // for (int i = 0; i < h.n; i++) {
    // //   std::cout << "h.h_x[" << i << "] = " << h.h_x[i] << std::endl;
    // // }

    // --- 1. 定义 CUDA Kernel 的启动配置 ---
    int threads_per_block = 512;
    // RHS kernel 需要清零和原子加，网格大小可以设置得大一些以保证并行度
    int blocks_per_grid = (h.m + threads_per_block - 1) / threads_per_block;
    // --- 2. 启动 Kernel 直接在 GPU 上更新 RHS 向量 ---
    CUDA_CHECK(cudaMemset(h.d_b, 0, h.n * sizeof(cuDoubleComplex)));

    thrust::device_vector<cuDoubleComplex> d_vals(reinterpret_cast<const cuDoubleComplex*>(vals), reinterpret_cast<const cuDoubleComplex*>(vals) + m);

    update_rhs_kernel<<<blocks_per_grid, threads_per_block>>>(
        h.d_b,
        h.n,
        h.d_branch_typs,
        h.d_branch_us,
        h.d_branch_vs,
        // h.d_branch_vals,
        // typs,
        // us,
        // vs,
        // vals,
        thrust::raw_pointer_cast(d_vals.data()),
        h.m,
        h.num_nonzero_nodes,
        h.num_v_sources
    );
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    // // ==============================================================
    // // =============         调试代码开始           =================
    // // ==============================================================

    // // 1. 在 CPU (Host) 上创建一个 std::vector 来接收数据
    // std::vector<cuDoubleComplex> h_b_debug_vector(h.n);

    // // 2. 将数据从 GPU (h.d_b) 拷贝到 CPU (h_b_debug_vector)
    // CUDA_CHECK(cudaMemcpy(h_b_debug_vector.data(),    // 目标 (CPU)
    //                       h.d_b,                      // 源 (GPU)
    //                       h.n * sizeof(cuDoubleComplex), // 拷贝的字节数
    //                       cudaMemcpyDeviceToHost));   // 拷贝方向：Device -> Host

    // // 3. 打印一些值来检查
    // std::cout << "--- [DEBUG] 检查 RHS (b 向量) ---" << std::endl;
    // std::cout << h.n << std::endl;
    // int print_count = (h.n > 100000) ? 10000 : h.n; // 最多打印前 20 个
    // double norm_b = 0.0; // 计算 L2 范数，这是最好的检查方法

    // for (int i = 0; i < h.n; ++i) {
    //     cuDoubleComplex val = h_b_debug_vector[i];
    //     if (i < print_count) {
    //         std::cout << "h.d_b[" << i << "] = (" << val.x << ", " << val.y << "j)" << std::endl;
    //     }
    //     norm_b += val.x * val.x + val.y * val.y;
    // }
    
    // norm_b = std::sqrt(norm_b);
    // std::cout << "RHS (b 向量) 的 L2 范数: " << norm_b << std::endl;
    // std::cout << h.n << std::endl;
    // std::cout << "--- [DEBUG] 检查完毕 ---" << std::endl;
    // sleep(5);

    // // ==============================================================
    // // =============         调试代码结束           =================
    // // ==============================================================

    // --- 3. 调用 cuDSS Solve ---
    CUDSS_CHECK(cudssExecute(h.handle, CUDSS_PHASE_SOLVE, h.config, h.data, h.matA, h.vecX, h.vecB));

    // --- 4. 从 GPU 拷贝解到 CPU (这步是必须的，因为Python代码需要结果) ---
    h.h_x.resize(h.n);
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<cuDoubleComplex*>(h.h_x.data()), h.d_x, 
                          h.n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

}

at::Tensor acCuDssHandleGetSolution(AcCuDssHandle &h) {
  at::Tensor sln =
      at::empty({h.n}, at::TensorOptions().dtype(at::kComplexDouble));
  std::complex<double> *sln_ptr = reinterpret_cast<std::complex<double> *>(
      sln.data_ptr<c10::complex<double>>());
  
  memcpy(sln_ptr, h.h_x.data(), h.n * sizeof(std::complex<double>));

  return sln;
}

void acCuDssHandleTerminate(AcCuDssHandle &h) {
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
  
  // Reset struct to a clean state
  h = {};
}


std::vector<at::Tensor>
acCuDssHandleExport(AcCuDssHandle &h, const at::Tensor &branch_typ,
                      const at::Tensor &branch_u, const at::Tensor &branch_v,
                      const at::Tensor &branch_val, double freq) {
    size_t m = branch_typ.size(0);
    const int *typs = branch_typ.data_ptr<int>();
    const int *us = branch_u.data_ptr<int>();
    const int *vs = branch_v.data_ptr<int>();
//   const std::complex<double> *vals =
//       reinterpret_cast<const std::complex<double> *>(branch_val.data_ptr<c10::complex<double>>());
    const c10::complex<double> *vals_c10 =
      branch_val.data_ptr<c10::complex<double>>();
    const std::complex<double> *vals =
      reinterpret_cast<const std::complex<double> *>(vals_c10);
  
    // 复用 CPU 上的三元组和 RHS 构建逻辑
    std::vector<Triplet> ts;
    std::vector<std::complex<double>> rhs_vec;

    initTriplets(ts, m, typs, us, vs, vals, freq);
    initRhs(rhs_vec, m, typs, us, vs, vals);

    int64_t n = rhs_vec.size();

    std::vector<int64_t> row_indices;
    std::vector<int64_t> col_indices;
    std::vector<std::complex<double>> values;

    for (const auto& t : ts) {
        row_indices.push_back(t.row);
        col_indices.push_back(t.col);
        values.push_back(t.val);
    }

    at::Tensor indices_tensor = torch::stack(
        {
            torch::tensor(row_indices, at::kLong),
            torch::tensor(col_indices, at::kLong),
        },
        0);

    at::Tensor values_tensor =
        at::empty({static_cast<int64_t>(values.size())},
                at::TensorOptions().dtype(torch::kComplexDouble));
    memcpy(values_tensor.data_ptr<c10::complex<double>>(), values.data(),
            values.size() * sizeof(std::complex<double>));

    at::Tensor mat =
        torch::sparse_coo_tensor(indices_tensor, values_tensor, {n, n});

    at::Tensor rhs = at::empty({static_cast<int64_t>(rhs_vec.size())},
                                at::TensorOptions().dtype(torch::kComplexDouble));
    memcpy(rhs.data_ptr<c10::complex<double>>(), rhs_vec.data(),
            rhs_vec.size() * sizeof(std::complex<double>));

    return {mat, rhs};
}


// =========================================================================
// ==              PYBIND11 模块定义                                      ==
// =========================================================================
namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<AcCuDssHandle>(m, "AcCuDssHandle")
        .def(py::init<>());
    m.def("acCuDssHandleInit", &acCuDssHandleInit, "acCuDssHandleInit");
    m.def("acCuDssHandleFactorize", &acCuDssHandleFactorize,
        "acCuDssHandleFactorize");
    m.def("acCuDssHandleSolve", &acCuDssHandleSolve, "acCuDssHandleSolve");
    m.def("acCuDssHandleGetSolution", &acCuDssHandleGetSolution,
        "acCuDssHandleGetSolution");
    m.def("acCuDssHandleTerminate", &acCuDssHandleTerminate,
        "acCuDssHandleTerminate");
    m.def("acCuDssHandleExport", &acCuDssHandleExport,
        "acCuDssHandleExport");
}