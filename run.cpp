#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental::matrix;
using bfloat16 = sycl::ext::oneapi::bfloat16;

// parameters of inner kernel
#define SG_SZ 8

#define TM 8
#define TN SG_SZ
#define TK 16

// template struct for matrix
template <typename T, size_t NUM_ROWS, size_t NUM_COLS>
struct big_matrix {
 private:
  T *mat;

 public:
  T *get_data() { return mat; }
  void set_data(T *data) { mat = data; }
  big_matrix(T *data) : mat(data) {}
};

template <typename T1, typename T2, size_t M, size_t N, size_t K>
void matrix_multiply(big_matrix<T1, M, N> &C, big_matrix<T2, M, K> &A,
                     big_matrix<T2, K, N> &B) {
  // kernel begin
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  sycl::buffer<bfloat16, 2> bufA(A.get_data(), sycl::range<2>(M, K));
  sycl::buffer<bfloat16, 2> bufB(B.get_data(), sycl::range<2>(K, N));
  sycl::buffer<float, 2> bufC(C.get_data(), sycl::range<2>(M, N));

  sycl::queue q;
  q.submit([&](sycl::handler &cgh) {
     sycl::accessor accC(bufC, cgh, sycl::read_write);
     sycl::accessor accA(bufA, cgh, sycl::read_only);
     sycl::accessor accB(bufB, cgh, sycl::read_only);

     cgh.parallel_for(
         sycl::nd_range<2>({NDRangeM, NDRangeN * SG_SZ}, {1, 1 * SG_SZ}),
         [=](sycl::nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           // The joint matrix API has to be accessed by all
           // the workitems in a subgroup these functions will
           // be called once by the subgroup no code divergence
           // between the workitems
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sycl::sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sycl::sub_group, bfloat16, use::a, TM, TK,
                        layout::row_major>
               sub_a;
           joint_matrix<sycl::sub_group, bfloat16, use::b, TK, TN,
                        layout::row_major>
               sub_b;
           joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> sub_c;

           // fill C with zeros
           joint_matrix_fill(sg, sub_c, 1.0);
           for (int k = 0; k < K / TK; k += 1) {
             joint_matrix_load(
                 sg, sub_a, accA.get_pointer() + (sg_startx * TM) * K + k * TK,
                 K);
             joint_matrix_load(
                 sg, sub_b,
                 accB.get_pointer() + (sg_starty / SG_SZ) * TN + k * TK * N, N);
             joint_matrix_mad(sg, sub_a, sub_b, sub_c);
           }
           joint_matrix_store(sg, sub_c,
                              accC.get_pointer() + (sg_startx + TM) * N +
                                  sg_starty / SG_SZ * TN,
                              N, layout::row_major);
         });
   }).wait();
}

float get_random() {
  float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  return r;
}

int main() {
  static constexpr size_t MATRIX_M = TM * 2;
  static constexpr size_t MATRIX_N = TN * 2;
  static constexpr size_t MATRIX_K = TK * 2;
  bfloat16 A[MATRIX_M][MATRIX_K];
  bfloat16 B[MATRIX_K][MATRIX_N];
  float C[MATRIX_M][MATRIX_N];
  float D[MATRIX_M][MATRIX_N];

  // init matrices
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_K; j++) {
      A[i][j] = bfloat16(get_random());
    }
  }
  for (int i = 0; i < MATRIX_K; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      B[i][j] = bfloat16(get_random());
    }
  }
  // Init C, D with 1.0
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      C[i][j] = 1.0;
      D[i][j] = 1.0;
    }
  }

  // reference implementation
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      for (int k = 0; k < MATRIX_K; k++) {
        D[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  // create structs to pass around
  big_matrix<float, MATRIX_M, MATRIX_N> MC((float *)&C);
  big_matrix<bfloat16, MATRIX_M, MATRIX_K> MA((bfloat16 *)&A);
  big_matrix<bfloat16, MATRIX_K, MATRIX_N> MB((bfloat16 *)&B);
  matrix_multiply(MC, MA, MB);

  // verify correctness
  bool res = true;
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      if ((fabs(C[i][j] - D[i][j])) > 1e-5) res = false;
    }
  }
  std::cout << (res ? "passed" : "failed") << std::endl;
  return !res;
}