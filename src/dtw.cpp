#include <iostream>
#include <vector>
#include <algorithm>

#include "omp.h"
#include "torch/extension.h"

template<class Arg>
void print(Arg arg) {
  std::cout << arg << "\n";
}
template <class Arg, class... Args>
void print(Arg arg, Args... args) {
  static const std::string sep = " ";
  constexpr int n = sizeof...(Args); 
    std::cout << arg;
    if (n > 0) {
        std::cout << sep;
        print(args...);
    }
    else {
      std::cout << std::endl;
    }
}

// Needed for backward pass; undef for inference only module
#define FORWARD_COMPUTE_ARGMIN


/**
 * Differential argmax.
 *
 * A higher N will provide better comperability between close values.
 */
inline at::Tensor soft_argmax(const at::Tensor x, int64_t N = 1) {
  // ln(sum(exp(N(x-max(x)))))/N + max(x)
  const auto max_x = at::max(x);
  const auto exp_x = at::exp((x - max_x) * N);
  const auto Z = at::sum(exp_x);
  const auto soft_min_x = (at::log(Z) / N + max_x);
  return exp_x / Z;
}

inline at::Tensor soft_argmin(const at::Tensor x, int64_t N = 1) {
  return soft_argmax(-x, N);
}

inline at::Tensor soft_max(const at::Tensor x, int64_t N = 1) {
  // ln(sum(exp(N(x-max(x)))))/N + max(x)
  const auto max_x = at::max(x);
  const auto exp_x = at::exp((x - max_x) * N);
  const auto Z = at::sum(exp_x);
  const auto soft_min_x = (at::log(Z) / N + max_x);
  return at::log(Z) / N + max_x;
}

inline at::Tensor soft_min(const at::Tensor x, int64_t N = 1) {
  return -soft_max(-x, N);
}

template<class T, size_t N>
constexpr size_t size(T (&)[N]) { return N; }

// int64_t ind_j[5] = {0, 1, 1, 1, 2};
// int64_t ind_k[5] = {1, 0, 1, 2, 1};
int64_t ind_j[3] = {0, 1, 1};
int64_t ind_k[3] = {1, 0, 1};

std::vector<at::Tensor> dtw_forward(const at::Tensor theta,
                                    const int scaling_factor = 1) {
  const auto batch_size = theta.size(0);
  const auto m = theta.size(1);
  const auto n = theta.size(2);

  const int64_t n_steps = size(ind_j);
  const int64_t max_step = std::max(*std::max_element(ind_j, ind_j + n_steps),
                                    *std::max_element(ind_k, ind_k + n_steps));
  const auto t_ind_j = -torch::from_blob(ind_j, n_steps, torch::kLong);
  const auto t_ind_k = -torch::from_blob(ind_k, n_steps, torch::kLong);

  // Add extra space around for edge case handling. The extra space after m, n is only
  // needed for a correct Q (required for the gradient computation).
  auto V = torch::zeros({batch_size, max_step + m + 1, max_step + n + 1},
                        torch::dtype(theta.dtype()));
  V.narrow(1, max_step, m).narrow(2, 0, max_step).fill_(1e10);
  V.narrow(2, max_step, n).narrow(1, 0, max_step).fill_(1e10);

  #ifdef FORWARD_COMPUTE_ARGMIN
    auto Q =
        torch::zeros({batch_size, m + 2 * max_step, n + 2 * max_step, n_steps},
                     torch::dtype(theta.dtype()));
  #endif

  #pragma omp parallel for
  for (int i = 0; i < batch_size; ++i) {
    // DP recursion
    for (int j = max_step; j < m + max_step; ++j) {
      for (int k = max_step; k < n + max_step; ++k) {
        const auto min = soft_min(V[i].index({
                                      t_ind_j + j,
                                      t_ind_k + k,
                                  }),
                                  scaling_factor);
        V[i][j][k] = theta[i][j - max_step][k - max_step] + min.item();

        #ifdef FORWARD_COMPUTE_ARGMIN
          Q[i][j][k] = soft_argmin(V[i].index({
                                      t_ind_j + j,
                                      t_ind_k + k,
                                  }),
                                  scaling_factor);
        #endif
      }
    }
  }
  return {V.narrow(1, max_step, m).narrow(2, max_step, n), Q};
}

at::Tensor dtw_backward(const at::Tensor Q, const int scaling_factor = 1) {
  const int64_t n_steps = size(ind_j);
  const int64_t max_step = std::max(*std::max_element(ind_j, ind_j + n_steps),
                                    *std::max_element(ind_k, ind_k + n_steps));

  const int batch_size = Q.size(0);
  const int m = Q.size(1) - 2 * max_step;
  const int n = Q.size(2) - 2 * max_step;
  const auto dtype = Q.dtype();

  const auto t_ind_j = torch::from_blob(ind_j, n_steps, torch::kLong);
  const auto t_ind_k = torch::from_blob(ind_k, n_steps, torch::kLong);
  const auto t_ind_s = torch::arange(n_steps, torch::kLong);

  auto E = torch::zeros({batch_size, m + 2 * max_step, n + 2 * max_step},
                        torch::dtype(dtype));
  E.narrow(1, m + max_step, 1).narrow(2, n + max_step, 1) = 1;
  // The gradient should always flow through the last item of both sequences.
  // Therefore, find the (1, 1) step that steps from the last element past the end.
  int s = 0;
  for (; s < n_steps; ++s) {
    if ((ind_j[s] == 1) && (ind_k[s] == 1)) { break; }
  }
  Q.narrow(1, m + max_step, 1).narrow(2, n + max_step, 1) = 0;
  Q.narrow(1, m + max_step, 1).narrow(2, n + max_step, 1).narrow(3, s, 1) = 1;

  #pragma omp parallel for
  for (int i = 0; i < batch_size; ++i) {
    for (int j = m + max_step - 1; j > 0; --j) {
      for (int k = n + max_step - 1; k > 0; --k) {
        const auto Q_steps = Q[i].index({
            t_ind_j + j,
            t_ind_k + k,
            t_ind_s,
        });
        const auto E_steps = E[i].index({
            t_ind_j + j,
            t_ind_k + k,
        });
        E[i][j][k] = at::sum(Q_steps * E_steps);
      }
    }
  }

  return E.narrow(1, max_step, m).narrow(2, max_step, n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &dtw_forward, "DTW forward");
  m.def("backward", &dtw_backward, "DTW backward");
}
