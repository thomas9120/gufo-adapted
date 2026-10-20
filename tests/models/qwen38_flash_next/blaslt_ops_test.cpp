#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/blaslt.hpp"

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

void Check(hipError_t status) {
  Require(status == hipSuccess, hipGetErrorString(status));
}

struct Buffer {
  void* data{nullptr};
  explicit Buffer(std::size_t bytes) { Check(hipMalloc(&data, bytes)); }
  ~Buffer() { (void)hipFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};

// Rounds to the nearest BF16 (the test values are finite).
std::uint16_t ToBf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >>
                                    16);
}

float FromBf16(std::uint16_t bits) {
  const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
  float value = 0.0F;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

}  // namespace

int main() {
  try {
    using gufo::models::qwen38_flash_next::rocm::BlasLt;
    std::string error;
    auto warmed = BlasLt::Create(nullptr, &error);
    Require(warmed != nullptr, error);
    constexpr int max_n = 4096;
    struct Shape {
      int m, k;
      hipDataType type;
    };
    // Router and alpha/beta (F16), the indexer keys and queries (BF16), and
    // a library shape.
    for (const auto [m, k, type] :
         {Shape{96, 2560, HIP_R_16F}, Shape{513, 2560, HIP_R_16F},
          Shape{128, 2560, HIP_R_16BF}, Shape{512, 2560, HIP_R_16BF},
          Shape{33, 96, HIP_R_16F}}) {
      const bool bf16 = type == HIP_R_16BF;
      // Device bits and the exact values they hold.
      std::vector<std::uint16_t> weights(static_cast<std::size_t>(m) * k);
      std::vector<std::uint16_t> input(static_cast<std::size_t>(max_n) * k);
      std::vector<float> weight_values(weights.size());
      std::vector<float> input_values(input.size());
      unsigned rng = 123;
      for (auto [bits, values] : {std::pair{&weights, &weight_values},
                                  std::pair{&input, &input_values}}) {
        for (std::size_t i = 0; i < bits->size(); ++i) {
          rng ^= rng << 13;
          rng ^= rng >> 17;
          rng ^= rng << 5;
          const float v = static_cast<float>(int(rng % 8193) - 4096) / 4096.0F;
          const __half h = __float2half_rn(v);
          (*bits)[i] = bf16 ? ToBf16(v) : __builtin_bit_cast(std::uint16_t, h);
          (*values)[i] = bf16 ? FromBf16((*bits)[i]) : __half2float(h);
        }
      }
      Buffer a(weights.size() * sizeof(std::uint16_t));
      Buffer b(input.size() * sizeof(std::uint16_t));
      Buffer c(static_cast<std::size_t>(m) * max_n * sizeof(float));
      Check(hipMemcpy(a.data, weights.data(),
                      weights.size() * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
      Check(hipMemcpy(b.data, input.data(),
                      input.size() * sizeof(std::uint16_t),
                      hipMemcpyHostToDevice));
      if (k == 2560) {
        // Exercise both an unaligned cache boundary and a wide prefill split.
        for (const auto [total, boundary] :
             {std::pair{136, 94}, std::pair{2048, 1025}}) {
          std::vector<float> bulk(std::size_t(m) * total), split(bulk.size());
          Require(warmed->Gemm(a.data, b.data, static_cast<float*>(c.data),
                               type, m, total, k, &error),
                  error);
          Check(hipMemcpy(bulk.data(), c.data, bulk.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
          for (const auto [off, n] : {std::pair{0, boundary},
                                      std::pair{boundary, total - boundary}}) {
            Require(
                warmed->Gemm(
                    a.data,
                    static_cast<std::uint16_t*>(b.data) + std::size_t(off) * k,
                    static_cast<float*>(c.data) + std::size_t(off) * m, type, m,
                    n, k, &error),
                error);
          }
          Check(hipMemcpy(split.data(), c.data, split.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
          Require(bulk == split, "projection changed across prefill chunks");
        }
      }
      // Vary shape order and use ragged sizes between the calibrated widths.
      for (int n : {4096, 2048, 9, 1024, 512, 129, 128, 513, 256}) {
        auto fresh = BlasLt::Create(nullptr, &error);
        Require(fresh != nullptr, error);
        std::vector<float> reference(static_cast<std::size_t>(m) * n);
        std::vector<float> actual(reference.size());
        for (int pass = 0; pass < 3; ++pass) {
          auto& blas = pass == 1 ? fresh : warmed;
          Require(blas->Gemm(a.data, b.data, static_cast<float*>(c.data), type,
                             m, n, k, &error),
                  error);
          Check(hipDeviceSynchronize());
          Check(hipMemcpy(actual.data(), c.data, actual.size() * sizeof(float),
                          hipMemcpyDeviceToHost));
          if (pass == 0) {
            reference = actual;
          } else {
            Require(std::memcmp(reference.data(), actual.data(),
                                actual.size() * sizeof(float)) == 0,
                    "projection changed across instances or plan reuse");
          }
        }
        for (int sample = 0; sample < 32; ++sample) {
          const int row = (sample * 79) % n;
          const int col = (sample * 127) % m;
          double expected = 0.0;
          for (int j = 0; j < k; ++j) {
            expected +=
                static_cast<double>(weight_values[std::size_t(col) * k + j]) *
                input_values[std::size_t(row) * k + j];
          }
          const float result = actual[std::size_t(row) * m + col];
          Require(std::isfinite(result) && std::abs(result - expected) < 0.002,
                  "projection differs from the F64 reference");
        }
      }
    }
    std::cout << "PASS: F16/BF16 projection replay, chunk boundaries, ragged "
                 "shapes and F64\n";
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
