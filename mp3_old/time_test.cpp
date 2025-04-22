#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "dual_number.h"

float function_float(float x) {

  return std::sin(x)*std::cos(x);
  //return std::sin(std::cos(x)) * x + std::exp(x) + std::log(std::sin(x) + 2.0f);
}

dual_num function_dual(const dual_num &x) {

  return sin(x)*cos(x);
  //return sin(cos(x)) * x + exp(x) + ln(sin(x) + dual_num(2.0f));
}


int main() {
  const int N = 1000000;
  const int repetitions = 10;
  std::vector<float> input_float(N);
  std::vector<dual_num> input_dual(N);

  for (int i = 0; i < N; ++i) {
    float x = static_cast<float>(i) / N * 20.0f - 10.0f;  // from -1 to 1
    input_float[i] = x;
    input_dual[i] = dual_num(x, 1.0f);
  }

  // Time float version
  auto start_float = std::chrono::high_resolution_clock::now();
  volatile float float_sum = 0.0f;
  for (int r = 0; r < repetitions; ++r) {
    for (float x : input_float) {
      float_sum += function_float(x);
    }
  }
  auto end_float = std::chrono::high_resolution_clock::now();
  double time_float = std::chrono::duration<double, std::milli>(end_float - start_float).count() / repetitions;

  // Time dual version
  auto start_dual = std::chrono::high_resolution_clock::now();
  volatile float dual_sum = 0.0f;
  for (int r = 0; r < repetitions; ++r) {
    for (const auto &x : input_dual) {
      dual_sum += function_dual(x).val;
    }
  }
  auto end_dual = std::chrono::high_resolution_clock::now();
  double time_dual = std::chrono::duration<double, std::milli>(end_dual - start_dual).count() / repetitions;

  // Report results
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Float avg time: " << time_float << " ms\n";
  std::cout << "Dual  avg time: " << time_dual << " ms\n";

  double overhead = ((time_dual - time_float) / time_float) * 100.0;
  std::cout << "Overhead: " << overhead << " %\n";

  return 0;
}