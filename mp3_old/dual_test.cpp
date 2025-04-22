#include "dual_number.h"
#include <cmath>
#include <gtest/gtest.h>

class DualNumberTest : public ::testing::Test {
protected:
  float tol = 1e-6f; // Tolerance for float comparisons
};
TEST_F(DualNumberTest, DualNumberClassConstructorTesting) {
  dual_num d0;              // Default
  dual_num d1(1);        // Value only
  dual_num d2(2, 3); // Value and dual

  EXPECT_FLOAT_EQ(d0.value(), 0.0f);
  EXPECT_FLOAT_EQ(d0.dual(), 0.0f);

  EXPECT_FLOAT_EQ(d1.value(), 1.0f);
  EXPECT_FLOAT_EQ(d1.dual(), 0.0f);

  EXPECT_FLOAT_EQ(d2.value(), 2.0f);
  EXPECT_FLOAT_EQ(d2.dual(), 3.0f);
}

TEST_F(DualNumberTest, VectorUnaryForexpAndcos) {
  dual_vector x = dual_vector({dual_num(0.0f, 1.0f), dual_num(1.0f, 2.0f)});
  dual_vector vx = exp(x);
  EXPECT_NEAR(vx[0].value(), std::exp(0.0f), tol);
  EXPECT_NEAR(vx[0].dual(), std::exp(0.0f) * 1.0f, tol);
  EXPECT_NEAR(vx[1].value(), std::exp(1.0f), tol);
  EXPECT_NEAR(vx[1].dual(), std::exp(1.0f) * 2.0f, tol);

  dual_vector vcos = cos(x);
  EXPECT_NEAR(vcos[0].value(), std::cos(0.0f), tol);
  EXPECT_NEAR(vcos[0].dual(), -std::sin(0.0f) * 1.0f, tol);
  EXPECT_NEAR(vcos[1].value(), std::cos(1.0f), tol);
  EXPECT_NEAR(vcos[1].dual(), -std::sin(1.0f) * 2.0f, tol);
}

TEST_F(DualNumberTest, VectorBasicOperations) {
  dual_vector a = dual_vector({dual_num(1.0f, 1.0f), dual_num(2.0f, 2.0f)});
  dual_vector b = dual_vector({dual_num(3.0f, 0.0f), dual_num(4.0f, 1.0f)});
  dual_vector c = dual_vector({dual_num(1.0f, 0.0f), dual_num(1.0f, 2.0f)});
  // a + b * c
  dual_vector result = a + b * c;

  EXPECT_FLOAT_EQ(result[0].value(), 4.0f); // 1 + 3
  EXPECT_FLOAT_EQ(result[0].dual(), 1.0f);  // 1 + 0
  EXPECT_FLOAT_EQ(result[1].value(), 6.0f); // 2 + 4
  EXPECT_FLOAT_EQ(result[1].dual(), 11.0f);  // 2 + 1*1+4*2
}

TEST_F(DualNumberTest, CompositeFunction_CosSinPlusExpLn) {
  dual_vector x = dual_vector({dual_num(1.5f, 1.0f)});
  dual_vector y = cos(sin(x)) + exp(ln(x));

  // Manually compute expected value and derivative
  float v = 1.5f;
  float sin_v = std::sin(v);
  float cos_sin_v = std::cos(sin_v);
  float cos_v = std::cos(v);
  float ln_v = std::log(v);
  float exp_ln_v = std::exp(ln_v); // == v

  float expected_value = cos_sin_v + v;
  float expected_derivative = 
    -std::sin(sin_v) * cos_v + 1.0f;  // Chain rule: d/dx[cos(sin(x))] + d/dx[exp(ln(x))] = -sin(sin(x))·cos(x) + 1

  EXPECT_NEAR(y[0].value(), expected_value, tol);
  EXPECT_NEAR(y[0].dual(), expected_derivative, tol);
}

