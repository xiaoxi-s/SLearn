#include "../src/activation.cpp"

#include <gtest/gtest.h>

struct ActivationTest : public ::testing::Test {
  

  virtual void SetUp() override {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat(10, 10);
  }

  virtual void TearDown() override {}
};

TEST_F(ActivationTest, test_tanh) {

}

TEST_F(ActivationTest, test_sigmoid) {

}

