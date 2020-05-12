#include "../src/linear_regression.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "CSVReader.hpp"

/*
  Linear regression test
*/
struct LinearRegressionTest : public ::testing::Test {
  linearregression::LinearRegression<double> lr_tune_all_false;
  linearregression::LinearRegression<double> lr_tune_all_true;

  virtual void SetUp() override {
    lr_tune_all_false = linearregression::LinearRegression<double>(false);
    lr_tune_all_true = linearregression::LinearRegression<double>(true);
  }

  virtual void TearDown() override {}
};

TEST_F(LinearRegressionTest, test_gaussian_init) {
  Eigen::MatrixXd X(5, 1);
  Eigen::VectorXd Y(5);
  X << 0, 1, 2, 3, 4;
  Y << 1, 2, 3, 4, 5;
  lr_tune_all_false.GaussianInit(1);

  // Parameter size would be 6 because of bias unit
  EXPECT_TRUE(lr_tune_all_false.parameters_.size() == 2);
  // the model should be usable
  EXPECT_TRUE(lr_tune_all_false.usable_);
}

TEST_F(LinearRegressionTest, test_normal_equation_tune_all_false) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X(5,
                                                                           1);
  Eigen::VectorXd Y(5);
  // Fit the function y = 4/3*x
  X << 0, 1, 2, 3, 4;
  Y << 1, 2, 3, 4, 5;
  std::cout << lr_tune_all_false.NormalEquation(X, Y) << std::endl;

  // Expect size of the parameters: should be 2
  EXPECT_TRUE(lr_tune_all_false.parameters_.size() == 2);
  // Expected values of two parameter after tuning
  EXPECT_EQ(*(lr_tune_all_false.parameters_.data()), 0);
  EXPECT_TRUE(*(lr_tune_all_false.parameters_.data() + 1) - 1.333333333333 <
              1e-12);
}

TEST_F(LinearRegressionTest, test_normal_equation_tune_all_true) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X(5,
                                                                           1);
  Eigen::VectorXd Y(5);

  // Fit the function y = x + 1
  X << 0, 1, 2, 3, 4;
  Y << 1, 2, 3, 4, 5;

  // Cost should be very close to 0
  EXPECT_TRUE(lr_tune_all_true.NormalEquation(X, Y) < 1e-25);

  // Expect size of the parameters: should be 2
  EXPECT_TRUE(lr_tune_all_true.parameters_.size() == 2);
  // Expected values of two parameter after tuning
  EXPECT_TRUE(*(lr_tune_all_true.parameters_.data()) - 1. < 1e-12);
  EXPECT_TRUE(*(lr_tune_all_true.parameters_.data() + 1) - 1. < 1e-12);
}

TEST_F(LinearRegressionTest,
       test_gradient_descent_not_regularize_all_trival_case) {
  Eigen::MatrixXd X(5, 1);
  Eigen::VectorXd Y(5);
  X << 0, 1, 2, 3, 4;
  Y << 1, 2, 3, 4, 5;
  lr_tune_all_false.GaussianInit(1);

  // Parameter size should be 2 after init
  EXPECT_TRUE(lr_tune_all_false.parameters_.size() == 2);
  // train the model with gradient descent. Cost should be around 0.166667
  lr_tune_all_false.GradientDescent(X, Y, 1000, 0.01);
  // Expected values of two parameter after tunin
  EXPECT_EQ(*(lr_tune_all_false.parameters_.data()), 0);
  EXPECT_TRUE(*(lr_tune_all_false.parameters_.data() + 1) - 1.333333333333 <
              1e-12);
}

TEST_F(LinearRegressionTest, test_gradient_descent_regularize_all_trival_case) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X(5,
                                                                           1);
  Eigen::VectorXd Y(5);
  X << 0, 1, 2, 3, 4;
  Y << 1, 2, 3, 4, 5;
  lr_tune_all_true.GaussianInit(1);
  // Parameter size should be 2 after init
  EXPECT_TRUE(lr_tune_all_true.parameters_.size() == 2);

  // Cost should be very close to 0
  EXPECT_TRUE(lr_tune_all_true.GradientDescent(X, Y, 100000, 0.01) < 1e-25);

  // Expect size of the parameters: should be 2
  EXPECT_TRUE(lr_tune_all_true.parameters_.size() == 2);
  // Expected values of two parameter after tuning
  EXPECT_TRUE(*(lr_tune_all_true.parameters_.data()) - 1 < 1e-12);
  EXPECT_TRUE(*(lr_tune_all_true.parameters_.data() + 1) - 1 < 1e-12);
}

TEST_F(LinearRegressionTest, test_gradient_descent_not_regularize_all_5_by_2) {
  Eigen::MatrixXd X(5, 2);
  Eigen::VectorXd Y(5);
  X << 0, 3, 1, 6, 2, 9, 3, 12, 4, 15;
  Y << 1, 2, 3, 4, 5;
  lr_tune_all_false.GaussianInit(2);

  // Parameter size should be 2 after init
  EXPECT_TRUE(lr_tune_all_false.parameters_.size() == 3);

  // Cost should be very close to 0
  EXPECT_TRUE(lr_tune_all_false.GradientDescent(X, Y, 100000, 0.01) < 1e-25);
  // Parameters should be 0, 0, 1/3
  EXPECT_TRUE(*(lr_tune_all_false.parameters_.data()) < 1e-12);
  EXPECT_TRUE(*(lr_tune_all_false.parameters_.data() + 1) < 1e-12);
  EXPECT_TRUE(*(lr_tune_all_false.parameters_.data() + 2) - 0.333333333333 < 1e-12);
}

TEST_F(LinearRegressionTest, test_gradient_descent_regularize_all_5_by_2) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X(5,
                                                                           2);
  Eigen::VectorXd Y(5);
  X << 0, 3, 1, 6, 2, 9, 3, 12, 4, 15;
  Y << 1, 2, 3, 4, 5;
  lr_tune_all_true.GaussianInit(2);
  // Parameter size should be 3 after init
  EXPECT_TRUE(lr_tune_all_true.parameters_.size() == 3);

  // Cost should be very close to 0
  EXPECT_TRUE(lr_tune_all_true.GradientDescent(X, Y, 10000, 0.01) < 1e-25);

  // Expected values of two parameter after tuning
  // Three parameters should be: 1, 0, 1, since the second feature is
  // linear dependent with the first feature.
  EXPECT_TRUE(*(lr_tune_all_true.parameters_.data()) - 1. < 1e-12);
  EXPECT_TRUE(*(lr_tune_all_true.parameters_.data() + 1) < 1e-12);
  EXPECT_TRUE(*(lr_tune_all_true.parameters_.data() + 2) - 1 < 1e-12);
}

TEST_F(LinearRegressionTest, test_gradient_descent_via_data_stored_in_csv) {
  // create file readers for labels and features
  std::string label_file("linear-regression-test-label.csv");
  std::string feature_file("linear-regression-test.csv");
  CSVReader csv_reader = CSVReader();
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Y =
      csv_reader.read_csv_as_matrix(label_file, ',');
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X =
      csv_reader.read_csv_as_matrix(label_file, ',');

  lr_tune_all_true.GaussianInit(3);// Initialize the model

  // should have 4 parameters
  EXPECT_TRUE(lr_tune_all_true.parameters_.size() == 4);
  // values of the parameters: 0, 1.11910281, -1.67282088,  0.06316338
  EXPECT_EQ(*(lr_tune_all_true.parameters_.data()), 0);
}

TEST_F(LinearRegressionTest, test_normal_equation_via_data_stored_in_csv) {}

TEST_F(LinearRegressionTest, test_learning_rate_overshooting_solution) {
  Eigen::MatrixXd X(5, 2);
  Eigen::VectorXd Y(5);
  X << 0, 3, 1, 6, 2, 9, 3, 12, 4, 15;
  Y << 1, 2, 3, 4, 5;
  std::cout << "Regression (tuning the bias unit) trivial case" << std::endl;
  lr_tune_all_true.GaussianInit(2);

  std::cout << "Parameters before initialize" << std::endl;
  lr_tune_all_true.PrintParameters();

  std::cout << lr_tune_all_true.GradientDescent(X, Y, 100, 100) << std::endl;

  lr_tune_all_true.PrintParameters();

  EXPECT_TRUE(false);
}
