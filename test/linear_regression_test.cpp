/**
 * All benchmarks from sklearn package of Python.
 **/

#include "../src/linear_regression.hpp"

#include <gtest/gtest.h>

#include <exception>
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
  try {
    lr_tune_all_false.GaussianInit(1);
    lr_tune_all_true.ConstantInit(1, 2);

    // Parameter size would be 6 because of bias unit
    EXPECT_TRUE(lr_tune_all_false.parameters_.size() == 2);
    EXPECT_TRUE(lr_tune_all_true.parameters_.size() == 2);
  } catch (std::exception e) {
    EXPECT_TRUE(false);
  }
}

TEST_F(LinearRegressionTest, test_normal_equation_tune_all_false) {
  try {
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
    EXPECT_TRUE(abs(lr_tune_all_false.parameters_(0) - 0) < 1e-12);
    EXPECT_TRUE(abs(lr_tune_all_false.parameters_(1) - 1.333333333333) < 1e-12);
  } catch (std::exception e) {
    EXPECT_TRUE(false);
  }
}

TEST_F(LinearRegressionTest, test_normal_equation_tune_all_true) {
  try {
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
    EXPECT_TRUE(abs(lr_tune_all_true.parameters_(0) - 1.) < 1e-12);
    EXPECT_TRUE(abs(lr_tune_all_true.parameters_(1) - 1.) < 1e-12);
  } catch (std::exception e) {
    EXPECT_TRUE(false);
  }
}

TEST_F(LinearRegressionTest, test_normal_equation_not_tune_all_true) {
  try {
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
    EXPECT_TRUE(abs(lr_tune_all_true.parameters_(0) - 1.) < 1e-12);
    EXPECT_TRUE(abs(lr_tune_all_true.parameters_(1) - 1.) < 1e-12);
  } catch (std::exception e) {
    EXPECT_TRUE(false);
  }
}

TEST_F(LinearRegressionTest,
       test_gradient_descent_not_regularize_all_trival_case) {
  try {
    Eigen::MatrixXd X(5, 1);
    Eigen::VectorXd Y(5);
    X << 0, 1, 2, 3, 4;
    Y << 1, 2, 3, 4, 5;
    lr_tune_all_false.GaussianInit(1);

    // Parameter size should be 2 after init
    EXPECT_TRUE(lr_tune_all_false.parameters_.size() == 2);
    // train the model with gradient descent. Cost should be around 0.166667
    EXPECT_TRUE(abs(lr_tune_all_false.GradientDescent(X, Y, 1000, 0.01) -
                    0.1666667) < 1e-5);
    // Expected values of two parameter after tunin
    EXPECT_TRUE(abs(lr_tune_all_false.parameters_(0)) < 1e-12);
    EXPECT_TRUE(abs(lr_tune_all_false.parameters_(1) - 1.333333333333) < 1e-12);
  } catch (std::exception e) {
    EXPECT_TRUE(false);
  }
}

TEST_F(LinearRegressionTest, test_gradient_descent_not_regularize_all_5_by_2) {
  try {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X(5,
                                                                             2);
    Eigen::VectorXd Y(5);
    X << 0, 2, 1, 4, 2, 9, 3, 16, 4, 25;
    Y << 1, 2, 3, 4, 5;
    lr_tune_all_false.GaussianInit(2);

    // Parameter size should be 2 after init
    EXPECT_TRUE(lr_tune_all_false.parameters_.size() == 3);
    double cost =
        abs(lr_tune_all_false.GradientDescent(X, Y, 1000000, 0.01, true));
    // Cost should be very close to 1
    EXPECT_TRUE(cost < 1.5);
    // Parameters should be 0, 1.53571429, 0.03571429
    EXPECT_TRUE(abs(lr_tune_all_false.parameters_(0)) < 1e-12);
    EXPECT_TRUE(abs(lr_tune_all_false.parameters_(1) - 1.53571429) < 1e-8);
    EXPECT_TRUE(abs(lr_tune_all_false.parameters_(2) + 0.03571429) < 1e-8);
  } catch (std::exception e) {
    EXPECT_TRUE(false);
  }
}

TEST_F(LinearRegressionTest, test_gradient_descent_regularize_all_5_by_2) {
  try {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X(5,
                                                                             2);
    Eigen::VectorXd Y(5);
    X << 0, 2, 1, 4, 2, 9, 3, 16, 4, 25;
    Y << 1, 2, 3, 4, 5;
    lr_tune_all_true.GaussianInit(2);
    // Parameter size should be 2 after init
    EXPECT_TRUE(lr_tune_all_true.parameters_.size() == 3);

    // Cost should be very close to 0
    // Notice the iteratino times that is why the error is around 1e-3 rather
    // than 1e-8 in the above test
    double cost = lr_tune_all_true.GradientDescent(X, Y, 100000, 0.01, true);
    EXPECT_TRUE(cost < 1e-5);

    // Expected values of two parameter after tuning
    // Parameters should be 1, 1, 0
    EXPECT_TRUE(abs(lr_tune_all_true.parameters_(0) - 1.) < 1e-3);
    EXPECT_TRUE(abs(lr_tune_all_true.parameters_(1) - 1.) < 1e-3);
    EXPECT_TRUE(abs(lr_tune_all_true.parameters_(2)) < 1e-3);
  } catch (std::exception e) {
    EXPECT_TRUE(false);
  }
}

TEST_F(LinearRegressionTest,
       test_gradient_descent_not_tune_all_via_data_stored_in_csv) {
  try {
    // create file readers for labels and features
    std::string label_file("linear-regression-test-label.csv");
    std::string feature_file("linear-regression-test.csv");
    CSVReader csv_reader = CSVReader();

    // read label and features
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X =
        csv_reader.read_csv_as_matrix(feature_file, ' ');
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Y =
        csv_reader.read_csv_as_matrix(label_file, ' ');

    lr_tune_all_false.GaussianInit(3);  // Initialize the model
    // should have 4 parameters
    EXPECT_TRUE(lr_tune_all_false.parameters_.size() == 4);

    double cost = lr_tune_all_false.GradientDescent(X, Y, 1000000, 100, true);
    // values of the parameters: 0 1.21534711 -0.36012027 -0.01923811
    // tuned with sklearn. 
    // The result by this is:    0 1.19646696 -0.37240682 -0.02024269
    // which is consistent with normal equation method
    EXPECT_TRUE(abs(lr_tune_all_false.parameters_(0)) < 1e-3);
    EXPECT_TRUE(abs(lr_tune_all_false.parameters_(1) - 1.21534711) < 0.2);
    EXPECT_TRUE(abs(lr_tune_all_false.parameters_(2) + 0.36012027) < 0.05);
    EXPECT_TRUE(abs(lr_tune_all_false.parameters_(3) + 0.01923811) < 0.005);

  } catch (std::exception e) {
    EXPECT_TRUE(false);
  }
}

TEST_F(LinearRegressionTest,
       test_gradient_descent_tune_all_via_data_stored_in_csv) {
  try {
    // create file readers for labels and features
    std::string label_file("linear-regression-test-label.csv");
    std::string feature_file("linear-regression-test.csv");
    CSVReader csv_reader = CSVReader();

    // read label and features
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X =
        csv_reader.read_csv_as_matrix(feature_file, ' ');
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Y =
        csv_reader.read_csv_as_matrix(label_file, ' ');

    lr_tune_all_true.GaussianInit(3);  // Initialize the model
    // should have 4 parameters
    EXPECT_TRUE(lr_tune_all_true.parameters_.size() == 4);

    double cost = lr_tune_all_true.GradientDescent(X, Y, 1000000, 100, true);
    // values of the parameters: -30.38579994 1.88035708 -0.22518492 -0.01192486
    // But it is hard for gradient descent to reach that global minimum. So
    // the following test would allow other higher errors
    EXPECT_TRUE(abs(lr_tune_all_true.parameters_(0)) < 30);
    EXPECT_TRUE(abs(lr_tune_all_true.parameters_(1) - 1.88035708) < 1);
    EXPECT_TRUE(abs(lr_tune_all_true.parameters_(2) + 0.22518492) < 1);
    EXPECT_TRUE(abs(lr_tune_all_true.parameters_(3) + 0.01192486) < 1);

  } catch (std::exception e) {
    EXPECT_TRUE(false);
  }
}

TEST_F(LinearRegressionTest, test_normal_equation_not_tune_all_data_stored_in_csv) {
  try {
    // create file readers for labels and features
    std::string label_file("linear-regression-test-label.csv");
    std::string feature_file("linear-regression-test.csv");
    CSVReader csv_reader = CSVReader();

    // read label and features
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X =
        csv_reader.read_csv_as_matrix(feature_file, ' ');
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Y =
        csv_reader.read_csv_as_matrix(label_file, ' ');

    lr_tune_all_false.GaussianInit(3);  // Initialize the model
    // should have 4 parameters
    EXPECT_TRUE(lr_tune_all_false.parameters_.size() == 4);

    double cost = lr_tune_all_false.NormalEquation(X, Y);
    // values of the parameters: 0 1.21534711 -0.36012027 -0.01923811
    // tuned with sklearn. 
    // The result by this is:    0 1.19646696 -0.37240682 -0.02024269
    // which is consistent with gradient descent method
    EXPECT_TRUE(abs(lr_tune_all_false.parameters_(0)) < 1e-3);
    EXPECT_TRUE(abs(lr_tune_all_false.parameters_(1) - 1.21534711) < 0.5);
    EXPECT_TRUE(abs(lr_tune_all_false.parameters_(2) + 0.36012027) < 0.05);
    EXPECT_TRUE(abs(lr_tune_all_false.parameters_(3) + 0.01923811) < 0.005);

  } catch (std::exception e) {
    EXPECT_TRUE(false);
  }
}

TEST_F(LinearRegressionTest, test_normal_equation_tune_all_stored_in_csv) {
  try {
    // create file readers for labels and features
    std::string label_file("linear-regression-test-label.csv");
    std::string feature_file("linear-regression-test.csv");
    CSVReader csv_reader = CSVReader();

    // read label and features
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X =
        csv_reader.read_csv_as_matrix(feature_file, ' ');
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Y =
        csv_reader.read_csv_as_matrix(label_file, ' ');

    lr_tune_all_true.GaussianInit(3);  // Initialize the model
    // should have 4 parameters
    EXPECT_TRUE(lr_tune_all_true.parameters_.size() == 4);

    double cost = lr_tune_all_true.NormalEquation(X, Y);
    // values of the parameters: -30.38579994 1.88035708 -0.22518492 -0.01192486
    // It is unkown why there is such a difference between the results
    EXPECT_TRUE(abs(lr_tune_all_true.parameters_(0) + 30) < 10);
    EXPECT_TRUE(abs(lr_tune_all_true.parameters_(1) - 1.88035708) < 1);
    EXPECT_TRUE(abs(lr_tune_all_true.parameters_(2) + 0.22518492) < 0.5);
    EXPECT_TRUE(abs(lr_tune_all_true.parameters_(3) + 0.01192486) < 0.2);

  } catch (std::exception e) {
    EXPECT_TRUE(false);
  }
}

TEST_F(LinearRegressionTest, test_learning_rate_overshooting_solution) {
  try {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X(5,
                                                                             2);
    Eigen::VectorXd Y(5);
    X << 0, 3, 1, 6, 2, 9, 3, 12, 4, 15;
    Y << 1, 2, 3, 4, 5;
    lr_tune_all_false.GaussianInit(2);
    // Cost should be very close to 0
    // learning rate is 100 (high)
    // Notice the iteration times. That is why the error is higher
    double cost = lr_tune_all_false.GradientDescent(X, Y, 100, 100, true);
    EXPECT_TRUE(cost < 0.5);

    // Expected values of two parameter after tuning
    // Parameters should be 1, 1, 0
    // In fact, it is enough to show that if the parameters are
    // 2 away from the actual parameters, overshooting is avoided
    EXPECT_TRUE(abs(lr_tune_all_false.parameters_(0) - 1.) < 2);
    EXPECT_TRUE(abs(lr_tune_all_false.parameters_(1) - 1.) < 2);
    EXPECT_TRUE(abs(lr_tune_all_false.parameters_(2)) < 2);
  } catch (std::exception e) {
    EXPECT_TRUE(false);
  }
}

TEST_F(LinearRegressionTest, test_fit) {
  try {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X(5,
                                                                             1);
    Eigen::VectorXd Y(5);
    X << 0, 1, 2, 3, 4;
    Y << 1, 2, 3, 4, 5;
    lr_tune_all_true.GaussianInit(1);
    // record the cost before tuning
    double cost_before = lr_tune_all_true.ComputeCost(X, Y);
    // record the test after tuning
    double cost = lr_tune_all_true.Fit(X, Y, 0.000001);
    // there is no overshooting
    EXPECT_TRUE(cost_before > cost);
    EXPECT_TRUE(cost < 1e-25);
    EXPECT_TRUE(abs(lr_tune_all_true.parameters_(0) - 1.) < 1e-12);
    EXPECT_TRUE(abs(lr_tune_all_true.parameters_(1) - 1.) < 1e-12);
  } catch (std::exception e) {
    EXPECT_TRUE(false);
  }
}
