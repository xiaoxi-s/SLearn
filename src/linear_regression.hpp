#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include <cmath>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigen>
#include <exception>
#include <iostream>
#include <random>

namespace linearregression {

/**
 * Class Name: LinearRegression
 *
 * Description:
 */
template <class Derived>
class LinearRegression {
 private:
  bool usable_;  // whether this model is usable
  double CheckLearningRate(
      Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>>
          X,
      Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, 1>> y, double, double);

  Derived GradientDescentWithThreshhold(
      Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>>
          X,
      Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, 1>> y, double threshold,
      double learning_rate, bool check_learning_rate = false,
      double lambda = 0);

 public:
  Eigen::Matrix<Derived, Eigen::Dynamic, 1> parameters_;
  bool fit_intercept_;  // Whether to fit the intercept (bias unit)

  LinearRegression();
  LinearRegression(bool fit_intercept_);
  void GaussianInit(const size_t, const double mu = 0,
                    const double sigma = 1);
  void ConstantInit(const size_t, const double);

  Derived NormalEquation(
      Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>>
          X,
      Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, 1>> y);

  Derived GradientDescent(
      Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>>
          X,
      Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, 1>> y,
      size_t iterations, double learning_rate, bool check_learning_rate = false,
      double lambda = 0);

  Derived ComputeCost(Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic,
                                               Eigen::Dynamic, Eigen::RowMajor>>
                          X,
                      Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, 1>> y,
                      double lambda = 0);

  Derived ComputeCostWithAugmentedFeature(
      Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>>
          X_plus,
      Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, 1>> y,
      double lambda = 0);

  Derived Fit(Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>>
                  X,
              Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, 1>> y,
              double threshold = 0.1, bool fit_intercept = true,
              double lambda = 0);

  Eigen::Matrix<Derived, Eigen::Dynamic, 1> Predict(
      Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>>
          X);

  void PrintParameters();
};

/**
 * Constructor of the LinearRegression class: set fit_intercept_ to false. 
 * Meaning: Do not fit theta_0
 *
 * @param Derived - template parameter of the model
 **/
template <class Derived>
LinearRegression<Derived>::LinearRegression() {
  this->usable_ = false;
  this->fit_intercept_ = false;
}

/** 
 * Constructor of the LinearRegression class: set fit_intercept_ 
 * as the parameter
 * 
 * @param Derived - template parameter of the model
 * @param fit_intercept_ - denotes whether to fit the intercept: bool
 **/
template <class Derived>
LinearRegression<Derived>::LinearRegression(bool fit_intercept_) {
  this->fit_intercept_ = fit_intercept_;
  this->usable_ = false;
}

/**
 * Initialize the values of the parameters by Gaussian distribution 
 * (mu = 0, sigmoid = 1)
 * 
 * @param Derived - template parameter of the model
 * @param size - the number of the parameters. Notice: do not the bias parameter: 
 *  const size_t
 * @param mu - the mean of Gaussian distribution: double
 * @param sigma - the standard deviation of Gaussian distribution: double
 **/
template <class Derived>
void LinearRegression<Derived>::GaussianInit(const size_t size,
                                             const double mu,
                                             const double sigma) {
  size_t size_of_parameters = size + 1;
  const Eigen::Index idx = size_of_parameters;
  parameters_ = Eigen::Matrix<Derived, Eigen::Dynamic, 1>(idx);

  std::default_random_engine generator;
  std::normal_distribution<double> distribution(mu, sigma);

  for (size_t i = 0; i < size_of_parameters; i++) {
    parameters_(i) = distribution(generator);
  }

  if (!fit_intercept_) parameters_(0) = 0;

  this->usable_ = true;
}

/**
 * Initialize the values of the parameters by Gaussian distribution 
 * (mu = 0, sigmoid = 1)
 * 
 * @param Derived - template parameter of the model
 * @param size - the number of the parameters. Notice: do not the bias parameter: 
 *  const size_t
 * @param value - the value to be set as: double
 **/
template <class Derived>
void LinearRegression<Derived>::ConstantInit(const size_t size,
                                             const double value) {
  size_t size_of_parameters = size + 1;
  const Eigen::Index idx = size_of_parameters;

  parameters_ = Eigen::Matrix<Derived, Eigen::Dynamic, 1>(idx);

  for (size_t i = 0; i < size_of_parameters; i++) {
    parameters_(i) = value;
  }

  if (!fit_intercept_) parameters_(0) = 0;

  this->usable_ = true;
}

/**
 * Use gradient descent to tune the parameter
 *
 * @param Derived - template parameter of the model
 * @param X - Feature matrix:
 *  Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic,
 *             Eigen::RowMajor>>
 * @param y - Label vector:
 *  Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, 1>>
 * @param learning_rate - Learning rate for linear regression: double
 * @param check_learning_rate - denotes whether to check learning rate to : bool
 * @param lambda - double
 *
 * @return cost - Derived
 **/
template <class Derived>
Derived LinearRegression<Derived>::GradientDescent(
    Eigen::Ref<
        Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        X,
    Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, 1>> y, size_t iterations,
    double learning_rate, bool check_learning_rate, double lambda) {
  if (!usable_) return -1;

  Eigen::Map<Eigen::Matrix<Derived, Eigen::Dynamic, 1>, 0,
             Eigen::InnerStride<1>>
      param(parameters_.data() + 1, parameters_.rows() - 1);

  Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      X_plus(y.rows(), X.cols() + 1);
  X_plus << Eigen::Matrix<Derived, Eigen::Dynamic, 1, Eigen::RowMajor>::Ones(
      y.rows()),
      X;
  size_t batch_size = X.rows();

  if (check_learning_rate)
    // check learning rate first
    learning_rate = CheckLearningRate(X, y, learning_rate, lambda);

  // begin gradient descent
  if (lambda == 0) {
    if (fit_intercept_) {
      for (size_t i = 0; i < iterations; i++) {
        parameters_ =
            parameters_ - learning_rate / batch_size *
                              (X_plus.transpose() * (X_plus * parameters_ - y));
      }

    } else {
      for (size_t i = 0; i < iterations; i++) {
        param = param - learning_rate / batch_size *
                            (X.transpose() * (X_plus * parameters_ - y));
      }
    }
  } else if (fit_intercept_) {  // Regularize theta_0

    for (size_t i = 0; i < iterations; i++) {
      parameters_ = parameters_ -
                    learning_rate / batch_size *
                        (X_plus.transpose() * (X_plus * parameters_ - y)) -
                    lambda * parameters_;
    }

  } else {  // Do not regularize theta_0

    for (size_t i = 0; i < iterations; i++) {
      param = param -
              learning_rate / batch_size *
                  (X.transpose() * (X_plus * parameters_ - y)) -
              lambda * param;
    }
  }
  return ComputeCost(X, y, lambda);
}

/**
 * Use param = (X^T * X)^(-1) * X^T * Y to initialize the parameter. It is
 * more efficent if the number of instances is small. The parameter size
 * would always be number of features + 1 (a bias unit added).
 *
 * @param Derived - template parameter of the model
 * @param X - feature matrix: Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic,
 *  Eigen::Dynamic, Eigen::RowMajor>>
 * @param y - label vector: Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic,
 *  1>>
 *
 * @return cost after tuning the parameters
 **/
template <class Derived>
Derived LinearRegression<Derived>::NormalEquation(
    Eigen::Ref<
        Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        X,
    Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, 1>> y) {
  // Initialize parameters
  parameters_ = Eigen::Matrix<Derived, Eigen::Dynamic, 1>(X.cols() + 1, 1);
  if (fit_intercept_) {  // fit intercept option is on
    Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        X_plus(y.rows(), X.cols() + 1);
    X_plus << Eigen::Matrix<Derived, Eigen::Dynamic, 1>::Ones(y.rows()), X;
    parameters_ << (X_plus.transpose() * X_plus).inverse() *
                       X_plus.transpose() * y;

    this->usable_ = true;
    return ComputeCost(X, y);
  } else {
    // if the model is usable, the bias unit would be the previous bias unit
    if (this->usable_)
      parameters_ << parameters_(0), (X.transpose() * X).inverse() * X.transpose() * y;
    else // otherwise, the bias unit would be 0
      parameters_ << 0, (X.transpose() * X).inverse() * X.transpose() * y;

    this->usable_ = true;
    return ComputeCost(X, y);
  }
}

/**
 * Compute cost of current parameter and given data set.
 *
 * @param Derived - template parameter of the model
 * @param X - feature matrix of m by n: Eigen::Ref<Eigen::Matrix<Derived,
 *  Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
 * @param y - label vector:
 *  Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, 1>>
 *
 * @return cost of prediction given the parameters. If the parameters are
 *  invalid return -1
 **/
template <class Derived>
Derived LinearRegression<Derived>::ComputeCost(
    Eigen::Ref<
        Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        X,
    Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, 1>> y, double lambda) {
  if (!usable_)  // return -1 if the model is not initiated
    return -1;

  Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      X_plus(y.rows(), X.cols() + 1);
  X_plus << Eigen::Matrix<Derived, Eigen::Dynamic, 1>::Ones(y.rows()), X;

  size_t batch_size = X.rows();

  Derived c = 0.5 / batch_size * (X_plus * parameters_ - y).transpose() *
              (X_plus * parameters_ - y);

  return c;
}

/**
 * Predict the output given a set of instances
 *
 * @param X - given feature matrix: Eigen::Ref<
 *  Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
 *
 * @return the predicted label vector: 
 **/
template <class Derived>
Eigen::Matrix<Derived, Eigen::Dynamic, 1> LinearRegression<Derived>::Predict(
    Eigen::Ref<
        Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        X) {
  Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic>
      X_plus(X.rows(), X.cols() + 1);
  X_plus << Eigen::Matrix<Derived, Eigen::Dynamic, 1>::Ones(X.rows()), X;

  return X_plus * parameters_;
}

/**
 * Use gradient descent or linear regression to tune the linear regression model
 * based on the feature number and property of the feature matrix
 *
 * @param Derived - template parameter of the model
 * @param X - feature matrix of m by n: Eigen::Ref<Eigen::Matrix<Derived,
 *  Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
 * @param y - label vector: Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic,
 *  1>>
 * @param threshold - difference between costs before and after tuning
 *  to stop fit: double
 * @param fit_intercept - denotes whether to fit the intercept: bool
 * @param lambda - regularization parameter: double
 *
 * @return cost of prediction given the parameters.
 **/
template <class Derived>
Derived LinearRegression<Derived>::Fit(
    Eigen::Ref<
        Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        X,
    Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, 1>> y, double threshould,
    bool fit_intercept, double lambda) {
  // reset the parameters in the model
  fit_intercept_ = fit_intercept;

  // test for valid input
  if (X.rows() != y.rows()) {
    return -1;
  }

  // supervised learning. not multi-model ONLY one label per instance
  if (y.cols() != 1) {
    return -1;
  }

  // choose between normal equation and gradient descent
  if (X.rows() > X.cols() && X.rows() <= 10000) {
    return NormalEquation(X, y);
  } else {
    return GradientDescentWithThreshhold(X, y, threshould, 10, true,lambda);
  }
}

/**
 * Print parameters as a column vector
 **/
template <class Derived>
void LinearRegression<Derived>::PrintParameters() {
  if (usable_) std::cout << parameters_ << std::endl;
}

/**
 * This function check the alpha rate to avoid overshooting.
 *
 * @param Derived - template parameter of the model
 * @param X - Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic,
 *  Eigen::RowMajor>>: feature matrix of m by n
 * @param y - Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, 1>>:
 *  label vector
 * @param learn_rate - double: learning rate to be checked
 * @param lambda - double: the regularization parameter
 *
 * @return a proper learning_rate (doubl)
 **/
template <class Derived>
double LinearRegression<Derived>::CheckLearningRate(
    Eigen::Ref<
        Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        X,
    Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, 1>> y,
    double learning_rate, double lambda) {
  // Initialize a new parameter vector as temporary variable
  Derived cost_before_tuning = 0;
  Derived cost_after_tuning = 0;

  Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      X_plus(y.rows(), X.cols() + 1);
  X_plus << Eigen::Matrix<Derived, Eigen::Dynamic, 1, Eigen::RowMajor>::Ones(
      y.rows()),
      X;
  size_t batch_size = X.rows();

  // create a temp parameter varible to test
  Eigen::Matrix<Derived, Eigen::Dynamic, 1> initial_param(parameters_);
  Eigen::Map<Eigen::Matrix<Derived, Eigen::Dynamic, 1>, 1,
             Eigen::InnerStride<1>>
      param(parameters_.data() + 1, initial_param.rows() - 1,
            Eigen::InnerStride<1>());

  // record the cost before tuning
  cost_before_tuning = ComputeCost(X, y, lambda);

  do {
    if (lambda == 0) {
      if (fit_intercept_) {
        parameters_ =
            parameters_ - learning_rate / batch_size *
                              (X_plus.transpose() * (X_plus * parameters_ - y));

      } else {
        param = param - learning_rate / batch_size *
                            (X.transpose() * (X_plus * parameters_ - y));
      }

    } else if (fit_intercept_) {  // Regularize theta_0
      parameters_ = parameters_ -
                    learning_rate / batch_size *
                        (X_plus.transpose() * (X_plus * parameters_ - y)) -
                    lambda * parameters_;

    } else {  // Do not regularize theta_0
      param = param -
              learning_rate / batch_size *
                  (X.transpose() * (X_plus * parameters_ - y)) -
              lambda * param;
    }
    cost_after_tuning = ComputeCost(X, y, lambda);
    if (cost_before_tuning > cost_after_tuning) {
      // overshooting do not happen
      parameters_ = initial_param;
      return learning_rate;
    } else {
      // overshooting happened
      // decrease learning rate
      learning_rate = learning_rate / 10;
      cost_before_tuning = cost_after_tuning;
    }
  } while (true);

  return learning_rate;
}

/**
 * Gradient descent based on difference between costs from two iterations.
 * Use gradient descent to tune the parameter
 *
 * @param Derived - template parameter of the model
 * @param X - Feature matrix:
 *  Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic,
 *             Eigen::RowMajor>>
 * @param y - Label vector:
 *  Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, 1>>
 * @param threshold - threshold of stopping: double
 * @param learning_rate - Learning rate for linear regression: double
 * @param check_learning_rate - denotes whether to check learning rate to : bool
 * @param lambda - double
 *
 * @return the cost after tuning. (If regularizatino is permitted,
 *   the cost would contain that. If the model is not usable, return -1)
 **/
template <class Derived>
Derived LinearRegression<Derived>::GradientDescentWithThreshhold(
    Eigen::Ref<
        Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        X,
    Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, 1>> y, double threshold,
    double learning_rate, bool check_learning_rate, double lambda) {
  if (!usable_) return -1;

  // parameter mapping
  Eigen::Map<Eigen::Matrix<Derived, Eigen::Dynamic, 1>, 0,
             Eigen::InnerStride<1>>
      param(parameters_.data() + 1, parameters_.rows() - 1);
  // augmented feature matrix X_plus
  Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      X_plus(y.rows(), X.cols() + 1);
  X_plus << Eigen::Matrix<Derived, Eigen::Dynamic, 1>::Ones(y.rows()), X;
  size_t batch_size = X.rows();

  // must check learning rate first
  learning_rate = CheckLearningRate(X, y, learning_rate, lambda);

  // variables to determine thresholds
  Derived cost_before_tuning =
      ComputeCostWithAugmentedFeature(X_plus, y, lambda);
  Derived cost_after_tuning = 0;

  // begin gradient descent
  while (abs(cost_before_tuning - cost_after_tuning) > threshold) {
    // record the cost before tuning
    cost_before_tuning = ComputeCostWithAugmentedFeature(X_plus, y, lambda);

    // each time train 100 times
    if (lambda == 0) {
      if (fit_intercept_) {
        for (size_t i = 0; i < 100; i++) {
          parameters_ = parameters_ -
                        learning_rate / batch_size *
                            (X_plus.transpose() * (X_plus * parameters_ - y));
        }
      } else {
        for (size_t i = 0; i < 100; i++) {
          param = param - learning_rate / batch_size *
                              (X.transpose() * (X_plus * parameters_ - y));
        }
      }
    } else if (fit_intercept_) {  // Regularize theta_0

      for (size_t i = 0; i < 100; i++) {
        parameters_ = parameters_ -
                      learning_rate / batch_size *
                          (X_plus.transpose() * (X_plus * parameters_ - y)) -
                      lambda * parameters_;
      }

    } else {  // Do not regularize theta_0

      for (size_t i = 0; i < 100; i++) {
        param = param -
                learning_rate / batch_size *
                    (X.transpose() * (X_plus * parameters_ - y)) -
                lambda * param;
      }
    }

    cost_after_tuning = ComputeCostWithAugmentedFeature(X_plus, y, lambda);
  }
  return cost_after_tuning;
}

/**
 * This function would compute cost with X_plus feature matrix
 *
 * @param X_plus - augmented feature matrix (one column inserted before the
 *  first column): Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic,
 *  Eigen::Dynamic, Eigen::RowMajor>>
 * @param y - label vector: Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic,
 *  1>>
 * @param lambda - regularization parameter: double
 *
 * @return cost after computing. If the model is not usable, return -1.
 **/
template <class Derived>
Derived LinearRegression<Derived>::ComputeCostWithAugmentedFeature(
    Eigen::Ref<
        Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        X_plus,
    Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, 1>> y, double lambda) {
  if (!usable_)  // return -1 if the model is not initiated
    return -1;

  size_t batch_size = X_plus.rows();
  Derived cost = 0.5 / batch_size * (X_plus * parameters_ - y).transpose() *
                 (X_plus * parameters_ - y);

  return cost;
}

}  // namespace linearregression
#endif /*linear-regression.hpp*/
