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
      const Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &X,
      const Eigen::Matrix<Derived, Eigen::Dynamic, 1> &Y, double, double);

  Derived GradientDescentWithThreshhold(
      const Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic> &X,
      const Eigen::Matrix<Derived, Eigen::Dynamic, 1> &Y, double threshold,
      double learning_rate, double lambda);

 public:
  Eigen::Matrix<Derived, Eigen::Dynamic, 1> parameters_;
  bool fit_intercept_;  // Whether to fit the intercept (bias unit)

  LinearRegression();
  LinearRegression(bool fit_intercept_);
  void GaussianInit(const size_t, const Derived mu = 0,
                    const Derived sigma = 1);
  void ConstantInit(const size_t, const Derived);

  Derived GradientDescent(
      const Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &,
      const Eigen::Matrix<Derived, Eigen::Dynamic, 1> &, size_t, double,
      bool check_learning_rate = false, double lambda = 0);

  Derived NormalEquation(
      Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>>
          X,
      Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, 1>> Y);

  Derived ComputeCost(
      const Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &X,
      const Eigen::Matrix<Derived, Eigen::Dynamic, 1> &Y, double lambda = 0);

  Derived ComputeCostWithAugmentedFeature(
      const Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic> &X_plus,
      const Eigen::Matrix<Derived, Eigen::Dynamic, 1> &Y, double lambda = 0);

  Derived Fit(Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>>
                  X,
              Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, 1>> Y,
              double threshold = 0.1, bool fit_intercept = true,
              double lambda = 0);

  Eigen::Matrix<Derived, Eigen::Dynamic, 1> Predict(
      const Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic> &);

  void PrintParameters();
};

// Eigen::Matrix<Derived>
/*
  Constructor of the LinearRegression class: set fit_intercept_ to false.
  Meaning: Do not fit theta_0
*/
template <class Derived>
LinearRegression<Derived>::LinearRegression() {
  this->usable_ = false;
  this->fit_intercept_ = false;
}

/*
  Constructor of the LinearRegression class: set fit_intercept_ as the
  parameter

  @param bool fit_intercept_ - to be set
*/
template <class Derived>
LinearRegression<Derived>::LinearRegression(bool fit_intercept_) {
  this->fit_intercept_ = fit_intercept_;
  this->usable_ = false;
}

/*
  Initialize the values of the parameters by Gaussian distribution (mu = 0,
  sigmoid = 1)

  @param const size_t sz - the number of the parameters: contain the bias
    parameter
  @param const Derived mu - the mean of Gaussian distribution
  @param const Derived sigma - the standard deviation of Gaussian distribution
*/
template <class Derived>
void LinearRegression<Derived>::GaussianInit(const size_t size,
                                             const Derived mu,
                                             const Derived sigma) {
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

/*
  Initialize the values of the parameters with a constant value

  @param const size_t sz - the number of the parameters: contain the bias
    parameter
  @param const Derived value - the constant value for initialization
*/
template <class Derived>
void LinearRegression<Derived>::ConstantInit(const size_t size,
                                             const Derived value) {
  size_t size_of_parameters = size + 1;
  const Eigen::Index idx = size_of_parameters;

  parameters_ = Eigen::Matrix<Derived, Eigen::Dynamic, 1>(idx);

  for (size_t i = 0; i < size_of_parameters; i++) {
    parameters_(i) = value;
  }

  if (!fit_intercept_) parameters_(0) = 0;

  this->usable_ = true;
}

/*
  Use gradient descent to tune the parameter

  @param const Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic>
    & X - Training set matrix of m by n
  @param const Eigen::Matrix<Derived, Eigen::Dynamic, 1>
    &Y - Training set lable of m by 1
  @param size_t iterations    - times for iterations of gradient descent
  @param double learning_rate - learning_rate of this model
  @param double lambda        - regularization term; 0 means no regularization

  @return the cost after tuning. (If regularizatino is permitted,
    the cost would contain that. If the model is not usable, return -1)
*/
template <class Derived>
Derived LinearRegression<Derived>::GradientDescent(
    const Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &X,
    const Eigen::Matrix<Derived, Eigen::Dynamic, 1> &Y, size_t iterations,
    double learning_rate, bool check_learning_rate, double lambda) {
  if (!usable_) return -1;

  Eigen::Map<Eigen::Matrix<Derived, Eigen::Dynamic, 1>, 0,
             Eigen::InnerStride<1>>
      param(parameters_.data() + 1, parameters_.rows() - 1);

  Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X_plus(Y.rows(),
                                                                X.cols() + 1);
  X_plus << Eigen::Matrix<Derived, Eigen::Dynamic, 1, Eigen::RowMajor>::Ones(Y.rows()), X;
  size_t batch_size = X.rows();

  if (check_learning_rate)
    // check learning rate first
    learning_rate = CheckLearningRate(X, Y, learning_rate, lambda);

  // begin gradient descent
  if (lambda == 0) {
    if (fit_intercept_) {
      for (size_t i = 0; i < iterations; i++) {
        parameters_ =
            parameters_ - learning_rate / batch_size *
                              (X_plus.transpose() * (X_plus * parameters_ - Y));
      }

    } else {
      for (size_t i = 0; i < iterations; i++) {
        param = param - learning_rate / batch_size *
                            (X.transpose() * (X_plus * parameters_ - Y));
      }
    }
  } else if (fit_intercept_) {  // Regularize theta_0

    for (size_t i = 0; i < iterations; i++) {
      parameters_ = parameters_ -
                    learning_rate / batch_size *
                        (X_plus.transpose() * (X_plus * parameters_ - Y)) -
                    lambda * parameters_;
    }

  } else {  // Do not regularize theta_0

    for (size_t i = 0; i < iterations; i++) {
      param = param -
              learning_rate / batch_size *
                  (X.transpose() * (X_plus * parameters_ - Y)) -
              lambda * param;
    }
  }
  return ComputeCost(X, Y, lambda);
}

/*
  Use param = (X^T * X)^(-1) * X^T * Y to initialize the parameter. It is
  more efficent if the number of instances is small. The parameter size
  would always be number of features + 1 (a bias unit added).

  @param Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic,
  Eigen::RowMajor>> & X - the instances' features
  @param Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, 1>>
    & Y - The expected label
  @return the cost of using such parameter to predict training instances
*/
template <class Derived>
Derived LinearRegression<Derived>::NormalEquation(
    Eigen::Ref<
        Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        X,
    Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, 1>> Y) {
  // Initialize parameters
  parameters_ = Eigen::Matrix<Derived, Eigen::Dynamic, 1>(X.cols() + 1, 1);
  if (fit_intercept_) {  // fit intercept option is on
    Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        X_plus(Y.rows(), X.cols() + 1);
    X_plus << Eigen::Matrix<Derived, Eigen::Dynamic, 1>::Ones(Y.rows()), X;
    parameters_ << (X_plus.transpose() * X_plus).inverse() *
                       X_plus.transpose() * Y;

    this->usable_ = true;
    return ComputeCost(X, Y);
  } else {
    parameters_ << 0, (X.transpose() * X).inverse() * X.transpose() * Y;

    this->usable_ = true;
    return ComputeCost(X, Y);
  }
}

/*
  Compute cost of current parameter and given data set.

  @param const Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic>
    & X - Training set matrix of m by n
  @param const Eigen::Matrix<Derived, Eigen::Dynamic, 1>
    &Y - Training set lable of m by 1
  @param double lambda - regularization term; 0 means no regularization

  @return the cost after tuning. (If regularizatino is permitted,
    the cost would contain that. If the model is not usable, return -1)
*/
template <class Derived>
Derived LinearRegression<Derived>::ComputeCost(
    const Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &X,
    const Eigen::Matrix<Derived, Eigen::Dynamic, 1> &Y, double lambda) {
  if (!usable_)  // return -1 if the model is not initiated
    return -1;

  Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X_plus(Y.rows(),
                                                                X.cols() + 1);
  X_plus << Eigen::Matrix<Derived, Eigen::Dynamic, 1>::Ones(Y.rows()), X;

  Eigen::Map<Eigen::Matrix<Derived, Eigen::Dynamic, 1>, 1,
             Eigen::InnerStride<1>>
      param(parameters_.data() + 1, parameters_.rows() - 1,
            Eigen::InnerStride<1>());

  size_t batch_size = X.rows();

  Derived c = 0.5 / batch_size * (X_plus * parameters_ - Y).transpose() *
              (X_plus * parameters_ - Y);

  return c;
}

/*
  Predict the output given a set of instances
  @param const Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic>
    & X_test - set of instances

  @return Eigen::Matrix<Derived, Eigen::Dynamic, 1> - the predicted value
    vector
*/
template <class Derived>
Eigen::Matrix<Derived, Eigen::Dynamic, 1> LinearRegression<Derived>::Predict(
    const Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic> &X_test) {
  // If there is no bias unit
  size_t size = X_test.rows();
  Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic> X_plus(
      X_test.rows(), X_test.cols() + 1);
  X_plus << Eigen::Matrix<Derived, Eigen::Dynamic, 1>::Ones(X_test.rows()),
      X_test;

  Eigen::Matrix<Derived, Eigen::Dynamic, 1> Y_hat = X_plus * parameters_;

  return Y_hat;
}

/*
  This method would train the model with gradient descent or normal
  equation based on input size.
*/
template <class Derived>
Derived LinearRegression<Derived>::Fit(
    Eigen::Ref<
        Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        X,
    Eigen::Ref<Eigen::Matrix<Derived, Eigen::Dynamic, 1>> Y, double threshould,
    bool fit_intercept, double lambda) {
  // reset the parameters in the model
  fit_intercept_ = fit_intercept;

  // test for valid input
  if (X.rows() != Y.rows()) {
    return -1;
  }

  // supervised learning. ONLY one label per instance
  if (Y.cols() != 1) {
    return -1;
  }

  // choose between normal equation and gradient descent
  if (X.rows() > X.cols() && X.rows() <= 100000) {
    return NormalEquation(X, Y);
  } else {
    return GradientDescentWithThreshhold(X, Y, 100, 0.1, lambda);
  }
}

/*
  Print parameters as a column vector
*/
template <class Derived>
void LinearRegression<Derived>::PrintParameters() {
  if (usable_) std::cout << parameters_ << std::endl;
}

/*
  This function check the alpha rate to avoid overshooting.

  @param double learning_rate - learning_rate of this model
  @return the initial learning rate or a smaller learning rate if the inital
    learning rate causes overshooting
*/
template <class Derived>
double LinearRegression<Derived>::CheckLearningRate(
    const Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &X,
    const Eigen::Matrix<Derived, Eigen::Dynamic, 1> &Y, double learning_rate,
    double lambda) {
  // Initialize a new parameter vector as temporary variable
  Derived cost_before_tuning = 0;
  Derived cost_after_tuning = 0;

  Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> X_plus(Y.rows(),
                                                                X.cols() + 1);
  X_plus << Eigen::Matrix<Derived, Eigen::Dynamic, 1>::Ones(Y.rows()), X;
  size_t batch_size = X.rows();

  // create a temp parameter varible to test
  Eigen::Matrix<Derived, Eigen::Dynamic, 1> initial_param(parameters_);
  Eigen::Map<Eigen::Matrix<Derived, Eigen::Dynamic, 1>, 1,
             Eigen::InnerStride<1>>
      param(parameters_.data() + 1, initial_param.rows() - 1,
            Eigen::InnerStride<1>());

  // record the cost before tuning
  cost_before_tuning = ComputeCost(X, Y, lambda);

  do {
    if (lambda == 0) {
      if (fit_intercept_) {
        parameters_ =
            parameters_ - learning_rate / batch_size *
                              (X_plus.transpose() * (X_plus * parameters_ - Y));

      } else {
        param = param - learning_rate / batch_size *
                            (X.transpose() * (X_plus * parameters_ - Y));
      }

    } else if (fit_intercept_) {  // Regularize theta_0
      parameters_ = parameters_ -
                    learning_rate / batch_size *
                        (X_plus.transpose() * (X_plus * parameters_ - Y)) -
                    lambda * parameters_;

    } else {  // Do not regularize theta_0
      param = param -
              learning_rate / batch_size *
                  (X.transpose() * (X_plus * parameters_ - Y)) -
              lambda * param;
    }
    cost_after_tuning = ComputeCost(X, Y, lambda);
    if (cost_before_tuning > cost_after_tuning) {
      parameters_ = initial_param;
      return learning_rate;
    } else {
      learning_rate = learning_rate / 10;
      cost_before_tuning = cost_after_tuning;
    }
  } while (true);

  return learning_rate;
}

/**
 *
 *
 * @param const Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic>
 *   & X - Training set matrix of m by n
 * @param const Eigen::Matrix<Derived, Eigen::Dynamic, 1>
 *   &Y - Training set lable of m by 1
 * @param Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic>
 *   & new_param - new parameter used temporarily in the linear regression model
 * @param double learning_rate - learning_rate of this model
 * @param double lambda        - regularization term; 0 means no regularization
 *
 * @return the cost after tuning. (If regularizatino is permitted,
 *   the cost would contain that. If the model is not usable, return -1)
 **/
template <class Derived>
Derived LinearRegression<Derived>::GradientDescentWithThreshhold(
    const Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic> &X,
    const Eigen::Matrix<Derived, Eigen::Dynamic, 1> &Y, double threshold,
    double learning_rate, double lambda) {
  if (!usable_) return -1;

  Eigen::Map<Eigen::Matrix<Derived, Eigen::Dynamic, 1>, 0,
             Eigen::InnerStride<1>>
      param(parameters_.data() + 1, parameters_.rows() - 1);
  Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic> X_plus(Y.rows(),
                                                                X.cols() + 1);
  X_plus << Eigen::Matrix<Derived, Eigen::Dynamic, 1>::Ones(Y.rows()), X;
  size_t batch_size = X.rows();

  // must check learning rate first
  learning_rate = CheckLearningRate(X, Y, learning_rate, lambda);

  // variables to determine thresholds
  Derived cost_before_tuning =
      ComputeCostWithAugmentedFeature(X_plus, Y, lambda);
  Derived cost_after_tuning = 0;

  // begin gradient descent
  while (abs(cost_before_tuning - cost_after_tuning) > threshold) {
    // record the cost before tuning
    cost_before_tuning = ComputeCostWithAugmentedFeature(X_plus, Y, lambda);

    // each time train 100 times
    if (lambda == 0) {
      if (fit_intercept_) {
        for (size_t i = 0; i < 100; i++) {
          parameters_ = parameters_ -
                        learning_rate / batch_size *
                            (X_plus.transpose() * (X_plus * parameters_ - Y));
        }
      } else {
        for (size_t i = 0; i < 100; i++) {
          param = param - learning_rate / batch_size *
                              (X.transpose() * (X_plus * parameters_ - Y));
        }
      }
    } else if (fit_intercept_) {  // Regularize theta_0

      for (size_t i = 0; i < 100; i++) {
        parameters_ = parameters_ -
                      learning_rate / batch_size *
                          (X_plus.transpose() * (X_plus * parameters_ - Y)) -
                      lambda * parameters_;
      }

    } else {  // Do not regularize theta_0

      for (size_t i = 0; i < 100; i++) {
        param = param -
                learning_rate / batch_size *
                    (X.transpose() * (X_plus * parameters_ - Y)) -
                lambda * param;
      }
    }

    cost_after_tuning = ComputeCostWithAugmentedFeature(X_plus, Y, lambda);
  }
  return cost_after_tuning;
}

/**
 * This function would compute cost with X_plus feature matrix
 *
 * @param
 * @param
 * @param
 *
 * @return
 **/
template <class Derived>
Derived LinearRegression<Derived>::ComputeCostWithAugmentedFeature(
    const Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic> &X_plus,
    const Eigen::Matrix<Derived, Eigen::Dynamic, 1> &Y, double lambda) {
  if (!usable_)  // return -1 if the model is not initiated
    return -1;

  size_t batch_size = X_plus.rows();
  Derived cost = 0.5 / batch_size * (X_plus * parameters_ - Y).transpose() *
                 (X_plus * parameters_ - Y);

  return cost;
}

}  // namespace linearregression
#endif /*linear-regression.hpp*/
