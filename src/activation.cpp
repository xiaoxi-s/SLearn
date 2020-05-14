#include <cmath>
#include <eigen3/Eigen/Dense>

namespace activation {

/*
  The function takes a matrix as a parameter and return a matrix with the same
  dimension but the value are the output of sigmoid function taking the previous
  corresponding elements as input.

  @param Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic>
    & mat - whose elements would be sigmoided. It is a variable reference

  @returns the temp variable reference of the computed matrix
*/
template <class Derived>
Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
sigmoid(Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
            mat) {
  return mat.exp();
}

/*
  The function takes a number as a parameter and return a sigmoid(number)

  @param Derived & value - the value would be sigmoided. It is a variable
  reference
  @returns the sigmoided value of the input
*/
template <class Derived>
Derived& sigmoid(Derived& value) {
  return 1 / (exp(value) + 1);
}

template <class Derived>
Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& tanh(
    Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        mat) {
  return mat.tanh();
}
}  // namespace activation