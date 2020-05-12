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
Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic>& sigmoid(
    Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic>& mat) {
  Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic> temp(mat.rows(),
                                                              mat.cols());
  size_t sz = mat.rows() * mat.cols();  // number of data pieces
  for (size_t i = 0; i < sz; i++) {
    temp(i) = 1 / (exp(mat(i)) + 1);
  }
  return temp;
}

/*
  The function takes a number as a parameter and return a sigmoid(number)

  @param Derived & value - the value would be sigmoided. It is a variable reference
  @returns the sigmoided value of the input
*/
template <class Derived>
Derived& sigmoid(Derived & value) {
  return 1/(exp(value) + 1);
}

}  // namespace activation