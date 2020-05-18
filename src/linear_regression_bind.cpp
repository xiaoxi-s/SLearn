#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include "linear_regression.hpp"

namespace py = pybind11;

// Dynamic attribute is not allowed
// not sure parameters would be returned as a reference or as a copy
PYBIND11_MODULE(slearn, m) {
  m.doc() = R"pbdoc(
      .. currentmodule:: slearn
      .. autosummary::
        :toctree: _generate

        slearn

  )pbdoc";
  py::module linear_model = m.def_submodule("linear_model", R"pbdoc(
      .. currentmodule:: linear_model
      .. autosummary::
        :toctree: _generate

  )pbdoc");
  py::class_<linearregression::LinearRegression<double>>(
      linear_model, "LinearRegression", R"pbdoc(
      .. currentmodule:: LinearRegression
      .. autosummary::
        :toctree: _generate
  )pbdoc")
      .def(py::init<>(), R"pbdoc(
        Initialize a LinearRegression model with fit_intercept set to False
        
      )pbdoc")                // default constructor
      .def(py::init<bool>())  // constructor initalizing fit_intercept_ variable
      .def_readwrite(
          "param", &linearregression::LinearRegression<double>::parameters_,
          py::return_value_policy::reference_internal)  // return parameters in
                                                        // this model
      .def_readwrite("fit_intercept",
                     &linearregression::LinearRegression<
                         double>::fit_intercept_)  // fit_intercept_ attribute
      .def("print_parameters",
           &linearregression::LinearRegression<double>::PrintParameters,
           R"pbdoc(
        Print the parameters in the linear regression model

        )pbdoc")  // print parameters
      .def("gaussian_init",
           &linearregression::LinearRegression<double>::GaussianInit,
           py::arg("size"), py::arg("mu") = 0, py::arg("sigma") = 1, R"pbdoc(
        Init parameters with size, and gaussian distribution
        )pbdoc")  // init params
      .def("constant_init",
           &linearregression::LinearRegression<double>::ConstantInit,
           py::arg("size"), py::arg("value"), R"pbdoc(
        Init parameters with size, and the same value
        )pbdoc")  // init params
      .def("normal_equation",
           &linearregression::LinearRegression<double>::NormalEquation,
           py::arg("X"), py::arg("y"), R"pbdoc(
        Use normal equation to tune the parameters given X and y.
        )pbdoc")  // normal equation
      .def("gradient_descent",
           &linearregression::LinearRegression<double>::GradientDescent,
           py::arg("X"), py::arg("y"), py::arg("iterations"),
           py::arg("learning_rate"), py::arg("check_learning_rate") = false,
           py::arg("lambda") = 0, R"pbdoc(
        Use gradient descent to tune the parameters given X, y and learning_rate.
        )pbdoc")  // gradient descent
      .def("compute_cost",
           &linearregression::LinearRegression<double>::ComputeCost,
           py::arg("X"), py::arg("y"), py::arg("lambda") = 0, R"pbdoc(
        Compute cost given X, y and lambda (regularization parameter).
        )pbdoc")  // compute cost
      .def("compute_cost_with_aug_feature",
           &linearregression::LinearRegression<
               double>::ComputeCostWithAugmentedFeature,
           py::arg("X_plus"), py::arg("y"), py::arg("lamda") = 0, R"pbdoc(
        Compute cost given X_plus, y and lambda (regularization parameter).
        )pbdoc")  // compute cost with augmented feature
      .def("fit", &linearregression::LinearRegression<double>::Fit,
           py::arg("X"), py::arg("y"), py::arg("threshold") = 0.1,
           py::arg("fit_intercept") = true, py::arg("lambda") = 0,
           py::return_value_policy::reference_internal, R"pbdoc(
        Fit the linear model based on X, y, cost difference threshold and lambda (regularization parameter).
        )pbdoc");  // fit method
}
