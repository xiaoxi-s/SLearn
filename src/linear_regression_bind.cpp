#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include "linear_regression.hpp"

namespace py = pybind11;

PYBIND11_MODULE(slearn, m) {
    py::class_<linearregression::LinearRegression<double>>(m, "LinearRegression")
        .def(py::init<>())
        .def(py::init<bool >())
        .def("print_parameters", & linearregression::LinearRegression<double>::PrintParameters)
        .def("gaussian_init", &linearregression::LinearRegression<double>::GaussianInit, py::arg("size"), py::arg("mu")=0, py::arg("sigma")=1)
        .def("constant_init", &linearregression::LinearRegression<double>::ConstantInit, py::arg("size"), py::arg("value"))
        .def("normal_equation", &linearregression::LinearRegression<double>::NormalEquation, py::arg("X"), py::arg("y"))
        .def("gradient_descent", &linearregression::LinearRegression<double>::GradientDescent, py::arg("X"), py::arg("y"), py::arg("iterations"), 
            py::arg("learning_rate"), py::arg("check_learning_rate")=false, py::arg("lambda")=0)
        .def("compute_cost", &linearregression::LinearRegression<double>::ComputeCost, py::arg("X"), py::arg("y"), py::arg("lambda")=0)
        .def("compute_cost_with_aug_feature", &linearregression::LinearRegression<double>::ComputeCostWithAugmentedFeature, py::arg("X_plus"), py::arg("y"), py::arg("lamda")=0);
}

