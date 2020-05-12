#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "linear_regression.hpp"

namespace py = pybind11;

PYBIND11_MODULE(slearn, m) {
    py::class_<linearregression::LinearRegression<double>>(m, "LinearRegression")
        .def(py::init<>())
        .def(py::init<bool >())
        .def("PrintParameters", & linearregression::LinearRegression<double>::PrintParameters)
        .def("GaussianInit", &linearregression::LinearRegression<double>::GaussianInit, py::arg("size"), py::arg("mu"), py::arg("sigma"))
        .def("ConstantInit", &linearregression::LinearRegression<double>::ConstantInit, py::arg("size"), py::arg("value"));
}

