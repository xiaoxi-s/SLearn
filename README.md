# SLearn


## Prerequisites


- Environment: Ubuntu >= 19.0.4
- Python >= 3.6
- GCC >= 4.8
- Libraries: `pybind11` for Python and `eigen3` for C++.

For how to install `pybind11`, see this [link](https://zoomadmin.com/HowToInstall/UbuntuPackage/python-pybind11)

For how to install `eigen3`, see this [link](http://eigen.tuxfamily.org/index.php?title=Main_Page)

# Install

This project aims at implementing a basic machine learning library with C++ and provides python interface. 

With pybind11 and eigen3 installed for C++, under the directory of Slearn, only through the command 

```
python3 setup.py install 
```

, slearn library could be used by python.
