#include <eigen3/Eigen/Dense>
#include <fstream>
#include <string>
#include <vector>

/**
 * This class simply acts as a csv file reader
 **/
class CSVReader {
 public:
  CSVReader();

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  read_csv_as_matrix(std::string &file_name, const char delimeter = ',');

  std::vector<std::vector<std::string>> read_csv(std::string &file_name,
                                                 const char delimeter = ',');
};