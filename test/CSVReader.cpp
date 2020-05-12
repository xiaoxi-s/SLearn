#include "CSVReader.hpp"

CSVReader::CSVReader() {}

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
CSVReader::read_csv_as_matrix(std::string file_name, const char delimeter) {
  std::vector<std::vector<std::string>> lines_of_string =
      read_csv(file_name, delimeter);
  // record the row number and the column number.
  size_t row_number = lines_of_string.size();
  size_t col_number = 0;
  if (row_number != 0)
    col_number = lines_of_string.front().size();

  // the matrix to be returned.
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      mat_returned = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>(row_number, col_number);

  // convert each entry into double and then store it into the matrix
  for (int i = row_number - 1; i >= 0; i--) {
    std::vector<std::string> one_line = lines_of_string.back();
    lines_of_string.pop_back();
    // each column in one row
    for (int j = col_number - 1; j >= 0; j--) {
      *(mat_returned.data() + i * row_number + j) =
          atof(one_line.back().c_str());
      one_line.pop_back();
    }
  }

  return mat_returned;
}

std::vector<std::vector<std::string>> CSVReader::read_csv(
    std::string file_name, const char delimeter) {
  std::ifstream fin("linear-regression-test-label.txt");  // open file

  // Store lines temporarily
  std::vector<std::vector<std::string>> matrix_in_lines =
      std::vector<std::vector<std::string>>();
  // store one line in string
  std::string temp_string;
  // store one line temporarily
  std::vector<std::string> words_in_line;

  if (fin.is_open()) {
    while (std::getline(fin, temp_string)) {
      // split one line of string
      std::stringstream string_builder(temp_string);
      // One line of values
      words_in_line = std::vector<std::string>();
      // one value
      std::string entry;
      while (std::getline(string_builder, entry, delimeter)) {
        // convert the strings in one line into double and store those in one
        // line
        words_in_line.push_back(entry);
      }
      matrix_in_lines.push_back(words_in_line);
      words_in_line = std::vector<std::string>();
    } 
    fin.close();
  }
  return matrix_in_lines;
}