#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

template <typename T>
void read_from_csv(std::string path, std::vector<T>& data) {
  std::ifstream file(path);
  // vector<F> data;
  // std::cout << path << std::endl;
  // std::cout << data.size() << std::endl;

  int index = 0;
  std::string line;
  while (getline(file, line)) {
    std::stringstream ss(line);
    std::string cell;
    while (getline(ss, cell, ',')) {
      // data.push_back(stod(cell));
      data[index++] = stod(cell);
      // cout << index << ".";
    }
  }
}

template <typename T>
void save_to_csv(std::string path, std::vector<T>& data) {
  int index = 0;
  // std::cout << "save_to_csv:" <<  path << std::endl;
  std::ofstream file(path);
  for (const auto& value : data) {
    file << (float)value << "\n";
    index++;
  }
  file.close();
  // std::cout << "nums:" << index << std::endl;
}
