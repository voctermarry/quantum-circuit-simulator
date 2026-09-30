#include <cstdlib>
#include <iostream>
#include <string_view>

#include "quantum_circuit/version.hpp"

namespace {

int failures = 0;

void expect(bool ok, std::string_view what) {
  if (!ok) {
    ++failures;
    std::cerr << "断言失败: " << what << '\n';
  }
}

}  // namespace

int main() {
  expect(!quantum_circuit::version().empty(), "version() 非空");
  expect(quantum_circuit::version() == quantum_circuit::kVersion, "version() 与 kVersion 一致");
  if (failures != 0) {
    std::cerr << failures << " 项断言失败\n";
    return 1;
  }
  std::cout << "全部断言通过\n";
  return 0;
}
