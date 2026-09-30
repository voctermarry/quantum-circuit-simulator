#pragma once

#include <string_view>

namespace quantum_circuit {

/// 当前版本号，与 CMake 工程版本一致。
inline constexpr std::string_view kVersion = "0.1.0";

/// 返回当前版本号。
[[nodiscard]] std::string_view version() noexcept;

}  // namespace quantum_circuit
