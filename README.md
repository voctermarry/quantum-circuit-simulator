## 用途

本项目是「量子线路仿真与验证平台」的代码仓库，用于逐步实现该方向的线路构建、状态仿真与结果验证能力。

当前处于基线状态：只有项目骨架，尚未实现任何业务算法。

## 环境与安装

- C++20 编译器（GCC 13 及以上）
- CMake 3.22 及以上

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## 测试

```bash
ctest --test-dir build --output-on-failure
```

基线只有骨架自检用例，后续新增用例同样通过 CTest 执行。

## 命令行入口

构建后提供 `quantum-circuit-simulator` 可执行文件：

```bash
./build/quantum-circuit-simulator version    # 打印版本号
./build/quantum-circuit-simulator --help     # 打印用法
```

## 现有公开接口

- 可执行程序 `quantum-circuit-simulator`，支持子命令 `version` 与 `help`
- C++ 静态库目标 `quantum_circuit_core`，公开头文件 `<quantum_circuit/version.hpp>`
- `quantum_circuit::version()` 返回当前版本号，`quantum_circuit::kVersion` 为同值常量

## 限制

- 除版本查询外没有其他功能。
- 输入输出格式、数据来源与算法均尚未定义。
