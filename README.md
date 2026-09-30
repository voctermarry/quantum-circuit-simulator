## 用途

本项目是「量子线路仿真与验证平台」的代码仓库，用于逐步实现该方向的线路构建、状态仿真与结果验证能力。

当前支持 OpenQASM 2.0 的一个子集：解析线路、以状态向量执行量子门，并按给定种子进行测量采样。

## 环境与安装

- Python 3.11 及以上

```bash
python -m pip install -e .
```

## 测试

```bash
python -m pytest
```

## 命令行入口

安装后提供 `quantum-circuit-simulator` 命令：

```bash
quantum-circuit-simulator version                    # 打印版本号
quantum-circuit-simulator --help                     # 打印用法
quantum-circuit-simulator simulate SOURCE [--shots N] [--seed S]
```

### simulate 子命令

- `SOURCE` 为 UTF-8 编码的 OpenQASM 文件路径；为 `-` 时读取标准输入。
- `--shots` 默认 1024，须为正整数；`--seed` 默认 0，接受有符号整数。
- 成功时 stdout 输出一行 JSON，字段顺序固定：
  `schema_version`、`shots`、`seed`、`num_qubits`、`num_clbits`、`counts`；
  `counts` 按键字典序输出，计数总和等于 shots。同一输入/shots/seed 的重复运行字节一致。

示例：

```bash
quantum-circuit-simulator simulate circuit.qasm --shots 1024 --seed 0
# {"schema_version": 1, "shots": 1024, "seed": 0, "num_qubits": 2, "num_clbits": 2, "counts": {"00": 512, "11": 512}}
```

### 支持的 OpenQASM 2.0 子集

- `OPENQASM 2.0;` 版本声明与 `include "qelib1.inc";`
- 单个 `qreg` 与单个 `creg`（大小须为正整数，寄存器名不可重复）
- `x`、`h`、`cx` 门与 `measure q[i] -> c[j];`，仅接受常量下标单个位
- `//` 行注释；语句按源码顺序生效，测量只能位于全部量子门之后
- 每个量子位/经典位最多参与一次测量，未写入的经典位结果为 0
- counts 的键按经典寄存器最高下标到最低下标组成固定宽度二进制串

### 错误处理

| 情况 | 退出码 | stderr（单行 JSON） |
| --- | --- | --- |
| 文件不存在/不可读/非合法 UTF-8 | 1 | `{"error": "io_error"}` |
| 词法或语法错误 | 2 | `{"error": "parse_error", "line": l, "column": c}` |
| 重复声明、越界、名称未声明、重复测量、测量后门等 | 2 | `{"error": "validation_error", "line": l, "column": c}` |
| `--shots`/`--seed` 参数非法 | 2 | argparse 用法错误（不读取输入） |

失败时 stdout 为空，不会输出部分统计。

## 现有公开接口

- 命令行程序 `quantum-circuit-simulator`
- Python 包 `quantum_circuit`，其 `__version__` 为当前版本号
