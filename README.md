## 用途

本项目是「量子线路仿真与验证平台」的代码仓库，用于逐步实现该方向的线路构建、状态仿真与结果验证能力。

当前支持一个 OpenQASM 2.0 子集的状态向量仿真入口 `simulate`，比较两份线路是否等价的 `equivalent` 入口，以及把线路化简为确定规范形式的 `optimize` 入口。

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
quantum-circuit-simulator version                # 打印版本号
quantum-circuit-simulator --help                 # 打印用法
quantum-circuit-simulator simulate circuit.qasm  # 仿真并采样
quantum-circuit-simulator equivalent a.qasm b.qasm  # 判断两份线路是否同一变换
quantum-circuit-simulator optimize circuit.qasm  # 化简为规范形式
```

### simulate 子命令

```bash
quantum-circuit-simulator simulate SOURCE [--shots N] [--seed S] [--noise-model PATH]
```

- `SOURCE`：UTF-8 编码的 OpenQASM 源文件路径；`-` 表示从标准输入读取。
- `--shots`：采样次数，默认 `1024`，必须为正整数。
- `--seed`：采样随机数种子，默认 `0`，接受有符号整数。相同源文件、shots、seed 的重复运行产生字节一致的输出。
- `--noise-model`：可选的 UTF-8 JSON 噪声模型文件路径；`-` 表示从标准输入读取（线路与模型不能同时取 `-`）。提供后改用密度矩阵仿真。

成功时 stdout 输出单行 JSON，字段顺序固定：

```json
{"schema_version": 1, "shots": 1024, "seed": 0, "num_qubits": 2, "num_clbits": 2, "counts": {"00": 512, "11": 512}}
```

`counts` 的键是按经典寄存器最高下标到最低下标排列的定宽二进制串，仅包含出现过的结果，按键的字典序输出；计数总和等于 shots，未写入的经典位保持 0。

### 噪声模型（密度矩阵仿真）

提供 `--noise-model` 后按密度矩阵演化，模型为 JSON 对象，只允许以下四个可选键（至少一个），值为 0 到 1 的有限 JSON 数字：

- `amplitude_damping`：|1〉以概率 γ 衰减到 |0〉。
- `phase_damping`：布居不变，非对角元乘以 1-γ。
- `bit_flip`：ρ → (1-p)ρ + pXρX。
- `depolarizing`：ρ → (1-p)ρ + pI/2。

每个量子门完成后，对该门涉及的量子位施加配置的通道；`cx` 的两位按下标升序处理，同一位固定按 `amplitude_damping`、`phase_damping`、`bit_flip`、`depolarizing` 的顺序处理。采样取最终密度矩阵的对角概率，噪声演化不消耗 seed。

成功输出字段顺序为 `schema_version`、`shots`、`seed`、`num_qubits`、`num_clbits`、`noise_model`、`counts`，其中 `schema_version` 为 2；`noise_model` 按固定通道顺序仅回显已给概率，模型的空白与键顺序不影响输出。

### equivalent 子命令

```bash
quantum-circuit-simulator equivalent LEFT RIGHT
```

- `LEFT`、`RIGHT`：UTF-8 编码的 OpenQASM 源文件路径；`-` 表示从标准输入读取，但两侧不能同时为 `-`（此时不读取标准输入）。
- 判断两份无噪声线路是否表示同一量子变换，覆盖所有输入态（比较完整酉矩阵，而非全零态采样）。比较不受空白、注释或寄存器声明名称影响。
- 量子位数相同时，取测量之前的门构成的酉矩阵，允许全局相位，距离定义为

  ```
  d = min_{|λ|=1} ‖U − λV‖_F / sqrt(2 × 2^n) = sqrt(1 − |tr(U†V)| / 2^n)
  ```

  仅当 `d ≤ 1e-10` 时变换相同。
- 量子位到经典位的测量映射（含经典寄存器宽度）也必须一致才判定为完全等价；寄存器名称不参与比较。
- 最多比较 8 个量子位。

成功时 stdout 输出单行 JSON，字段顺序固定为 `schema_version`、`equivalent`、`reason`、`left_num_qubits`、`right_num_qubits`、`distance`、`tolerance`，其中 `schema_version` 为 1：

```json
{"schema_version": 1, "equivalent": true, "reason": "equivalent", "left_num_qubits": 2, "right_num_qubits": 2, "distance": 0.0, "tolerance": 1e-10}
```

`reason` 的取值：

- `equivalent`：变换相同（允许全局相位）且测量布局一致，`equivalent` 为 `true`。
- `measurement_layout_mismatch`：变换相同，但经典寄存器宽度或量子位→经典位映射不同；保留 `distance`。
- `unitary_distance`：`distance` 超过容差（布局差异此时不再单独报告）。
- `qubit_count_mismatch`：两侧量子位数不同，此时 `distance` 为 `null` 且不构造酉矩阵。

失败时（退出码非 0）stdout 为空，stderr 输出单行 JSON，错误对象带 `input` 字段标明出错的一侧（`left` 或 `right`）；两侧都有问题时按 LEFT、RIGHT 顺序报告首个错误。

### optimize 子命令

```bash
quantum-circuit-simulator optimize SOURCE
```

- `SOURCE`：UTF-8 编码的 OpenQASM 源文件路径；`-` 表示从标准输入读取。
- 把线路化简为确定的规范形式：同一依赖链上成对的 `x`、`h` 以及控制位和目标位相同的 `cx` 消去（两门之间仅有与所用量子位不相交的门时视为相邻）；同一量子位同轴的 `rx`、`ry`、`rz` 合并，角度规范到 `(-pi, pi]`，绝对值不超过 `1e-12` 的旋转删除，接近 `pi` 的 `rx`、`ry` 不替换为 `x`；互不共享量子位的门按最小量子位、最大量子位、门名和参数文本稳定重排，其余门保留依赖顺序。化简与重排持续至稳定，仅独立门书写顺序不同的输入产生字节一致的输出。
- 规范线路统一使用 `q`、`c` 作为寄存器名，保留尺寸与量子位到经典位的测量映射，测量按经典位、量子位下标升序输出；结果在 `equivalent` 口径下与原线路等价（含全局相位容忍与测量布局）。

成功时 stdout 输出单行 JSON，字段顺序固定为 `schema_version`、`num_qubits`、`num_clbits`、`original_gate_count`、`optimized_gate_count`、`changed`、`qasm`，其中 `schema_version` 为 1：

```json
{"schema_version": 1, "num_qubits": 1, "num_clbits": 1, "original_gate_count": 2, "optimized_gate_count": 0, "changed": true, "qasm": "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n"}
```

门数只统计测量前的量子门；`changed` 表示门序列是否变化；`qasm` 是完整线路文本，按版本声明、include、寄存器、门、测量逐项换行并以换行结尾，旋转角使用最多 17 位有效数字的十进制或小写科学计数法（负零写成 `0`），可被现有解析器重读。相同输入重复调用产生字节一致的输出；没有量子门的线路也正常返回。

错误处理与 `simulate` 相同：源文件不存在、不可读或不是合法 UTF-8 时退出码为 1，stderr 单行 JSON 的 `error` 为 `io_error`；词法、语法和语义错误为 `parse_error` 或 `validation_error`（带源码位置），退出码为 2；命令行参数错误以退出码 2 结束且不读取输入。所有失败的 stdout 为空。

### 支持的 OpenQASM 2.0 子集

```
OPENQASM 2.0;
include "qelib1.inc";
qreg name[size];
creg name[size];
x q[i];
h q[i];
cx q[i], q[j];
rx(angle) q[i];
ry(angle) q[i];
rz(angle) q[i];
measure q[i] -> c[k];
```

- 必须以 `OPENQASM 2.0;` 和 `include "qelib1.inc";` 开头。
- 恰好声明一个非空 `qreg` 和一个非空 `creg`；支持 `//` 行注释。
- 门与测量只接受带常量下标的单个位；语句按源码顺序生效。
- `rx`、`ry`、`rz` 恰好接收一个以弧度表示的角参数；角表达式支持十进制整数、实数（含科学计数法）、常量 `pi`、圆括号、一元 `+`/`-` 以及二元 `+`、`-`、`*`、`/`（乘除优先于加减，同级从左到右，括号优先）。
- 测量只能出现在所有量子门之后，每个量子位、经典位最多参与一次测量。

### 退出码与错误输出

失败时 stdout 为空，stderr 输出单行 JSON：

| 情况 | 退出码 | error |
| --- | --- | --- |
| 文件不存在、不可读或不是合法 UTF-8 | 1 | `io_error` |
| 噪声模型路径不可读 | 1 | `io_error` |
| 词法/语法错误（带从 1 开始的 line、column） | 2 | `parse_error` |
| 重复声明、寄存器越界或大小非法、名称未声明、重复测量、测量后仍有量子门、角表达式中出现 `pi` 以外的名称、除以零、数值字面量溢出或结果非有限数等（带源码位置） | 2 | `validation_error` |
| 噪声模型内容不合规（非法 UTF-8/JSON、非对象、空对象、未知键、重复键、概率非 0 到 1 的有限数字）或线路与模型同时取 `-`（此时不读取标准输入） | 2 | `noise_model_error` |
| 带噪声线路超过 10 个量子位 | 3 | `simulation_error` |
| `equivalent` 任一侧文件不存在、不可读或不是合法 UTF-8 | 1 | `io_error`（带 `input`：`left`/`right`） |
| `equivalent` 两侧同时取 `-`（此时不读取标准输入） | 2 | `comparison_error` |
| `equivalent` 任一侧词法/语法或语义错误 | 2 | `parse_error` / `validation_error`（带 `input`：`left`/`right`，按 LEFT、RIGHT 顺序报告首个错误） |
| `equivalent` 任一侧线路超过 8 个量子位 | 3 | `simulation_error`（带 `input`：`left`/`right`） |
| `--shots`/`--seed` 等命令行参数错误（不读取输入） | 2 | 参数用法错误 |

## 现有公开接口

- 命令行程序 `quantum-circuit-simulator`（`version`、`simulate` 与 `equivalent` 子命令）
- Python 包 `quantum_circuit`，其 `__version__` 为当前版本号

## 限制

- 仅支持 OpenQASM 2.0 的上述子集：`x`、`h`、`cx` 与参数化门 `rx`、`ry`、`rz`，以及按位测量，不支持条件执行或多寄存器。
- 状态向量随量子位数指数增长，寄存器大小上限为 20；密度矩阵噪声仿真上限为 10 个量子位；`equivalent` 稠密酉矩阵比较上限为 8 个量子位。
