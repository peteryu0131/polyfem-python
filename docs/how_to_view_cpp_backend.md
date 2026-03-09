# 如何查看 C++ 后端的 State::init() 方法

## 问题背景

当 Python 端传递正确的 JSON 配置给 C++ 后端时，C++ 后端的 `State::init()` 方法会在解析 JSON 时自动填充所有可选字段的默认值。这导致 JSON schema 验证失败，因为某些字段（如 `ADAM`、`L-BFGS`、`Newton` 等）应该是互斥的，但 C++ 后端同时填充了所有字段。

## 查找 State::init() 方法的位置

### 方法 1: 查看 PolyFEM GitHub 仓库（推荐）

PolyFEM 的源代码托管在 GitHub 上。根据 `cmake/recipes/polyfem.cmake`，项目使用的是特定 commit：

```cmake
CPMAddPackage("gh:polyfem/polyfem#e8bd3d3")
```

**步骤：**

1. 访问 PolyFEM GitHub 仓库：
   ```
   https://github.com/polyfem/polyfem
   ```

2. 切换到对应的 commit：
   ```
   https://github.com/polyfem/polyfem/tree/e8bd3d3
   ```

3. 查找 `State` 类定义：
   - 搜索文件：`State.hpp` 和 `State.cpp`
   - 通常位于：`src/polyfem/state/` 目录下

4. 查看 `State::init()` 方法：
   - 在 `State.cpp` 中查找 `void State::init(const json &args, bool strict_validation)` 方法
   - 查看它如何解析 JSON 和填充默认值

### 方法 2: 在本地构建目录查找

如果项目已经构建过，源代码可能在以下位置：

1. **CMake 构建目录**：
   ```
   _deps/polyfem-src/  (如果 CPM 下载了源代码)
   ```

2. **检查 CPM 缓存**：
   CPM 通常将源代码缓存在：
   ```
   ~/.cache/CPM/  (Linux/Mac)
   %LOCALAPPDATA%\CPM\  (Windows)
   ```

3. **查找 State 相关文件**：
   ```powershell
   # Windows PowerShell
   Get-ChildItem -Path . -Recurse -Filter "State.*" -ErrorAction SilentlyContinue | Select-Object FullName
   ```

### 方法 3: 直接克隆 PolyFEM 仓库

如果需要完整查看源代码：

```bash
git clone https://github.com/polyfem/polyfem.git
cd polyfem
git checkout e8bd3d3
```

然后查看：
- `src/polyfem/state/State.hpp` - 类定义
- `src/polyfem/state/State.cpp` - 实现，特别是 `init()` 方法

## 关键代码位置

### State::init() 方法实现

**文件位置**: `polyfem/src/polyfem/state/StateInit.cpp` (第 133-349 行)

**方法签名**:
```cpp
void State::init(const json &p_args_in, const bool strict_validation)
```

**Python 绑定调用** (`polyfem-python/src/state/state.cpp` 第 130 行):
```cpp
self.init(json::parse(json_string), strict_validation);
```

### State::init() 完整执行流程

1. **处理 common.json** (第 139 行)
   ```cpp
   apply_common_params(args_in);
   ```
   - 实现位置: `polyfem/src/polyfem/utils/JSONUtils.cpp` (第 14-59 行)
   - 如果 JSON 包含 `"common"` 字段，加载并合并 common.json
   - 使用 `merge_patch` 合并，当前配置优先覆盖 common.json

2. **加载 JSON Schema** (第 141-163 行)
   ```cpp
   jse::JSE jse;
   jse.strict = strict_validation;
   const std::string polyfem_input_spec = POLYFEM_INPUT_SPEC;
   // 加载 input-spec.json
   rules = jse.inject_include(rules);
   ```
   - Schema 文件: `polyfem/json-specs/input-spec.json`
   - 包含对 `nonlinear-solver-spec.json` 的引用 (通过 `polysolve.json`)

3. **处理 Solver 配置** (第 165-188 行)
   ```cpp
   polysolve::linear::Solver::select_valid_solver(args_in["solver"]["linear"], logger());
   // 处理 augmented_lagrangian/nonlinear 的默认值
   ```

4. **JSON Schema 验证** (第 189-195 行)
   ```cpp
   const bool valid_input = jse.verify_json(args_in, rules);
   if (!valid_input) {
       logger().error("invalid input json:\n{}", jse.log2str());
       throw std::runtime_error("Invald input json file");
   }
   ```
   - **关键**: 验证发生在填充默认值**之前**

5. **注入默认值** (第 198 行) ⚠️ **问题根源**
   ```cpp
   this->args = jse.inject_defaults(args_in, rules);
   ```
   - **这是导致问题的关键步骤**
   - `jse.inject_defaults()` 会填充 schema 中定义的所有默认值
   - 包括互斥字段（`ADAM`、`L-BFGS`、`Newton` 等）的默认值
   - 导致这些字段同时存在，违反 schema 的互斥要求

6. **后续初始化** (第 199-348 行)
   - 初始化单位系统
   - 设置输出目录和 logger
   - 初始化时间设置
   - 初始化 assembler 和 problem

### 相关文件位置

1. **State 类定义**: `polyfem/src/polyfem/State.hpp` (第 78-95 行)
2. **State::init() 实现**: `polyfem/src/polyfem/state/StateInit.cpp` (第 133-349 行)
3. **apply_common_params()**: `polyfem/src/polyfem/utils/JSONUtils.cpp` (第 14-59 行)
4. **Python 绑定**: `polyfem-python/src/state/state.cpp` (第 123-131 行)
5. **JSON Schema**: `polyfem/json-specs/input-spec.json`
6. **Nonlinear Solver Schema**: 通过 `polyfem/json-specs/polysolve.json` 引用 `nonlinear-solver-spec.json`

### JSON Schema 结构

- `/solver/nonlinear` 在 `input-spec.json` 中定义为可选字段 (第 1449 行)
- 通过 `polysolve.json` 引用 `nonlinear-solver-spec.json` (第 12-16 行)
- `nonlinear-solver-spec.json` 定义了 `ADAM`、`L-BFGS`、`Newton` 等互斥字段
- Schema 使用 `oneOf` 或类似规则要求这些字段互斥

## 问题根源分析

### 为什么会出现这个问题？

1. **验证和填充的顺序问题**:
   - 第 189 行: `jse.verify_json(args_in, rules)` - 验证**输入**的 JSON
   - 第 198 行: `jse.inject_defaults(args_in, rules)` - 填充**所有**默认值
   - 验证发生在填充之前，但填充后没有再次验证

2. **jse.inject_defaults() 的行为**:
   - 该方法会遍历 schema 中定义的所有字段
   - 对于每个有默认值的字段，如果输入 JSON 中没有该字段，就填充默认值
   - **问题**: 它不会检查字段之间的互斥关系（如 `oneOf` 规则）
   - 因此会同时填充 `ADAM`、`L-BFGS`、`Newton` 等所有 solver 的默认值

3. **Schema 的互斥要求**:
   - `nonlinear-solver-spec.json` 中定义了这些 solver 字段为互斥
   - 使用 `oneOf` 或类似规则要求只能存在一个
   - 当 `jse.inject_defaults()` 填充所有默认值后，所有字段同时存在，违反互斥规则

### jse 库说明

- `jse::JSE` 是 PolyFEM 使用的 JSON Schema 验证和默认值注入库
- 通过 CMake 依赖管理引入: `cmake/recipes/jse.cmake`
- 主要方法:
  - `jse.verify_json(json, rules)`: 验证 JSON 是否符合 schema
  - `jse.inject_defaults(json, rules)`: 注入 schema 中定义的默认值
  - `jse.inject_include(rules)`: 处理 schema 文件中的 `include` 引用

## 可能的解决方案

### 方案 1: 修改 C++ 后端（如果可能）

如果能够修改 PolyFEM 源代码，可以：

1. **修改 `State::init()` 方法** (第 198 行后):
   ```cpp
   this->args = jse.inject_defaults(args_in, rules);
   // 添加: 清理互斥字段，只保留实际使用的
   ```

2. **修改 `jse.inject_defaults()` 逻辑**:
   - 使其在填充默认值时检查 `oneOf` 规则
   - 对于互斥字段，只填充一个（根据实际使用的或第一个）

3. **修改验证顺序**:
   - 在填充默认值后再次验证
   - 如果验证失败，清理互斥字段

### 方案 2: 在 Python 端预处理 JSON（当前实现）

在 `polyfempy/api/solve.py` 中已经实现：

1. **清理互斥字段** (第 341-349 行):
   ```python
   # 移除所有 solver 配置字段
   for k in ["ADAM", "L-BFGS", "L-BFGS-B", "Newton", ...]:
       del nonlinear[k]
   ```

2. **只保留实际使用的配置**:
   - 根据 `solver_type` 只保留对应的配置
   - 删除其他所有互斥字段

### 方案 3: 使用 common.json

通过添加 `common.json` 引用，让 C++ 后端自己处理合并：
- 优点: 利用 C++ 后端的合并逻辑
- 缺点: 仍然可能遇到默认值填充问题

## 调试技巧

### 1. 添加调试输出

在 `polyfempy/api/solve.py` 中，我们已经添加了调试输出：

```python
print("DEBUG: Final solver/nonlinear JSON (before sending to C++):")
print(json.dumps(solver_nonlinear_dict, indent=2))
```

### 2. 查看 C++ 后端接收到的 JSON

可以在 C++ 绑定代码中添加日志，查看 `State::init()` 接收到的 JSON。

### 3. 对比工作示例

参考 `examples/python_config_5_cubes.py`（用 API 类手动构建 cfg）的完整流程。

## 完整代码流程示例

### Python 端调用流程

```python
# polyfempy/api/solve.py
def solve(vertices=None, cells=None, cfg=None, ...):
    # 1. 将配置转换为 JSON
    full_json = cfg.to_dict()
    
    # 2. 处理 common.json 引用
    processed_json = _process_json_config(full_json, cfg)
    
    # 3. 清理互斥字段
    _clean_json_for_cpp(processed_json)
    
    # 4. 转换为字符串
    json_string = json.dumps(processed_json)
    
    # 5. 调用 C++ 后端
    solver.set_settings(json_string, strict_validation=False)
```

### C++ 后端处理流程

```cpp
// polyfem-python/src/state/state.cpp (第 130 行)
self.init(json::parse(json_string), strict_validation);

// polyfem/src/polyfem/state/StateInit.cpp (第 133 行)
void State::init(const json &p_args_in, const bool strict_validation) {
    json args_in = p_args_in; // mutable copy
    
    // 步骤 1: 处理 common.json
    apply_common_params(args_in);  // 第 139 行
    
    // 步骤 2: 加载 schema
    jse::JSE jse;
    jse.strict = strict_validation;
    // 加载 input-spec.json
    rules = jse.inject_include(rules);  // 第 159 行
    
    // 步骤 3: 验证输入 JSON
    const bool valid_input = jse.verify_json(args_in, rules);  // 第 189 行
    if (!valid_input) {
        throw std::runtime_error("Invald input json file");
    }
    
    // 步骤 4: 注入默认值 ⚠️ 问题发生在这里
    this->args = jse.inject_defaults(args_in, rules);  // 第 198 行
    // 此时 this->args 包含了所有默认值，包括互斥字段
    
    // 步骤 5: 后续初始化...
}
```

### apply_common_params() 实现

```cpp
// polyfem/src/polyfem/utils/JSONUtils.cpp (第 14-59 行)
void apply_common_params(json &args) {
    if (!args.contains("common"))
        return;
    
    // 加载 common.json
    const std::string common_params_path = resolve_path(args["common"], args["root_path"]);
    json common_params;
    file >> common_params;
    
    // 递归处理嵌套的 common.json
    apply_common_params(common_params);
    
    // 合并: common.json 作为 base，当前配置作为 override
    common_params.merge_patch(args);
    args = common_params;
    
    args.erase("common"); // 移除 common 字段
}
```

## 相关文件

### C++ 后端文件

- **State 类定义**: `polyfem/src/polyfem/State.hpp` (第 78-95 行)
- **State::init() 实现**: `polyfem/src/polyfem/state/StateInit.cpp` (第 133-349 行)
- **apply_common_params()**: `polyfem/src/polyfem/utils/JSONUtils.cpp` (第 14-59 行)
- **JSONUtils 头文件**: `polyfem/src/polyfem/utils/JSONUtils.hpp` (第 14 行)

### Python 绑定文件

- **Python 绑定**: `polyfem-python/src/state/state.cpp` (第 123-131 行)
- **Solve 函数**: `polyfempy/api/solve.py` (第 220-354 行)
- **配置类**: `polyfempy/api/config.py` (第 2447-2576 行)

### JSON Schema 文件

- **主 Schema**: `polyfem/json-specs/input-spec.json`
- **Polysolve Schema**: `polyfem/json-specs/polysolve.json` (引用 `nonlinear-solver-spec.json`)
- **CMake 配置**: `polyfem/CMakeLists.txt` (第 151 行定义 `POLYFEM_INPUT_SPEC`)

## 下一步

1. 访问 PolyFEM GitHub 仓库查看 `State::init()` 实现
2. 理解默认值填充逻辑
3. 确定是否有选项可以禁用默认值填充
4. 或者找到其他解决方案
