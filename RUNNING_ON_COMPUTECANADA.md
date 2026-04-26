# 在 Compute Canada / Alliance 上运行前要做的事

本文说明：在**新终端**或 **Slurm 作业**里，跑 `experiment/` 下脚本（例如 `run_impact_microstructure.py`）之前，环境应如何准备。

---

## 核心原则（务必记住）

1. **`module load` 在前，`source venv` 在后**  
   若顺序反了，`which python` 往往会变成 **`/cvmfs/...` 系统 Python**，此时即使用 `pip` 把包装进了 `polyfem_env`，**`python your_script.py` 也找不到那些包**。

2. **改过 `PATH` 后执行 `hash -r`（bash）**  
   Shell 会缓存「`python` 对应哪个路径」。激活 venv 或 `module load` 之后用 **`hash -r`** 清缓存，避免 **`which python` 已对、实际跑的仍是旧路径**。（zsh 可用 **`rehash`**。）

3. **每次怀疑环境时先检查**：

   ```bash
   which python
   python -c "import sys; print(sys.executable)"
   ```

   应显示 **`/home/<你>/polyfem_env/bin/python`**（路径按你的 home 修改）。

4. **Gmsh**：本仓库实验代码 `import gmsh`。在 Alliance 自带 Python 上，`pip install gmsh` 常因 wheel / 平台标签策略失败，推荐用 **Gmsh 官方 Linux64 SDK + `PYTHONPATH`**（见下文）。可选：**改 venv 的 `activate`**，每次激活自动 `source` Gmsh 脚本（见 **C-2**）。

---

## 一次性准备（只需做一次）

### A. Python 虚拟环境 `polyfem_env`

若尚未创建（示例）：

```bash
module load StdEnv/2023
python3 -m venv "$HOME/polyfem_env"
```

在 venv 里安装构建与运行依赖（路径用 **`$HOME/polyfem_env/bin/python -m pip`** 更稳）：

```bash
unset PIP_CONFIG_FILE
"$HOME/polyfem_env/bin/python" -m pip install -U pip setuptools wheel nanobind numpy Cython
```

若实验/后处理需要读取 **VTU**（例如 `*.vtu`），还需要可选依赖 **`meshio`**：

```bash
"$HOME/polyfem_env/bin/python" -m pip install meshio
# 或： "$HOME/polyfem_env/bin/python" -m pip install 'polyfempy[io]'
```

从源码装 **Shapely** 前通常需要：

```bash
module load StdEnv/2023 geos
"$HOME/polyfem_env/bin/python" -m pip install --no-build-isolation "shapely>=2" -i https://pypi.org/simple
```

**可微实验**需要 **PyTorch** 时（只装 `torch`、不必装 `torchvision`）：

- **优先**：保留联盟 pip 配置，让 wheelhouse 参与解析：

  ```bash
  export PIP_CONFIG_FILE=/cvmfs/soft.computecanada.ca/config/python/pip-x86-64-v4-gentoo2023.conf
  ```

  （若在 **v3** 栈，把文件名改成 **`pip-x86-64-v3-gentoo2023.conf`**，在 `/cvmfs/soft.computecanada.ca/config/python/` 下 `ls pip-*` 查看。）

  ```bash
  "$HOME/polyfem_env/bin/python" -m pip install torch
  ```

- **只要 CPU、且上一步装不上**：`"$HOME/polyfem_env/bin/python" -m pip install torch --index-url https://download.pytorch.org/whl/cpu`

- **不要**在一条命令里写 `torchvision`，除非代码真的 `import torchvision`。

### B. 编译并 editable 安装 `polyfempy`

**【已 module · 仓库根】**

```bash
module load StdEnv/2023 cmake
# 仓库在 scratch 下：若当前已在 scratch，用 ``cd polyfem-python``；否则例如：
cd "$HOME/scratch/polyfem-python"
unset PIP_CONFIG_FILE
"$HOME/polyfem_env/bin/python" -m pip install -e . --no-build-isolation
```

> `ninja` 仅在你选择用 Ninja 作为 CMake generator 时需要；而且在部分 Alliance 节点上 **`module load ninja` 可能不可用**。若确实需要，请先用 `module spider ninja` 查正确加载方式；否则不必加载。

> 若 `setup.py` 仍误报 `polyfem_env` 路径：请使用仓库里已修复的版本（只拒绝 `<仓库>/env/bin`，不误判 `polyfem_env`）。

### C. Gmsh SDK + 环境脚本

#### C-1. 创建 `$HOME/env_polyfem_gmsh.sh`

1. 下载并解压官方 SDK（版本号以 [gmsh.info](https://gmsh.info/#Download) 为准），例如放到 `$HOME/opt/gmsh/`。
2. 创建脚本（**把 `GMSH_SDK` 改成你本机解压路径**）：

```bash
cat > "$HOME/env_polyfem_gmsh.sh" <<'EOF'
export GMSH_SDK="$HOME/opt/gmsh/gmsh-4.15.2-Linux64-sdk"
export PYTHONPATH="$GMSH_SDK/lib${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$GMSH_SDK/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$GMSH_SDK/bin${PATH:+:$PATH}"
EOF
chmod +x "$HOME/env_polyfem_gmsh.sh"
```

#### C-2.（推荐）激活 venv 时自动加载 Gmsh

在 **`$HOME/polyfem_env/bin/activate` 文件末尾**追加：

```bash
# Auto-load Gmsh SDK when using this venv
if [ -f "${HOME}/env_polyfem_gmsh.sh" ]; then
    . "${HOME}/env_polyfem_gmsh.sh"
fi
```

这样 **`source .../polyfem_env/bin/activate` 后不必再手敲** `source env_polyfem_gmsh.sh`。  
**注意**：`deactivate` **不会**自动撤销 Gmsh 对 `PYTHONPATH` / `LD_LIBRARY_PATH` 的修改；需要干净环境时 **开新终端** 或手动 `unset`/重登。

---

## 每次开新终端：运行前要执行的顺序

**【登录节点或计算节点 · 交互 shell】**

若已做 **C-2**，可跳过下面注释里的第 4 步。

```bash
# 1) 模块（按需要增减）
module load StdEnv/2023

# 2) 激活 venv（必须在 module 之后）
source "$HOME/polyfem_env/bin/activate"
hash -r

# 3) 确认 Python 是 venv 的（必做）
which python
python -c "import sys; print(sys.executable)"

# 若第 3 步不是 polyfem_env，再执行：
#   source "$HOME/polyfem_env/bin/activate" && hash -r
# 或： export PATH="$HOME/polyfem_env/bin:$PATH" && hash -r

# 4) Gmsh SDK（若未在 activate 里自动加载，则保留此行）
# source "$HOME/env_polyfem_gmsh.sh"

# 5) 进仓库（不要用字面量 ``/path/to/...``；在登录节点上常见写法如下）
#     若提示符已在 ``scratch`` 目录：``cd polyfem-python``
#     若在家目录或其它位置：``cd "$HOME/scratch/polyfem-python"``（或 ``cd "/scratch/$USER/polyfem-python"``，以你 clone 位置为准）
cd polyfem-python

# 6) 可选：快速自检
python -c "import gmsh, numpy, shapely, polyfempy; print('imports OK')"

# 7) 运行示例
python experiment/new_experiment/run_impact_microstructure.py
```

Experiment 02 的 cross-body shape-gradient 诊断可以这样跑：

```bash
python -m experiment.experiment_api_solve.diagnose_experiment02_cross_body_shape_gradient
```

它会输出 `cross_body_shape_gradient_probe.json` 和
`cross_body_shape_gradient_probe.txt`，用于比较 finite difference 与当前
adjoint gradient。

更稳妥（**不依赖** `PATH` 上的 `python` 名字）：

```bash
source "$HOME/env_polyfem_gmsh.sh"   # 若 activate 已自动加载 Gmsh，可省略
cd polyfem-python   # 需已在 scratch；否则 ``cd "$HOME/scratch/polyfem-python"``
"$HOME/polyfem_env/bin/python" experiment/new_experiment/run_impact_microstructure.py
```

---

## 使用 `pip` 与「不只联盟源」

- 联盟会通过 **`PIP_CONFIG_FILE`**、`find-links`（wheelhouse）、**`constraints.txt`** 控制解析；**`python -m pip config list`** 可查看当前源与约束。
- **装联盟维护好的包（如 numpy、部分 torch）**：通常 **保留** `PIP_CONFIG_FILE` 再 `pip install` 更稳。
- **联盟 wheelhouse 没有、但 PyPI 有的包**：可叠加 PyPI，例如：

  ```bash
  python -m pip install 某包 --extra-index-url https://pypi.org/simple
  ```

- **想尽量只用 PyPI**：可 **`unset PIP_CONFIG_FILE`** 再用 **`-i https://pypi.org/simple`**；但 **torch / gmsh** 等可能反而 **`versions: none`** 或装到不兼容版本，需谨慎。

---

## Slurm 批作业模板

将下面中的路径、账号、`#SBATCH` 行按集群说明修改。

```bash
#!/bin/bash
#SBATCH --job-name=polyfem_exp
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail

module load StdEnv/2023

source "$HOME/polyfem_env/bin/activate"
hash -r
source "$HOME/env_polyfem_gmsh.sh"

cd "${SLURM_SUBMIT_DIR:-$HOME/scratch/polyfem-python}"
"$HOME/polyfem_env/bin/python" experiment/new_experiment/run_impact_microstructure.py
```

建议在作业头部增加 `#SBATCH --output=...`、`#SBATCH --error=...`，并把路径指到仓库根下的 **`slurm_logs/`**（提交前执行一次 `mkdir -p slurm_logs`；本仓库已跟踪该目录）。`theta_failure_retest` 的现成脚本见 `scripts/slurm/sbatch_run_theta_failure_retest.sh`。

> 若已在 **activate** 里自动 `source env_polyfem_gmsh.sh`，作业里仍建议 **显式再 `source` 一次**，避免个别批处理环境未走交互式 `activate` 的完整逻辑。

---

## 常见问题

| 现象 | 处理 |
|------|------|
| `ModuleNotFoundError: polyfempy` | `which python` 是否指向 venv；必要时 **`pip install -e . --no-build-isolation`** |
| `No module named pip` | 用了系统 `python`，改用 **`$HOME/polyfem_env/bin/python -m pip`** |
| `which python` 是 `/cvmfs/...` | **`module load` 之后再 `source activate`**，**`hash -r`**，或 **`export PATH=$HOME/polyfem_env/bin:$PATH`** |
| `pip install gmsh` 失败 | 用 **Gmsh SDK + `env_polyfem_gmsh.sh`**；**不要**依赖 PyPI 上仅 manylinux 的 wheel（与当前 pip 标签常不兼容） |
| `pip install torch` 报 `versions: none` | **保留** `PIP_CONFIG_FILE` 再装；或 **`--index-url https://download.pytorch.org/whl/cpu`**；勿盲目长期 **`unset PIP_CONFIG_FILE`** |
| 编 Shapely 缺 `geos_c.h` | **`module load geos`** 后再 **`pip install shapely --no-build-isolation`** |
| `solve_differentiable` 要 PyTorch | **`pip install torch`**（见上文 **A**），不必默认装 `torchvision` |
| `Reading VTU requires meshio` | 在 venv 安装 **`meshio`**：`python -m pip install meshio`（或 `pip install 'polyfempy[io]'`） |

---

## 路径速查（请改成你的用户名）

| 项 | 示例路径 |
|----|-----------|
| venv | `$HOME/polyfem_env` |
| 仓库 | `/scratch/<用户>/polyfem-python` 或 `$HOME/scratch/polyfem-python`；在 `scratch` 下可直接 **`cd polyfem-python`** |
| Gmsh SDK | `$HOME/opt/gmsh/gmsh-*-Linux64-sdk` |

将本文中的 **`polyfem_env` 路径**、**仓库目录**（Alliance 上常见：先 **`cd scratch`** 再 **`cd polyfem-python`**，或 **`cd "$HOME/scratch/polyfem-python"`**）、**`GMSH_SDK`** 换成你自己的即可。**不要**使用不存在的占位路径 ``/path/to/polyfem-python``。
