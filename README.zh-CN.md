# MOview

[English](README.md) | **简体中文**

MOview 是一个用 Python 编写的分子轨道波函数读取与 OpenGL 可视化程序。它可以读取 Gaussian formatted checkpoint（`.fchk`、`.fch`）和 Molden 文件，计算指定分子轨道在三维网格上的波函数值，提取正、负等值面，并同时显示分子结构、轨道表面、原子标签和坐标轴。

程序同时提供 GUI 与批处理模式：GUI 用于交互式查看轨道，批处理模式用于解析检查、自动化测试和输出等值面统计信息。

## 主要功能

- 读取 Gaussian FCHK/FCH 与 Molden 波函数。
- 显示 alpha/beta 轨道、能量、占据数、HOMO 和 LUMO。
- 同时比较最多 9 个轨道，并同步或独立旋转视图。
- 使用明确的正数等值面值绘制正、负波函数表面，默认值为 `0.05`。
- 调整原子大小；默认比例为 `1.00x`。
- 原子标签支持 `Off`、`Number`、`Element`、`Number + element`，例如 `1`、`Ca`、`1Ca`；标签字号可调。
- 标签定位支持默认的 `Attached` 和兼容的 `Floating`：Attached 贴在原子三维坐标上并参与透视，Floating 始终显示在表层并可能有引导线。
- 原子风格支持 `Ball & stick`、`Space filling`、`Licorice`。
- 等值面风格支持 `Glass`、`Solid`、`Wireframe`、`Solid + edges`。
- 正、负相颜色可从带色块的命名预设中独立选择；配置文件可修改预设 RGB 或添加新颜色。
- Grid 不设应用层面的上限；当 Grid 大于 256 时，在计算前显示性能和内存警告。
- 修改 Grid、Margin 或 Isovalue 后，程序会在输入停止 650 ms 后取消旧任务并重新预渲染附近轨道。
- 大文件在工作线程中解析，避免阻塞 Qt 主界面。
- 网格计算、等值面缓存和后台预取均有内存/分辨率保护。

## 安装

建议先进入用于运行 MOview 的 Conda 环境，再从项目根目录安装：

```bash
git clone https://github.com/Alfred-YQLi/py-moview.git
cd py-moview
conda activate wavefunction
python -m pip install -e ".[gui]"
```

核心依赖为 `numpy` 和 `scikit-image`。GUI 还需要 `PyQt6`、`PyOpenGL` 和 `pyqtgraph`。

## 使用方法

未安装软件包时，需要在项目根目录（包含 `pyproject.toml` 和 `moview/` 包目录的位置）运行：

```bash
cd /path/to/py-moview
python -m moview tests/wavefunctions/save_uks_UNO.fch
```

安装为 editable package 后，也可以在任意目录使用控制台命令：

```bash
moview /path/to/wavefunction.fch
```

不要进入 `moview/` 包目录后执行 `python -m moview`，也不要把波函数路径放在 `python -m` 后当作模块名。`-m moview` 启动的是 Python 包，波函数路径是它后面的普通参数。

### 直接加入 PATH

仓库提供可执行入口 `bin/moview`。它会根据脚本自身位置找到项目，因此不安装 editable package 也可以从任意目录启动：

```bash
export PATH="/path/to/py-moview/bin:$PATH"
moview /path/to/wavefunction.fch
```
启动前仍应激活安装了 GUI 依赖的 Conda 环境；入口使用当前 PATH 中的 `python3`。

常用命令：

```bash
python -m moview --help
python -m moview file.fch --fchk
python -m moview file.molden.input --molden
python -m moview file.fch --grid 96 --margin 4
python -m moview file.fch --prefetch-workers 4
python -m moview file.fch --config /path/to/custom.ini
python -m moview file.fch --no-auto-render
```

### 批处理模式

批处理模式不加载 PyQt/OpenGL，会输出文件格式、原子数、基函数数、轨道信息、实际网格形状、等值面值和三角形数量：

```bash
python -m moview tests/wavefunctions/save_uks.fch --batch --grid 16 --orbital 212
python -m moview tests/wavefunctions/save.molden.input --batch --grid 16 --orbital 212
python -m moview file.fch --batch --spin beta --orbital 52 --grid 64 --iso 0.03 --margin 4
```

批处理模式的轨道编号从 1 开始。Grid 最小值为 8，Margin 不能为负数，Isovalue 必须是有限正数且默认为 `0.05`。Grid 大于 256 时会在 stderr 输出性能警告，但批处理仍按请求继续计算。

## 配置文件

项目根目录提供 `moview.example.ini` 模板。复制为用户专用的 `moview.ini` 后再修改；`moview.ini` 已被 Git 忽略，不会意外提交机器相关的核心数或内存设置：

```bash
cp moview.example.ini moview.ini
```

程序按以下顺序寻找配置，找到第一个后停止：

1. 命令行 `--config PATH`。
2. 环境变量 `MOVIEW_CONFIG`。
3. 当前目录的 `moview.ini`。
4. macOS 的 `~/Library/Application Support/MOview/config.ini`。
5. `$XDG_CONFIG_HOME/moview/config.ini`，默认是 `~/.config/moview/config.ini`。
6. 项目根目录的 `moview.ini`。

显式命令行参数优先于配置，例如 `--grid 96` 会覆盖 `[render] grid`。没有找到配置文件时使用相同的内置默认值；修改配置后需要重新启动程序。

### 资源参数

`[resources]` 中的内存单位均为 MiB：

| 参数 | 作用 |
| --- | --- |
| `basis_workers` | 构建缓存 BasisGrid 时使用的线程数。 |
| `background_jobs` | 同时运行的后台预渲染批次数；默认 1。 |
| `basis_cache_mib` | BasisGrid 缓存总预算。 |
| `max_basis_cache_entry_mib` | 单个 BasisGrid 可进入缓存的最大大小，不能超过总预算。 |
| `render_cache_mib` | 轨道标量场和等值面缓存总预算。 |
| `render_cache_entries` | 渲染缓存条目数的附加上限。 |
| `prefetch_field_budget_mib` | 后台预渲染可规划的标量场总预算。 |
| `max_prefetch_orbitals` | 一次预渲染最多排队的轨道数。 |
| `grid_chunk_points` | 分块计算一次处理的网格点数；较小值降低临时内存，较大值可能提高吞吐。 |
| `surface_face_limit` | 每个正/负相上传到 OpenGL 的最大三角形数。 |

`background_jobs` 只改变调度并发度，不会绕过 `prefetch_field_budget_mib`。极端参数仍受实际 CPU、内存和显存限制。

### 默认显示与颜色

`[render]` 可设置默认 Grid、Margin、Isovalue、表面/原子风格、原子大小、标签内容、`label_placement = attached|floating` 和正负相颜色；`[view]` 可设置 Zoom、视图同步、坐标轴和自动首帧渲染。

`[colors]` 使用“名称 + 0 到 255 的 RGB”定义颜色。同名条目修改内置预设，新名称会追加到 GUI 选择器：

```ini
[render]
positive_color = Mint
negative_color = Violet

[colors]
Red = 255, 64, 64
Mint = 40, 210, 160
```

颜色引用不区分大小写，但 GUI 和顶部图例使用配置中定义的名称。非法 RGB、未知设置、未知颜色和矛盾的内存预算会在启动时给出明确错误。

## GUI 操作

左侧面板始终保留文件信息、HOMO/LUMO、比较输入和轨道表。设置被分成两个标签页。

### Compute

- `Spin`：在 unrestricted 波函数中选择 alpha 或 beta。
- `Grid`：最长方向上的目标网格点数；实际三维形状根据分子包围盒比例生成。
- `Margin / bohr`：分子包围盒外扩距离。
- `Isovalue`：正数等值面值；默认 `0.05`，可用滑块或数值输入修改。

修改 Grid、Margin 或 Isovalue 时，旧的后台预渲染会立即失效；输入停止 650 ms 后，程序按新参数重新预渲染。Grid 大于 256 时，预渲染开始前也会先显示性能确认框。

### Display

- `Surface`：切换 Glass、Solid、Wireframe、Solid + edges。
- `+ phase` / `- phase`：两个带色块选择器分别设置正、负波函数相位颜色；只重建 OpenGL 显示项。
- `Atoms`：切换 Ball & stick、Space filling、Licorice。
- `Atom size`：缩放原子半径，不改变键半径，也不会重新计算轨道。
- `Labels` 左侧选择器：关闭标签，或显示编号、元素、编号+元素。
- `Labels` 右侧选择器：`Attached` 将标签贴在原子前表面，随原子旋转并按距离产生透视缩放，且可被更近的几何体遮挡；`Floating` 使用屏幕避让和引导线，适合标签密集时阅读。
- `Label size`：调整标签字号。
- `Zoom`：调整视图缩放。
- `Sync views`：比较多个轨道时同步旋转与缩放。
- `Axes`：显示或隐藏右下角坐标轴。

外观设置只重建轻量的 OpenGL 显示项。切换原子大小、原子风格、标签内容、标签定位、标签字号、等值面风格或相位颜色，不会重新解析文件、计算轨道网格或运行 marching cubes。Attached 模式把全部标签合并为一个纹理图集和一次 OpenGL 绘制；标签关闭时不创建任何标签资源。

视图区顶部使用两行紧凑信息栏：第一行显示轨道、占据数和能量，第二行显示 Isovalue、Grid 以及带色块的正负相颜色名称。比较模式中每个子视图只保留一行轨道摘要。

### 轨道比较

`Compare` 输入使用从 1 开始的轨道编号，支持逗号、空格和范围：

```text
45,46,47
45 46 47
45-49
49-45
```

### 快捷键

- `A` / `D`：选择前一个/后一个轨道并延迟渲染。
- 方向键：旋转当前视图。
- `+` / `-`：缩放。
- `C`：进入旋转中心选择模式，然后点击原子。

当焦点位于轨道表、输入框、下拉框、数值框或滑块上时，全局快捷键不会抢占这些控件的键盘操作。

## Grid 与性能

Grid 表示分子包围盒最长轴上的点数。计算量和最终标量场内存近似随 Grid 的三次方增长，因此“没有软件上限”并不代表任意值都能在当前机器内存中完成。

- GUI 在 Grid 大于 256 时弹出一次确认框，并显示实际三维点数和单个 float32 标量场的估算内存。
- 同一个 Grid 值确认后，本次文件会话中不重复弹窗；切换文件后重新确认。
- 默认情况下，预计完整 `BasisGrid` 不超过 640 MiB 时，GUI 使用基函数缓存，以保持低分辨率和多轨道切换速度；阈值可在配置中修改。
- 超过该阈值时，GUI 使用 float32 分块、多轨道批量计算，不构造 `n_basis x n_points` 的巨型矩阵。
- 默认基函数缓存总预算为 768 MiB，轨道/等值面缓存总预算为 512 MiB。
- 后台预渲染不使用固定 Grid 截止值。默认最多处理 48 个 HOMO/LUMO 附近轨道，并按 192 MiB 标量场预算动态减少任务数。
- 当单个标量场已经超过 192 MiB 时，程序跳过后台预渲染，但仍允许用户通过 `Render` 主动计算。
- 已缓存轨道只在等值面值变化时重新运行 marching cubes，不会重复计算波函数标量场。
- 异步计算绑定到当前波函数文件代次；切换文件后，旧任务不能写入新文件的基函数缓存。
- 默认关闭原子标签，因此新增标签布局不会影响默认旋转性能。

## 主程序工作流程

### GUI 调用顺序

1. `moview/__main__.py` 调用 `moview.cli.main()`。
2. `moview/cli.py` 先调用 `moview.config.load_config()`，再用配置值构造完整参数解析器；显式 CLI 参数覆盖配置。
3. GUI 模式启用 macOS 原生 stderr 过滤，再导入 `moview.gui.main_window.run_gui()`。
4. `run_gui()` 创建 `QApplication` 和 `OpenGLViewer`，资源预算、默认样式和颜色随 `AppConfig` 注入。
5. `moview/gui/layout.py` 创建侧栏、Compute/Display 控件、颜色选择器、轨道表和 OpenGL 视图区。
6. `OpenGLViewer.load_wavefunction()` 在线程池中调用 `moview.parsers.parse_wavefunction()`，避免主线程卡顿。
7. `moview/parsers/__init__.py` 检测格式并分派给 `FCHKParser` 或 `MoldenParser`。
8. 解析器构造 `moview.wavefunction.Wavefunction`，其中包含原子、基组壳层、能量、占据数和 MO 系数矩阵。
9. `moview.analysis.compute_bonds()` 根据共价半径和原子距离建立键列表。
10. `populate_orbitals()` 填充轨道表，并选择默认 HOMO。
11. `render_selected()` 解析单轨道或比较轨道，检查高 Grid 警告并查询轨道缓存。
12. 小型网格通过 `compute_basis_grid()` 与 `compute_orbital_grid_from_basis()` 计算；大型网格通过 `compute_orbital_grid_float32()` 或 `compute_orbital_grids_float32()` 分块计算。
13. `moview.surface.extract_isosurfaces()` 使用 marching cubes 生成正、负 `SurfaceMesh`。
14. `moview.gui.gl_view` 把 `SurfaceMesh`、原子和键转换为 OpenGL 项；Attached 标签使用单个深度测试的纹理图集批次，Floating 标签使用单次 QPainter 避让绘制。
15. `SceneSlot` 保存每个比较视图的网格、表面、旋转、相机和显示项；主窗口更新标题、状态和缓存。
16. 后台按内存预算预渲染 HOMO/LUMO 附近轨道；Grid、Margin 或 Isovalue 改动后通过防抖计时器重启，前台渲染始终优先。

### 批处理调用顺序

1. `moview.cli.main()` 加载配置，用配置构造默认参数并校验显式 CLI 覆盖值。
2. `run_batch()` 调用 `parse_wavefunction()` 返回 `Wavefunction`。
3. `moview.grid.compute_orbital_grid()` 按配置的 `grid_chunk_points` 分块生成点，并以 float64 计算指定轨道。
4. `extract_isosurfaces()` 生成正、负等值面。
5. `run_batch()` 打印轨道、网格和三角形统计。

## 目录与模块说明

### 包入口与数据模型

| 文件 | 作用 |
| --- | --- |
| `bin/moview` | 可直接加入 PATH 或建立符号链接的源码入口。 |
| `moview/__init__.py` | 对外导出核心数据结构、解析、网格和表面 API。 |
| `moview/__main__.py` | `python -m moview` 的包入口。 |
| `moview.example.ini` | 可复制的资源、渲染、视图和命名颜色配置模板。 |
| `moview/cli.py` | 配置优先加载、参数解析、批处理、高 Grid CLI 警告和 GUI 懒加载。 |
| `moview/config.py` | 配置发现、类型/范围校验、不可变设置模型和命名 RGB 颜色。 |
| `moview/wavefunction.py` | `Shell`、`Wavefunction` 数据模型及轨道能量/占据访问。 |
| `moview/constants.py` | 单位换算、元素符号、元素颜色、共价半径和核心常量。 |
| `moview/cache.py` | 配置可写的 Matplotlib/XDG 运行时缓存目录。 |

### 波函数解析

| 文件 | 作用 |
| --- | --- |
| `moview/parsers/__init__.py` | 文件格式检测与解析器分派。 |
| `moview/parsers/common.py` | Fortran `D` 指数等数值文本转换。 |
| `moview/parsers/fchk.py` | Gaussian formatted checkpoint 解析。 |
| `moview/parsers/molden.py` | Molden 的 Atoms、GTO、MO 和球谐标记解析。 |
| `moview/parsers/orca.py` | ORCA 解析的预留模块；当前没有实现或接入 ORCA 格式。 |

### 基函数、网格与表面

| 文件 | 作用 |
| --- | --- |
| `moview/basis/__init__.py` | 汇总导出 Gaussian、球谐和 float32/float64 求值 API。 |
| `moview/basis/gaussian.py` | Gaussian primitive 归一化、Cartesian 幂和壳层函数数目。 |
| `moview/basis/spherical.py` | spherical-to-Cartesian 变换矩阵。 |
| `moview/basis/evaluate.py` | float64/float32 的壳层、单轨道和多轨道求值。 |
| `moview/grid.py` | `GridSpec`、分块网格点、BasisGrid 和轨道网格计算。 |
| `moview/surface.py` | `SurfaceMesh`、marching cubes、可选平滑和批处理表面入口。 |
| `moview/analysis/__init__.py` | 导出几何分析 API。 |
| `moview/analysis/geometry.py` | 基于共价半径的分子键识别。 |

### GUI

| 文件 | 作用 |
| --- | --- |
| `moview/gui/main_window.py` | 主窗口状态、异步任务、缓存、颜色、信息栏、比较视图、预取和相机控制。 |
| `moview/gui/layout.py` | Compute/Display 控件、颜色选择器、轨道表、紧凑信息栏和画布布局。 |
| `moview/gui/gl_view.py` | OpenGL 视图、fog shader、表面材质、分子几何、透视贴附/浮动批量标签和鼠标交互。 |
| `moview/gui/widgets.py` | 数值滑块、角落坐标轴、视图宿主和 `SceneSlot` 状态。 |
| `moview/gui/presentation.py` | 显示风格/标签选项、高 Grid 阈值和警告文本。 |
| `moview/gui/styles.py` | Qt 样式表和 GUI 配色。 |
| `moview/gui/native_stderr.py` | 精确过滤已知无害的 macOS TSM/IMK 和 Qt keymapper 输入法诊断，保留 Python traceback、OpenGL 错误和其他 stderr。 |
| `moview/gui/__init__.py` | GUI API 的懒加载导出，确保批处理不要求 PyQt。 |

### 测试

| 文件 | 作用 |
| --- | --- |
| `tests/test_config.py` | 配置发现、覆盖优先级、资源/颜色解析、错误输入和 CLI 默认值。 |
| `tests/test_smoke.py` | 常量、标签、真实高 Grid 规格、Molden 错误输入和 macOS 日志过滤。 |
| `tests/test_gui.py` | GUI 默认值、资源配置、颜色/标题同步、无上限 Grid、预渲染、缓存、异步加载、快捷键和显示样式。无 GUI 依赖时自动跳过。 |
| `tests/test_wavefunctions.py` | 解析所有样例，并为 FCHK/Molden 生成代表性低 Grid 等值面。 |
| `tests/wavefunctions/` | 自动测试使用的本地波函数文件。 |

`tests/wavefunctions/` 中的大型波函数是本地测试数据，不提交到 Git。将文档中列出的样例放入该目录后会自动运行格式集成测试；样例缺失时这些测试明确跳过。

## 开发检查

基础环境：

```bash
python -m compileall -q -x '(^|/)\._' moview tests
python -m unittest discover -s tests -v
python -m moview tests/wavefunctions/save_uks.fch --batch --grid 16 --orbital 212
python -m moview tests/wavefunctions/save.molden.input --batch --grid 16 --orbital 212
```

GUI 环境中的离屏行为测试：

```bash
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v
```

离屏 Qt 插件通常不能创建真实 OpenGL context，因此会输出 `QOpenGLWidget is not supported on this platform`；GUI 行为测试仍可运行。真实 OpenGL 显示需要在正常 macOS/桌面会话中启动程序检查。

外置 macOS 磁盘有时会产生 AppleDouble `._*` 文件。`compileall` 的 `-x '(^|/)\._'` 用于跳过这些元数据文件。

editable 安装和源码入口可在这类外置卷上正常使用。正式构建发布 wheel 时，应使用 APFS 或其他不会在 setuptools 生成目录内创建 `._*` 文件的干净 Git checkout。

## 当前限制

- ORCA 波函数解析尚未实现。
- 极高 Grid 仍可能因物理内存不足失败；程序只取消人工上限，不绕过机器资源限制。
- marching cubes 需要 `scikit-image`。
- GUI 需要桌面 OpenGL 环境；纯 SSH/离屏平台可能无法创建 OpenGL context。
