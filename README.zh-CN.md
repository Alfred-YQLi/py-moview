# MOview

[English](README.md) | **简体中文**

MOview 读取 Gaussian formatted checkpoint（`.fchk`/`.fch`）和 Molden 文件中的分子轨道，
在三维网格上计算轨道值，并在交互式 OpenGL 窗口中同时显示正、负等值面与分子结构。
批处理模式无需 GUI，并可检查文件是否被正确解析。

## 主要功能

- 支持 Gaussian FCHK/FCH 与 Molden（S–H 壳层、SP 壳层、笛卡尔与球谐函数、限制性与非限制性）。
- **自动识别 Molden 约定。** 不同程序对 Molden 格式中未明确规定的细节处理不同
  （ORCA 的 `orca_2mkl` 会把原始高斯函数归一化因子乘进收缩系数，并翻转 |m| ≥ 3 的 f/g 函数符号）。
  MOview 用解析重叠积分检查 MO 正交归一性（`C S Cᵀ = I`），自动选择正确的约定，
  并在文件面板和 `--batch` 输出中说明。
- 轨道表显示 HOMO/LUMO 标签（HOMO−n、LUMO+n）、占据数以及 Eh 和 eV 能量；
  单击即渲染，⌘/Ctrl 或 Shift 多选可并排比较最多 9 个轨道，也可输入 `210-214`。
- 等值面风格：Glass、Solid、Wireframe、Solid + edges；平滑法线、双面光照与深度雾化；正负相颜色可配置。
- 原子风格：Ball and stick、Space filling、Licorice；成键判断使用完整的共价半径表；原子标签可贴附于三维原子或浮动显示。
- 亮色/暗色主题：工具栏右侧的太阳/月亮菜单可选 System（默认，实时跟随系统外观）、Light、Dark。
- 快速且省内存：利用高斯函数在 x、y、z 上可分离的性质（精确计算），三维轨道场只需一次矩阵乘法
  （约 1000 个基函数时，Grid 64 约 10 ms，Grid 128 约 45 ms）。轨道按需渲染，只预渲染当前轨道的
  前后两个轨道，因此 `A`/`D` 切换几乎即时；只改等值面值时只重跑 marching cubes。
- 可将视图保存为 PNG；可把文件拖入窗口打开。

## 安装

需要 Python 3.10 或更新版本。在项目根目录、进入所用环境（例如 `conda activate wavefunction`）后执行：

```bash
python -m pip install ".[gui]"
```

开发时可改用 `python -m pip install -e ".[gui]"`（修改代码立即生效）。
核心/批处理依赖 `numpy`、`scipy`、`scikit-image`；GUI 另需 `PyQt6`、`PyOpenGL`、`pyqtgraph`。

在 Linux 上 `gui` extra 将 PyQt6 限定为 `>=6.7.1,<6.10`，其 x86-64 wheel 仍兼容 glibc 2.28
（CentOS/RHEL 8）；其它平台使用最新 PyQt6。查看版本：

```bash
python -c 'from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR; print("PyQt", PYQT_VERSION_STR, "Qt", QT_VERSION_STR)'
```

### Linux 说明

PyQt6 wheel 自带 Qt，但 X11 的 `xcb` 插件需要若干系统库。MOview 在启动 Qt 前检查这些库，
缺失时打印库名和对应的 `dnf`/`apt` 命令，而不是让 Qt 直接崩溃。CentOS/RHEL 8 中
`libxcb-cursor.so.0` 来自 EPEL：

```bash
sudo dnf install epel-release
sudo dnf install xcb-util-cursor
```

GUI 需要桌面 OpenGL 2.1 或更高（硬件驱动或 Mesa llvmpipe）。常用环境变量：

```bash
env -u SESSION_MANAGER moview file.fch                         # 屏蔽 ICE 会话警告
env -u SESSION_MANAGER LIBGL_ALWAYS_SOFTWARE=1 moview file.fch # 强制 Mesa 软件渲染
QT_LOGGING_RULES="qt.qpa.gl=true" moview file.fch              # OpenGL 诊断信息
```

批处理模式不会导入 Qt 或 OpenGL。

## 使用

```bash
moview                                   # 打开窗口，然后点击 Open 或拖入文件
moview path/to/wavefunction.fch          # 打开并渲染 HOMO
moview file.molden.input --grid 96 --iso 0.03
moview file.fch --no-auto-render
moview file.fch --config my.ini
```

未安装时，可在项目根目录运行 `python -m moview ...`，或把 `bin/` 加入 `PATH`。

### 鼠标与键盘

| 操作 | 方式 |
| --- | --- |
| 旋转 | 左键拖动、方向键 |
| 绕视线旋转 | 右键拖动 |
| 平移 | Shift + 左键拖动、中键拖动 |
| 缩放 | 滚轮、`+` / `-` |
| 上一个 / 下一个轨道 | `A` / `D`，工具栏 ‹ › |
| 旋转中心 | 按 `C` 后点击原子（`Esc` 取消） |
| 重置视图 | `R` |
| 打开 / 渲染 / 保存图片 | ⌘/Ctrl `O`、⌘/Ctrl `R`、⌘/Ctrl `S` |

在文本框中输入时不会触发这些快捷键。

### 设置面板

- **Isosurface**：等值面值（对数滑块或精确输入，绘制 ±值）、最长边网格点数、盒子边距（bohr）。
  下方显示网格形状、点数与间距。Grid 大于 256 时会先显示内存估计并请求确认。
- **Appearance**：等值面风格、+ψ/−ψ 颜色、原子风格与大小、标签（编号、元素或两者）及其位置和字号。
  外观修改不会重新计算轨道。
- **View**：比较视图同步旋转、坐标轴。

## 批处理模式

```bash
moview file.fch --batch                          # HOMO，默认网格与等值面值
moview file.molden --batch --check               # 同时检查 MO 正交归一性
moview file.fch --batch --spin beta --orbital lumo+1 --grid 64 --iso 0.03
```

`--orbital` 接受从 1 开始的编号或 `homo`、`lumo`、`homo-2`、`lumo+1` 等。
`--check` 输出 `max|<i|i>-1|` 与 `max|<i|j>|`，约 `1e-6` 或更小说明文件被正确读取。

## 配置

复制 `moview.example.ini` 为 `moview.ini` 后修改。按以下顺序使用第一个找到的文件：
`--config PATH`、`$MOVIEW_CONFIG`、`./moview.ini`、
`~/Library/Application Support/MOview/config.ini`（macOS）、
`$XDG_CONFIG_HOME/moview/config.ini`（或 `~/.config/moview/config.ini`）、包目录旁的 `moview.ini`。
命令行参数优先于配置文件。

| 节 | 键 |
| --- | --- |
| `[resources]` | `render_cache_mib`、`render_cache_entries`（等值面缓存）、`grid_chunk_points`（每次矩阵乘法的网格点数）、`surface_face_limit` |
| `[render]` | `grid`、`margin_bohr`、`isovalue`、`surface_style`、`atom_style`、`atom_scale`、`label_mode`、`label_placement`、`label_size`、`positive_color`、`negative_color` |
| `[view]` | `zoom`、`sync_views`、`show_axes`、`auto_render`、`theme`（system、light、dark） |
| `[colors]` | `名称 = R, G, B`，修改或新增颜色预设 |

内存单位为 MiB。未知键和非法值会在启动时报告具体文件与键名。旧版预渲染流程的键
（`basis_workers`、`background_jobs`、`basis_cache_mib`、`max_basis_cache_entry_mib`、
`prefetch_field_budget_mib`、`max_prefetch_orbitals`）仍可被读取但不再起作用，旧的 `moview.ini` 可继续使用。

## 项目结构

| 路径 | 内容 |
| --- | --- |
| `moview/cli.py` | 命令行、批处理、延迟加载 GUI |
| `moview/config.py` | 配置查找与校验 |
| `moview/constants.py`、`moview/elements.py` | 单位与默认值；元素符号、颜色、共价半径 |
| `moview/core/` | 波函数数据模型、基函数约定、轨道计算、网格、等值面、重叠积分、成键 |
| `moview/parsers/` | 格式识别、FCHK 与 Molden 读取 |
| `moview/gui/` | 程序入口、主窗口、渲染服务与缓存、面板、视口、着色器、主题 |
| `moview/gui/platform/` | Linux xcb 预检、macOS stderr 过滤、OpenGL 格式 |
| `tests/` | 单元测试与 GUI 测试 |

## 开发

```bash
python -m unittest discover -s tests -v
MOVIEW_FIXTURES=/path/to/real/files python -m unittest discover -s tests   # 同时测试真实文件
```

GUI 测试以无界面方式运行（`QT_QPA_PLATFORM=offscreen`），Qt 可能打印
"QOpenGLWidget is not supported on this platform"，属正常现象。

## 限制

- 不读取 ORCA `.gbw` 等原生格式，请用 `orca_2mkl name -molden` 转换。
- 极高网格受内存限制；GUI 会先估算开销。
- GUI 需要桌面 OpenGL；无图形界面的 SSH 环境请使用批处理模式。
