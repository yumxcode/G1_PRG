# C1 PRG 自适应步态时钟模块 — 实现报告

`prg_clock.py`：Contact-triggered Phase-Reset Gait clock（PRG）参考实现。
把 v71 固定外生步态时钟（纯时间函数 `φL=(t/T*(v))%1`）升级为带状态的自适应混合时钟：
**足端接触事件相位重置（hard/soft/none 三档）+ 策略频率调制（δf 经 α 课程混合）**。
本模块 isaaclab-free：仅依赖 numpy；torch 可用时同一 API 原生运行于 torch 张量
（CPU/CUDA），无 torch 时自动落到 numpy 同构层——部署 / sim2sim twin / CI 均可直接复用。

## 交付文件

| 文件 | 内容 |
|---|---|
| `output/prg_clock.py` | PRG 模块（常量 + v71 曲线 + `PRGState` 状态机） |
| `output/tests/test_prg_clock.py` | 全套单测（pytest 兼容 + 可独立 `python3` 直跑） |
| `output/report.md` | 本报告 |

---

## 1. API 说明

### 1.1 模块级常量（全部可被同名环境变量覆盖，import 时读取）

| 常量 | 默认 | 含义 |
|---|---|---|
| `PRG_RESET_MODE` | `"hard"` | 相位重置模式 ∈ {"hard","soft","none"}；非法值 import 即报错 |
| `PRG_RESET_GAIN` | `2.0` | β，soft 重置速率 [1/s] |
| `PRG_SYNC_WINDOW` | `0.25` | γ，swing 尾同步窗口（φ∈[1−γ,1)） |
| `PRG_SYNC_HEAD` | `0.05` | stance 头窗口宽度（φ∈[0, 0.05)） |
| `PRG_ALPHA_MAX` | `0.5` | α 课程上限 |
| `PRG_ALPHA_RAMP_S` | `10.0` | α 线性课程时长 [s] |
| `PRG_FMOD_SCALE` | `0.5` | δf 映射为 ±50% f_cmd |
| `PRG_SOFT_CLAMP` | `0.15` | soft 重置单步最大相位校正量 [周期比例] |
| `PRG_FREEZE_ALPHA` | `"0"` | `"1"` 时冻结 α≡0（B0 退化门用） |
| `PRG_BACKEND` | `"auto"` | `"torch"` / `"numpy"` / `"auto"`（有 torch 用 torch） |
| `PRG_T_*` `PRG_D_*` `PRG_H_*` | 见下 | v71 步态曲线常量（截距/斜率/上下限） |

v71 曲线（`t_star(v)`、`duty_star(v)`、`h_star(v)`、`f_cmd_star(v)`，后端中立）：

```
T*(v) = clamp(1.46 − 0.36·v, 0.70, 1.75)   单脚周期 [s]
d*(v) = clip (0.68 − 0.07·v, 0.55, 0.70)   占空比
h*(v) = clip (0.025 + 0.025·v, 0.025, 0.075)  摆动峰值 [m]
```

### 1.2 `PRGState`

```python
st = PRGState(num_envs, device=None, dtype=None,
              reset_mode=None,      # 覆盖 PRG_RESET_MODE
              freeze_alpha=None,    # 覆盖 PRG_FREEZE_ALPHA
              backend=None)         # 覆盖 PRG_BACKEND

st.reset(env_ids=None)   # phi[ids,0]=0, phi[ids,1]=0.5, contact_prev=False, t_ep=0
st.step(dt, v_cmd, contact, delta_f=None)   # 推进一拍，返回 phi
st.clock_obs()           # (N,4) [sin2πφL, cos2πφL, sin2πφR, cos2πφR]
st.targets(v_cmd)        # {"duty":(N,), "h":(N,), "stance":(N,2), "swing":(N,2)}
st.alpha                 # (N,) 当前 α
st.f_eff                 # (N,) 最近一步有效频率 [Hz]
```

状态：`phi (N,2) ∈ [0,1)`、`contact_prev (N,2) bool`、`t_ep (N,)`。
列 0 = 左脚，列 1 = 右脚。输入接受 list / numpy / torch 张量；标量 `v_cmd` 自动广播。

`step` 单步语义（严格按 spec 顺序）：

```
a) f_cmd = 1/T*(v_cmd)
   α    = clamp(t_ep/PRG_ALPHA_RAMP_S, 0, 1) * PRG_ALPHA_MAX        (freeze 时 ≡0)
   f_pol = f_cmd·(1 + PRG_FMOD_SCALE·δf)      (δf=None 时 f_pol=f_cmd)
   f_eff = (1−α)·f_cmd + α·f_pol
b) φ    += f_eff·dt          （两脚同 f_eff）
c) touchdown = contact & ¬contact_prev     （上升沿，逐脚独立）
   hard: φ[touchdown ∧ φ∈同步窗] = 0
   soft: φ[touchdown ∧ φ∈同步窗] += clip(−β·dt·wrap_err(φ), ±PRG_SOFT_CLAMP·dt·β)
d) φ = φ mod 1;  contact_prev = contact;  t_ep += dt
```

`δf` 输入会被 clamp 到 [−1,1]（防策略输出越界，见实现决策 D8）。

## 2. 重置语义图解（文字）

以左脚一个周期为例（d*=0.6 示意）：

```
φ:   0        d*                     1−γ      1.0
     |---stance---|------swing------|---尾窗---|
     ↑ touchdown 锚点                ↕ 同步窗口 [1−γ,1)∪[0,0.05)
```

- **锚定约定（与 v71 一致）**：φ=0 定义为该脚 touchdown 时刻 = stance 起点。
  stance 窗口为 φ∈[0, d*)，swing 为 [d*, 1)。
- **hard 重置**：脚从空中→接触的上升沿，且此刻相位落在同步窗口
  （swing 尾 [1−γ,1) 或 stance 头 [0,0.05)）时，该脚相位立即置 0
  ——"脚落地即重新对表"。另一只脚不受影响（逐脚独立重置，天然保持反相结构）。
- **soft 重置**：同样的触发条件，但不跳变，而是以受限速率把相位拉回锚点：
  每次 touchdown 事件施加 corr = clip(−β·dt·err, ±PRG_SOFT_CLAMP·dt·β)，
  其中 err 是 φ 在单位圆上到 0 的带符号绕回误差（∈[−0.5,0.5)）。
  未触限时每个 touchdown 事件把误差乘 (1−β·dt)（0.96 @dt=0.02）。
- **同步窗口的作用**：窗口外的接触上升沿（如 stance 中段的虚假接触、双支撑
  抖动）不会触发重置，防止相位被拖乱；只有"本该落地"的时间段内的落地才
  被认定为同步事件。
- **episode 复位**：`reset()` 把左脚锚回 0、右脚锚回 0.5（反相）、接触记忆
  清零、课程时间清零。

## 3. sim2sim twin 集成说明

同一模块无 isaaclab 依赖、无 GPU 依赖，部署/回放侧**同参数重建状态机**：

1. **参数一致性**：训练与部署使用相同的 `PRG_*` 环境变量（或构造参数）
   `reset_mode / β / γ / head / α_max / ramp / fmod_scale / soft_clamp`
   以及 v71 曲线常量。建议在导出策略时把这些参数一并 dump 进部署包，
   部署侧按 dump 值显式传参（而非依赖环境变量默认值）。
2. **输入契约**：每步只需要 `dt`（控制周期，与训练一致，如 0.02s）、
   `v_cmd`（命令速度）、`contact (1,2)`（两侧足底接触布尔，来自 MuJoCo
   接触判定，判定阈值与训练侧 sensor 一致）、`delta_f`（策略头输出的
   频率调制通道，tanh 后 ∈[−1,1]；B0 回放时为 None）。
3. **部署循环（MuJoCo rollout）**：

```python
from prg_clock import PRGState
st = PRGState(num_envs=1, reset_mode="hard", backend="numpy")
obs_clock = st.clock_obs()          # 拼入策略观测（与训练同布局）
while not done:
    action, delta_f = policy(obs)   # delta_f: 策略输出的调制头
    mujoco_step(action)
    contact = feet_contact(mj_data)
    st.step(dt, v_cmd, contact, delta_f)
    obs_clock = st.clock_obs()      # 下一步观测
    tg = st.targets(v_cmd)          # 可选：喂 gait_metrics 指标
```

4. **确定性**：float64 默认 dtype 保证双端逐位一致（B0 退化门 atol=1e-6
   的根基）；若部署侧改用 float32，需重跑退化门确认精度。
5. **回放/评测**：回放 npz 轨迹时按记录的 contact 序列重放 `step()`，
   即可离线复算每步相位、clock_obs 与相位-接触同步误差等指标。

## 4. 与 v71 的差异表

| 维度 | v71（外生固定时钟） | C1 PRG（本模块） |
|---|---|---|
| 相位来源 | 纯时间解析式 `φL=(t/T*(v))%1`，无状态 | 带状态积分 `φ += f_eff·dt`，事件可改写 |
| 接触事件 | 不影响时钟 | touchdown（同步窗内）重置该脚相位（hard/soft） |
| 步频 | 恒为 `1/T*(v_cmd)` | `f_eff=(1−α)f_cmd+α·f_cmd(1±50%)`，策略可调制 |
| 策略输入 | 时钟只进 obs（4 维 sin/cos） | 同布局 4 维 obs + 策略输出 δf 通道 |
| 课程 | 无 | α 从 0 线性升至 0.5（10 s），初期即 v71 |
| 退化 | — | `reset_mode="none"+freeze_alpha+δf=None` 时**逐位回到 v71**（B0 门） |
| 占空比/摆高 | d*(v)、h*(v) 曲线 | 完全相同（`targets()` 输出） |
| obs 布局 | [sin2πφL, cos2πφL, sin2πφR, cos2πφR] | 完全相同（`clock_obs()`） |
| 反相结构 | φR=(φL+0.5)%1 解析保证 | 初始锚 0/0.5 + 两脚同频积分保持；重置逐脚独立 |

## 5. 实现决策（spec 之外的自由度）

- **D1 dtype 默认 float64**：B0 退化门要求 atol=1e-6 的长序列逐位一致，
  float32 累积误差在 1e3 步量级可逼近阈值；float64 下余量 >3 个数量级。
  生产训练如需 float32 可显式传 `dtype`，但须重跑退化门。
- **D2 双后端同构层**：`PRG_BACKEND=auto`（有 torch 用 torch，否则 numpy），
  一份代码经 `_clamp/_where/_stack/_logical_not` 四个分发助手兼容两后端；
  numpy 路径即 sim2sim twin 的部署形态。
- **D3 同步窗口同时约束 hard 与 soft**：spec 把 γ 写为"touchdown 触发重置"
  的总门槛，两档共用同一 `in_sync` 掩码；窗口外 touchdown 一律不改相位。
- **D4 stance 头窗宽度命名为 `PRG_SYNC_HEAD=0.05`**：spec 给出数值但未命名，
  做成可 envvar 覆盖的常量。
- **D5 时变 v 的退化对照采用积分形式**：v71 解析式 `φ=(t/T*(v_t))%1` 在 v 时变
  时会整段重算历史（数学上并非时钟的良定义实现）；spec 指定退化门为
  "积分 dt 累加"，故恒 v 用解析式对照（逐位），时变 v 用"每步 `+=dt/T*(v_k)`"
  的 v71 积分形式对照（atol=1e-9）。输入节选不含 robolab `mdp.gait_clock_obs`
  函数体，已请 lead 核对实际实现方式（黑板 fact b000000000016 一并提出
  obs 列序问题：节选注释写 [sinL,sinR,cosL,cosR]，本模块按任务 spec 实现
  [sinL,cosL,sinR,cosR]）。
- **D6 相位比较采用圆上绕回误差**：退化门在相位恰跨整数边界时（积分商
  11.000000000000002 vs 解析商 10.999999999999998），线性 allclose 会误报
  差 1.0；测试用 `((a−b+0.5)%1)−0.5` 的圆上误差判据，sin/cos 的 obs 对照
  仍用线性 allclose（连续量无此问题）。
- **D7 软重置 wrap 误差定义**：err = φ<0.5 ? φ : φ−1（单位圆上到 0 的最短
  带符号路径，∈[−0.5,0.5)），φ≈0.97 对应 err≈−0.03（"落后"于锚点），
  corr 为正向前拉；clamp 上限 `PRG_SOFT_CLAMP·dt·β`=0.3 周期/s。
- **D8 δf 输入 clamp 到 [−1,1]**：策略头经 tanh 输出可能有数值越界，
  入口统一限幅，保证 f_eff 界 `f_cmd·(1±0.5α)` 恒成立。
- **D9 `reset(env_ids)` 同时清零该 env 的 `f_eff/α` 缓存**：日志语义与
  t_ep 一致（episode 起点），避免复位后读到上一 episode 的频率。
- **D10 初始即接触的行为**：reset 后 contact_prev=False，若第一步就接触则
  记 touchdown；φL=0 本就在 stance 头窗内，重置为 0 是幂等无害操作，
  φR=0.5 不在窗口不受影响——无需特判。

## 6. 测试与验证

测试套件（17 个用例，pytest 兼容、可 `python3` 直跑，双后端自动覆盖）：

| 用例 | 覆盖 |
|---|---|
| `test_constants_match_spec` | 全部默认常量 = spec 数值；v71 曲线 vs 独立参考实现 |
| `test_envvar_overrides` | PRG_* envvar 覆盖生效；非法模式 import 报错（子进程验证） |
| `test_v71_gate_constant_speed` | **B0 退化门**：恒 v（4 档覆盖 T* clamp 三段）×1500 步 ×
  δf∈{None,0}，φ/obs/反相/f_eff vs 解析式逐位一致（atol=1e-6） |
| `test_v71_gate_piecewise_speed` | 时变 v vs v71 积分形式（atol=1e-9） |
| `test_hard_reset_anchor_in_window` | swing 尾窗 touchdown → φ 归零；对侧脚不受扰；持续接触不再触发 |
| `test_hard_reset_ignores_out_of_window` | 窗口外 touchdown 不重置 |
| `test_hard_reset_stance_head_window_and_per_foot` | stance 头窗触发；右脚重置/左脚保持（镜像反相） |
| `test_no_touchdown_hard_equals_none` | 无接触边沿时 hard/soft/none 轨迹逐位相同（重置纯事件驱动） |
| `test_soft_reset_pulls_toward_zero` | corr 公式精确性（含 clamp 段与出窗情形）+ 方向 + 误差缩小 |
| `test_soft_reset_monotone_convergence` | 每周期 touchdown 的相位误差单调收缩（整周期 55 步无净漂移构造） |
| `test_fmod_clamp_and_bounds` | δf=±1 落在 f_cmd·(1∓0.5α) 端点；内部值在界内；α=0 时 δf 无效 |
| `test_alpha_curriculum_monotone` | α 单调、=min(t/10,1)·0.5、10 s 封顶 0.5；freeze 恒 0 |
| `test_episode_reset_scoping` | reset([1]) 只锚定 env1，env0/2 逐位不变 |
| `test_reset_all_and_initial_anchor` | 全量复位锚 [0, 0.5] |
| `test_clock_obs_layout` | (N,4) 布局 [sinL,cosL,sinR,cosR]、值域、与独立参考一致 |
| `test_targets` | d*/h* 曲线与 stance/swing 划分（含 clamp 段） |
| `test_repr_and_input_coercion` | list/标量输入、非法 shape 报错、repr |

实跑结果（macOS，Python 3.9.6，numpy 2.0.2，本地无系统 torch）：

```
$ cd output && python3 tests/test_prg_clock.py
... （17 行 PASS）
17/17 tests passed (backends: ['numpy'], torch=False)
$ python3 -c 'import ast; ast.parse(open("prg_clock.py").read()); \
              ast.parse(open("tests/test_prg_clock.py").read()); print("OK")'
OK
```

本地为 torch-free 环境：按任务说明采用 numpy 同构层实跑（API 签名不变）；
torch 后端路径与 numpy 同构，装好 torch 后同一套测试自动双后端执行
（`BACKENDS = ["numpy", "torch"]`）——见下节实际执行记录。

### 实际执行记录（2026-11-17，macOS arm64，Python 3.9.6）

本地系统 python3 为 torch-free（numpy 2.0.2），按任务说明以 numpy 同构层实跑；
另在工作区 venv 安装 CPU torch 2.8.0 后同一套测试自动双后端执行：

```
$ cd output
$ python3 tests/test_prg_clock.py            # 系统python3, numpy-only
17/17 tests passed (backends: ['numpy'], torch=False)          # exit 0

$ ../work/venv/bin/python tests/test_prg_clock.py   # venv, torch 2.8.0
17/17 tests passed (backends: ['numpy', 'torch'], torch=True)  # exit 0
# 每个用例在 numpy 与 torch 两个后端各跑一遍（BACKENDS 自动展开）

$ ../work/venv/bin/python -m pytest tests/test_prg_clock.py -q
17 passed in 10.33s                                            # pytest 兼容

$ python3 -c 'import ast; [ast.parse(open(f).read()) for f in \
    ["prg_clock.py","tests/test_prg_clock.py"]]; print("OK")'
OK
```

torch 后端验证要点：v71 退化门（atol=1e-6）、硬/软重置、fmod 限幅、α 课程、
episode reset、obs 布局全部在 torch.float64 CPU 张量上同样通过。
建议 lead 合入 g1_prg 后在 CI（torch 环境）保留该测试文件作为 B0 回归门。
