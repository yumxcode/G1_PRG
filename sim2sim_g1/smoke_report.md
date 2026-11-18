# smoke_report.md — G1 sim2sim 管线冒烟报告

日期：2026-11-18 · 环境：macOS arm64（本机），venv Python 3.9.6，
mujoco **3.2.7** / numpy 2.0.2 / imageio+ffmpeg / onnxruntime 1.19.2（全部
工作区内 `.venv/`，未污染系统 python）。模型：`g1_mocap_29dof.xml`
（输入 tar 原样，运行时 strip-mesh 预处理，见 §3 差异 D1）。

## 1. 冒烟矩阵（全部实际跑过）

| # | 场景 | 策略 | 执行器/PD | PRG | 结果 |
|---|------|------|-----------|-----|------|
| s1 | 静态站立 4.5 s | zero | native + servostand pd-json | none | **站满 200 步**，fell=false，位移 0.05 m，vxy_err 0.027 m/s，**mp4 78 KB**（`s1_zero_servostand.mp4`） |
| s2 | RL 增益零策略 | zero | native + G1_CFG v1 | none | 1.40 s 前倾倒（预期：RL kp 靠策略平衡，见 g1_pd_calibration.md §2.3），JSON 完整 |
| s3 | 随机策略 | random-mlp | native + v1 | hard | 0.94 s 倒（随机策略预期），**prg_reset_events=2**（TD 重置事件实际触发），f_eff=0.857 Hz，JSON+npz 完整 |
| s4 | 随机策略 | random-mlp | native + v1 | soft | 0.92 s 倒，soft 限幅正常，JSON 完整 |
| s5 | **npz 通道** | `random.policy.npz`（extract_ckpt_npz 格式） | native + v1 | hard | 加载/推理/30 维动作（含 delta_f 尾通道）正常，JSON 完整 |
| s6 | **ONNX 通道** | `random.onnx`（融合归一化器） | native + v1 | none | onnxruntime 推理正常（act_dim=30 自动探测），JSON 完整 |

产物：`work/smoke/s1..s6*.json`、`s1.npz/s3.npz`（gait_metrics 兼容）、
`pd_servostand.json`/`pd_stand.json`（--pd-json 样例）、`s1_zero_servostand.mp4`。

## 2. 纯逻辑单测（无 MuJoCo 依赖，全过）

`output/tests/test_g1_twin.py`（6/6 PASS，numpy-only）：
- `none` 模式 1500 步 == v71 解析式，obs 1e-12 / 圆上相位 1e-9；
- 常速周期闭合（T*(1.0)=1.10 s @ 55 步 × 20 ms）；
- `hard`：sync 窗口内 TD 把 φL 重锚到 0.000（reset 计数=1）；
- `soft`：单步修正 0.024（= 自然推进+限幅 clamp 0.006，无硬跳）；
- delta_f=+1 → f_eff 精确等于 (1-α)f_cmd+α·f_cmd·1.5（1.136 Hz）；
- freeze_alpha+none+δf=0 == v71 积分（3e-15）。

`output/tests/test_g1_metrics.py`（合成 12 s 双足步态，PASS）：
gait_metrics_summary 精确恢复 cycle=1.100 s / duty=0.636（真值 0.65，Schmitt
离散化 -0.014）/ swing=0.400 s / slip=0.050 m/s；相位同步误差在完美同步时
读 0.0000、注入 +0.1 周期偏移后读 0.1000——**相位-接触同步误差指标语义闭环**。

s3/s5（随机策略）rollout 内 gait 指标 n_events=0：存活 <1 s 采不到完整
Schmitt 窗口（TD 发生在 settle 期），属无真策略冒烟的固有限制；真实步态
指标待 B0 ckpt（≥10 s 存活）后自动产出（管线同一代码路径，已被合成单测覆盖）。

## 3. 与 Isaac 侧的已知差异清单（验收 7）

| # | 项 | MuJoCo twin | Isaac (G1AmpEnvCfg) | 影响/处置 |
|---|----|-------------|--------------------|-----------|
| D1 | 几何/碰撞 | **strip-mesh**：输入 tar 无 STL；保留每脚 4 个 r=5 mm 碰撞球（跟 x=-0.05×2、前掌 x=+0.12×2）+ 肩部 4 个 primitive；躯干/腿 mesh 碰撞体被删 | custom-collision URDF（primitive + sole cylinders）+ self-collision ON | 脚-地接触语义对齐（sole 球≈cylinders）；**躯干/腿自碰缺失**——手臂穿躯干等非法姿态不终止；diffusion 大的策略注意 |
| D2 | 重力 | -9.81（XML 默认） | -9.81（默认） | 一致 |
| D3 | 积分器 | native=implicitfast（隐式 kv）；timestep 0.001（XML 默认 0.002 已改） | PhysX TGS implicit actuator PD | native 是最贴近的 MuJoCo 类比（X1 v58 结论迁移）；explicit 模式数值发散（见 D6） |
| D4 | 接触参数 | floor friction 0.6 / condim 3 / priority 1（vendor XML 自带）；solref 默认 | URDF 无显式系数 → IsaacLab 默认（≈0.5） | **待统一（标定 §4.5）**；影响滑移绝对值，不影响相对比较 |
| D5 | 时延 | 默认 0 注入（--latency-steps/--action-lag/--action-lpf 可加） | DelayedPD 0-2 步（0-40 ms）随机化 | 评测时按随机化区间注入，不是单点 |
| D6 | 执行器模型 | native position servo（kp/kv/forcerange=同表）| implicit PD + armature | G1 目录 armature（0.0036-0.0251）使 explicit PD 发散（kd·dt_eff/armature 1.6-5.6 > 0.8），**默认 native**；见 g1_pd_calibration.md §2.2 |
| D7 | 被动动力学 | dof_damping=0 / frictionloss=0（G1 URDF 无 dynamics 标签，两侧一致）+ armature=v1 目录表 | 同 URDF + armature 同表 | 一致（X1 的 damping=1.0 对齐不适用于 G1——已按 G1 URDF 实际置 0） |
| D8 | 关节顺序 | lab↔mj 恒等映射（URDF 声明序==XML actuator 序，g1_joint_order.py assert） | IsaacLab 按 URDF 树序 | **待 B0 ckpt 复核**（若 Isaac 实际顺序不同，只需改 g1_joint_order.py 一处表） |
| D9 | obs | 300 维 term-major×3，clock=[sinL,sinR,cosL,cosR]（黑板确认的 v71 权威列序） | 同（prg_clock.clock_obs 修正后同序） | 一致；PRG twin 用训练同源 prg_clock.py 原文件 |
| D10 | gait 指标 npz | hinge_names=G1 名（knee_joint 等） | gait_metrics.py 硬编码 X1 名（knee_pitch_joint）于 G3/KH 段 | G1 复用 gait_metrics 需按 g1_joint_order.py 的 X1→G1 角色映射改名（g1_sim2sim 内置指标不受影响） |

## 4. 冒烟中的关键判定（为什么这些行为是"对"的）

1. **零策略 + RL 增益必倒（s2）不是链路 bug**：逐步诊断显示 CoM 在踝前
   2.4 cm、踝 kp40 静态背屈 0.2 rad → 前倾越过支撑边界。这是 RL 低 kp
   设计（策略负责平衡）的物理结果；提高单组增益无效、踝+腰+膝联合伺服
   刚度站满 4 s（s1）证明模型/映射/PD/接触链路本身正确。
2. **explicit PD 发散是数值问题、native 修复**：失重悬空（消除接触与
   重力）纯关节闭环实测 explicit pd-every=5 腕 |dq|≈40 rad/s 持续振荡；
   判据表见 g1_pd_calibration.md §2.2。
3. **PRG hard 重置在真 rollout 里触发**（s3 resets=2）：接触事件流从
   MuJoCo ncon 几何判定 → PRGState touchdown 边沿 → φ 重锚，链路通。

## 5. TODO（不阻塞验收，标注清楚）

- [ ] B0 ckpt 到位后：真实策略全速段评测（0.2-1.5 m/s）、D8 顺序复核、
      gait 指标真实值（冒烟仅合成序列验证语义）。
- [ ] gait_metrics.py G1 化改名（X1→G1 角色映射已在 g1_joint_order.py 头部）。
- [ ] 摩擦/接触参数与 Isaac 侧统一（D4，标定 §4.5）。
- [ ] armature 台架辨识（标定 §4.1，影响 sim2real 而非 sim2sim）。
- [ ] mp4 目前 soft 渲染（matplotlib 骨架）；Mac 无 GL 栈时够用，容器内
      可 `--render gl`。X1 的 GL 路径（MjvCamera follow）已保留在代码中。

## 6. 复现命令

```bash
cd <workspace> && .venv/bin/python output/g1_sim2sim.py \
  --ckpt zero --xml output/g1_mocap_29dof.xml --cmd 0 0 0 --duration 4.5 \
  --prg-mode none --actuator native --pd-json work/smoke/pd_servostand.json \
  --json out.json --log out.npz --video out.mp4
.venv/bin/python output/tests/test_g1_twin.py && .venv/bin/python output/tests/test_g1_metrics.py
```
