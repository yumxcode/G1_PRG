# g1_pd_calibration.md — Unitree G1 (29-DoF) 首版 PD / armature 标定方案

状态：**v1 首版（2026-11-18），全部为推导/目录值，零真机辨识**。按
`x1_armature_pd.md`（X1 v68 标定方法论）给出迁移路线与待辨识清单。
sim2sim 评测已可用 v1 表跑（`g1_sim2sim.py` 内置 + `--pd-json` 运行时覆盖）；
任何 sim2real 结论必须等 §4 辨识完成后重开。

---

## 1. 方法论迁移（X1 v68 → G1）

X1 标定四原则（x1_armature_pd.md），G1 对应动作：

| X1 原则 | G1 迁移动作 | 状态 |
|---|---|---|
| armature：实验室辨识优先，日志估计补齐 | 用 unitree_rl_lab 电机目录值（rotor+2 级减速反射） | **未辨识（目录值）** |
| kd 需求超上限时降 kp 保 ζ（kp=(kd/2ζ)²/I） | G1 固件量程未确认；先保 ζ≈0.7-0.8 | 推导 |
| Isaac 200 Hz 显式 PD 稳定性：kd·dt/I<0.8 | **G1 全线越界（见 §2.2），改 native implicitfast** | 已实测 |
| 训练/sim2sim/部署三处同表 | 本表 = G1_CFG 训练侧 verbatim；sim2sim 内置同表 | 已对齐 |

## 2. 首版 v1 表（g1_sim2sim.py 内置 `G1_PD` / `G1_ARMATURE`）

来源 = 训练侧 `g1_robot_cfg_ref.py`（G1_CFG），逐项出处分级：

| 关节组 | kp/kd | effort | armature | kp/kd 出处 | armature 出处 |
|---|---|---|---|---|---|
| hip_pitch (×2) | 100 / 8 | 88 | 0.0102 | unitree_rl_gym [HIGH] | 7520-14 目录 |
| hip_roll (×2) | 100 / 8 | **139** | 0.0251 | 同上 [HIGH] | 7520-22 目录 |
| hip_yaw (×2) | 100 / 8 | 88 | 0.0102 | 同上 [HIGH] | 7520-14 目录 |
| knee (×2) | 150 / 8 | **139** | 0.0251 | 同上 [HIGH] | 7520-22 目录 |
| ankle_pitch (×2) | 40 / 4 | 50 | 0.0036 | 同上 [HIGH]（kd 4 = X1 踝 pitch ζ0.73 惯例） | 5020-16 目录 |
| ankle_roll (×2) | 40 / 2.5 | 50 | 0.0036 | 同上 [HIGH] | 5020-16 目录 |
| waist_yaw | 100 / 8 | 88 | 0.0102 | 电机类比 hip_yaw [MED] | 7520-14 目录 |
| waist_roll/pitch | 40 / 8 | 50 | 0.0036 | 电机类比踝 [MED] | 5020-16 目录 |
| shoulder_pitch/roll (×2) | 100 / 8 | 25 | 0.0036 | IsaacLab locomotion + v2 stand-probe 修复 [MED-HIGH] | 5020-16 目录 |
| shoulder_yaw (×2) | 60 / 8 | 25 | 0.0036 | 同上 [MED-HIGH] | 5020-16 目录 |
| elbow (×2) | 80 / 8 | 25 | 0.0036 | 同上 [MED-HIGH] | 5020-16 目录 |
| wrist_roll (×2) | 20 / 4 | 25 | 0.0036 | IsaacLab locomotion [MED-HIGH] | 5020-16 目录 |
| wrist_pitch/yaw (×2) | 20 / 4 | **5** | 0.00425 | 同上 [MED-HIGH] | W4010-25 目录 |

effort/velocity 上限与 GMR URDF == 官方 `g1_29dof_rev_1_0` 一致
（hip_roll/knee 139 Nm/20 rad/s 为 rev_1_0 值；wrist_pitch/yaw 5 Nm 最弱）。

## 2.2 [实测发现] MuJoCo explicit PD 在 G1 上数值不可行 — 必须 native

2026-11-18 冒烟实测（`work/smoke/`，mujoco 3.2.7，timestep 0.001）：

- X1 判据 `kd·dt_eff/armature < 0.8`（dt_eff = pd-every·timestep = 0.005 s）
  在 G1 目录 armature 下**全线越界**：hip_pitch 3.9、hip_roll 1.6、
  wrist 4.7~5.6（X1 能跑 explicit 是因为其辨识 armature 大：hip_pitch 0.208、
  ankle 0.15——反射惯量比 G1 目录值大一个数量级）。
- 失重悬空、纯关节闭环（消除接触/重力）：explicit+pd-every=5 下腕关节
  发散（|dq| ≈ 40 rad/s 持续振荡，力矩需求 100 Nm 级 vs 限 25）；
  与地面接触后腕甩动把角动量传入躯干，加速倾倒。
- **结论：G1 sim2sim 默认 `--actuator native`**（MuJoCo position servo
  kp/kv/forcerange 同表 + implicitfast 积分器，隐式解 kv 阻尼，无发散条件，
  且语义上更贴 Isaac 的 implicit actuator）。explicit 路径保留（对照实验用），
  使用时需 `--pd-every 1` 并接受小惯量关节（腕/肘）仍临界。
- native 下零动作仍倒（见 §2.3）——那是 RL 增益的物理行为，不是执行器模型问题。

## 2.3 [实测发现] 零动作静态站立需要踝/腰 kp ≥ ~150-200（RL 增益靠策略补偿）

零策略（动作恒 0 = PD 锁 DEFAULT_Q）+ v1 表（踝/腰 kp=40）：
1.4 s 前倾倒。机制（native 模式诊断）：站立姿态 CoM 在踝轴前方 ~2.4 cm
（34.4 kg 系统总质量），踝 pitch 静态背屈误差 = 8.1 Nm/40 ≈ 0.2 rad →
躯干前倾 11° → CoM 前移越过支撑多边形前缘（x=+0.125 m）。腰 pitch
kp=40 同理叠加。提高腰/踝无效的单变量扫描（40→150 全倒）说明需要
**踝+腰+膝联合刚度**：`--pd-json` 伺服刚度表（踝 pitch 200/10、踝 roll
150/8、腰 pitch 200/12、腰 roll 150/10、膝 200/10、hip_pitch 150/10）下
零动作完整站满 4.0 s（位移 0.05 m，|v_xy 误差| 0.027 m/s）——证明模型/
映射/PD/接触链路正确，且给出「静态站立增益带」的下界参考
（真机 G1 官方站立演示用的就是高刚度位置控制；RL 训练低 kp 是刻意留出
策略主动平衡的余地，零动作必倒是预期物理行为，不是链路 bug）。

## 3. 运行时覆盖（验收 4：写入 XML 或运行时参数）

`g1_sim2sim.py --pd-json <file>`，格式 `{"<joint>": [kp, kd, effort]}`，
按关节名覆盖内置表（未知关节名 / 非法三元组直接 assert）。冒烟用过的
两个样例：`work/smoke/pd_servostand.json`（静态站立标定）、
`work/smoke/pd_stand.json`（踝腰加强版）。JSON 结果的
`meta.pd_json`/`meta.pd_table` 记录生效来源。

## 4. 待辨识项清单（按优先级）

1. **armature（全部 29 关节）**：目录值 → 台架/日志辨识。X1 教训：辨识后
   hip_pitch 从 0.01→0.208（20×），直接决定 sim2sim 腿部动力学可信度。
   协议：自由振荡衰减（每关节单独激励）+ 行走日志拟合（X1 md §1）。
2. **kd 兑现率**：真机下发 kd 与实际阻尼的比值（X1 实测 0.9-1.0）；
   G1 电机固件量程上限未确认（X1 的 MIT kd≤8 上限约束在 G1 上未知）。
3. **踝 pitch/roll kd 分裂**（4.0/2.5 沿 X1 ζ 0.73/0.93 惯例）：待 G1
   踝惯量实测后重推。
4. **腰三轴**：kp=40-100 为电机类比 [MED]，最低置信度组；真机一手
   扭矩测试可快速校验。
5. **接触摩擦**：MuJoCo floor friction=0.6（vendor XML）vs Isaac 侧
   URDF 导入默认（无显式 contact 系数时 IsaacLab 默认 0.5/0.5）；
   待 Isaac 侧确认后统一（影响滑移指标绝对值）。
6. **延迟谱**：训练侧 DelayedPDActuatorCfg min_delay=0/max_delay=2（0-2
   步 @50 Hz = 0-40 ms 随机化）；评测注入用 `--action-lag`/`--latency-steps`
   对齐随机化区间（不是单点）。

## 5. 版本记录

- v1（2026-11-18）：G1_CFG verbatim 首版 + native-mode 结论 + 静态站立
  增益带实测（work/smoke/ 存证：悬空发散诊断、增益扫描、servostand 4 s
  站立、6 场景冒烟矩阵）。
