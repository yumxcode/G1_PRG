# G1 机器人配置包 — 交付报告

任务：grp-9f4c7a833b80bdae--t1（G1 机器人资产与环境配置，robolab 薄继承）
日期：2026-11-17 · 状态：完成，静态验证全绿（见下）

## 1. 交付文件清单与仓库落位

| 交付文件（output/） | 合入 g1_prg 仓库的目标路径 | 说明 |
|---|---|---|
| `g1.py` | `roboparty_train/robolab/robolab/assets/robots/g1.py` | G1_CFG ArticulationCfg（29 DoF，rev_1_0） |
| `g1_amp_env_cfg.py` | `roboparty_train/robolab/robolab/tasks/manager_based/amp/g1_amp_env_cfg.py` | 薄继承 X1AmpEnvCfg + 名称重定向 |
| `g1_amp_agent_cfg.py` | `.../tasks/manager_based/amp/agents/g1_amp_agent_cfg.py` | 仿 f1w：仅改 experiment 身份 |
| `__init__.py` | `.../tasks/manager_based/amp/__init__.py` | 原 8 个注册 + G1-AMP / G1-AMP-Play（完整文件） |
| `tests/test_g1_static.py` | 仓库 `tests/`（或任意目录） | 纯 xml+ast，无 isaaclab |
| `urdf_diff_report.md` | 文档（建议 docs/ 或任务记录） | 两版 URDF 逐关节对比 + 基底选择 |
| `report.md` | 本文件 | |

另需由 lead 放置的资产（不在本包内，来源 GMR vendored unitree_g1）：

- URDF → `data/robots/g1/urdf/g1_custom_collision_29dof.urdf`（g1.py 引用此路径，仿 x1 的 `robots/x1/urdf/f1.urdf`）
- mesh → `data/robots/g1/urdf/meshes/*.STL`（URDF 内引用 `meshes/...` 相对 URDF 目录，上游 g1_description 原生布局，零 URDF 改动；`assets_robots_init.py` 需加一行 `from .g1 import *`）

## 2. URDF 基底选择（详见 urdf_diff_report.md）

**选 GMR `g1_custom_collision_29dof.urdf`**（robot name `g1_29dof_rev_1_0`）：

- 两版关节名集合与顺序**完全一致**（29 revolute）；位置限位完全一致；
- **唯一限位差异** hip_roll：GMR 139 Nm/20 rad/s vs UMR 88/32。GMR 正确 —— 官方
  unitree_ros `g1_29dof_rev_1_0.urdf` 在 rev 1.0 把 hip_roll 换到 7520-22 电机，四个独立来源
  确认（官方 URDF、IsaacLab issue #8530 全文、mujoco-menagerie、mjlab），UMR 是 rev_1.0 之前的旧值
  （少 37% 力矩、多给 60% 速度上限——训练限位与 sim2sim 导出都会错）；
- 碰撞：GMR 全原始体（胶囊化后 13 个凸体，双脚各 2 个 r=0.02/长 0.15 的滚动式足底）vs
  UMR 全 mesh 碰撞 + 24 个装饰微球 → GMR 对 4096 envs RL 友好（cook/接触对便宜、支撑多边形明确）；
- 惯量：GMR 为 rev_1_0 实测（torso 6.78 kg vs UMR 8.562 kg；pelvis 3.813 kg 与官方 blob 逐字节一致）；
  运动学差异仅躯干链 3 处 z 偏移（GMR 躯干高 1 cm）。

已知取舍：GMR 的 pelvis / hip_pitch / hip_roll / ankle_pitch / wrist 无碰撞体（平地速度任务可接受，
摔倒恢复类任务前需补）；UMR 的传感器帧（imu/d435/mid360）GMR 没有，GMR 的 IMU 帧为
`imu_in_torso`、足尖帧 `left/right_toe_link`（GMR 独有，便于足端统计）。

## 3. kp/kd 数值表（g1.py，全部来源与置信度注于文件头）

| 关节组 | 关节 | kp (Nm/rad) | kd (Nms/rad) | effort/vel (URDF rev_1_0) | armature | 来源 | 置信度 |
|---|---|---|---|---|---|---|---|
| waist_legs | waist_yaw | 100 | 2 | 88/32 | 0.0102 | kp/kd 按电机型号类比（同 hip_yaw 电机）；armature unitree_rl_lab 7520-14 | 中 |
| waist_legs | waist_roll/pitch | 40 | 2 | 50/37 | 0.0036 | 同上（同 ankle 电机 5020-16） | 中 |
| waist_legs | hip_pitch / hip_yaw | 100 | 2 | 88/32 | 0.0102 | unitree_rl_gym g1_config.py（官方，本次逐字核对）；armature 7520-14 | 高 / 中 |
| waist_legs | hip_roll | 100 | 2 | **139/20** | 0.0251 | kp/kd 同 hip 家族；限位=官方 rev_1.0；armature 7520-22 | 高 / 中 |
| waist_legs | knee | 150 | 4 | 139/20 | 0.0251 | unitree_rl_gym；armature 7520-22 | 高 / 中 |
| ankles | ankle_pitch / roll | 40 | 2 | 50/37 | 0.0036 | unitree_rl_gym（与 IsaacLab ankle 40/2 逐字一致）；5020-16 | 高 / 中 |
| arms | shoulder_pitch/roll/yaw, elbow | 40 | 8 | 25/37 | 0.0036 | IsaacLab G1 29dof locomotion 型谱（IsaacLab 同文件 shoulder_yaw/elbow/wrist 40/8、20/4 逐字核对；刻意不用 manipulation 变体的 100/15 硬臂） | 中高 |
| arms | wrist_roll | 20 | 4 | 25/37 | 0.0036 | 同上 | 中高 |
| wrists | wrist_pitch / yaw | 20 | 4 | 5/22 | 0.00425 | 同上；W4010-25 | 中高 |

所有组 `min_delay=0, max_delay=2`（沿用 X1 延迟模型）；`soft_joint_pos_limit_factor=0.90`、
solver/rigid props 与 x1.py 逐字段一致。

**web 取证记录（2026-11-17）**：unitree_rl_gym `legged_gym/envs/g1/g1_config.py`（整文件 base64 解码）；
unitree_ros `motor_info.md`、`README.md`、目录与 rev_1_0 blob；IsaacLab issue #8530（API JSON 全文）；
IsaacLab `isaaclab_assets/robots/unitree.py` main@a8544937（尾部逐字）；unitree_rl_lab
`unitree.py` + `unitree_actuators.py`（电机常数）；mjlab `g1_constants.py`（HOME 关键帧
z=0.783675）。raw.githubusercontent / jsdelivr / statically 在本沙箱不可达，全部改走 api.github.com。

## 4. initial_state 推导（g1.py 文件头 + 测试内可复算）

- 位形：hip_pitch −0.15 / knee +0.30 / ankle_pitch −0.15（足底恰好水平），elbow +0.30（臂自然下垂微屈），
  其余 0 —— G1 官方 demo 站姿族；
- URDF FK（GMR 文件，测试 `t_init_state_and_fk` 复算）：左腿链 ankle_roll 原点在 pelvis 系
  z = −0.7497；足底圆柱局部 z=−0.02、r=0.02 → 最低点 −0.7897 → **站高 0.7897 m**；
- 取 `pos=(0, 0, 0.79)`（FK 精确触地 +0.3 mm）；交叉验证：IsaacLab G1_29DOF init z=0.794（同位形族）、
  unitree_rl_gym 0.80（−0.1/0.3/−0.2 位形+落体裕量）、mjlab HOME 0.783675。

## 5. 环境薄继承要点（g1_amp_env_cfg.py）

- 仿 F1WAmpEnvCfg 手法：`super().__post_init__()` 后**只**覆盖差异——robot=G1_CFG、
  motion 目录/权重（G1 数据集占位 `g1_lab_v1`，空权重使 AMP 采样惰性，C1 零 AMASS 依赖）、
  命令带 lin_vel_x **0.2–1.5 m/s**（`G1_CMD_LO/HI` 可调）、lin_vel_y ±0.2、ang_vel_z ±0.5；
  PLAY 变体 `G1_PLAY_VEL`（默认 0.8）；
- **名称重定向机制**：X1→G1 的 body/joint 名差异（base_link→pelvis、lumbar→waist、
  knee_pitch→knee、elbow_pitch/yaw→elbow/wrist_roll、wrist_roll→wrist_yaw）通过一个
  幂等 walker 在 obs/rewards/events/terminations 全量改写（含 SceneEntityCfg 的
  body_names/joint_names(_expr)，str 与 list 两种形态都处理），不复制 X1AmpEnvCfg 任何
  env-var 分支内部实现，`X1_*` 环境变量语义原样保留（运行时打印 `[G1-RETARGET]` 变更清单，
  便于 gradmotion 日志做改动生效性审计）；
- 手臂映射按**链位角色**对应（X1 elbow_yaw=前臂旋转 → G1 wrist_roll；X1 wrist_roll=手部偏摆 →
  G1 wrist_yaw），并显式处理了同名异义：两版都有 wrist_roll_link 但角色不同，按角色映射；
- disc key bodies：10 个角色等价 G1 体（膝/踝/肘/肩pitch/腕pitch），`G1_KEY_BODY_NAMES`
  模块常量，供未来 G1 重定向 schema 对齐。

## 6. 静态验证（实测输出）

```
$ python3 tests/test_g1_static.py          # 系统 python3，无 isaaclab
[PASS] ast.parse all produced .py — 5 files
[PASS] URDF has 29 revolute joints — robot=g1_29dof_rev_1_0
[PASS] foot/pelvis bodies exist in URDF — feet + pelvis present
[PASS] g1.py joint regex covers URDF joints exactly — 29/29 joints, 4 groups, no overlap
[PASS] g1.py effort/velocity limits == URDF limits — 58 per-joint limits (incl. hip_roll 139/20)
[PASS] init_state joints + FK standing height — FK 0.7897 m vs cfg z 0.79
[PASS] env cfg retarget maps resolve to G1 names — 12 body maps, 7 joint rules, 10 key bodies
[SKIP] URDF mesh files exist (conditional) — 34/34 未随包提供；约定 data/robots/g1/urdf/meshes/
8/8 passed, 1 skipped, 0 failed
```

（SKIPP 项按任务要求条件跳过并注明：GMR 版碰撞为原始体，mesh 仅为视觉，不影响物理；
合入仓库、放入 meshes 后该项自动转 PASS。测试通过 `G1_URDF`/`G1_CFG`/`G1_ENV_CFG`
环境变量或目录约定自动寻位。）

开发期本地验证另含：retarget walker 桩测试（str 写回、幂等、角色映射）与两版 URDF
限位/惯量/运动学全量 diff 脚本（本文数字全部来自实测）。

## 7. 遗留风险与建议

1. **运行时未验证**（本机无 isaaclab）：`X1AmpEnvCfg` 全文不在输入包中（仅节选），obs/rewards 中
   可能存在节选外的 body/joint 引用——重定向 walker 是全覆盖式的，字段路径未知也能改到；但
   **lead 合入后请跑一次 G1-AMP 冒烟**（gradmotion 镜像 BJX00000178），看日志里的
   `[G1-RETARGET]` 行确认改动生效面，并确认 `motion_data.motion_dataset.key_body_names`
   属性名与实际实现一致（env cfg 侧做了 hasattr 防御）。
2. **waist kp/kd 为电机型号类比值**（无公开 G1 运控腰部增益）：首轮训练若腰部晃动，先降
   waist_yaw kp 100→60 再说；armature 为目录值非辨识值——X1 经验（x1_armature_pd.md）表明
   走真机前需日志辨识，sim2sim 不受影响。
3. **X1 奖励项中的 X1 运动学常数**（如 base_height 目标 ~0.75、feet_air_time 时窗等，若存在于
   完整 x1_amp_env_cfg.py）未被数值缩放：G1 站高 0.79 vs X1 0.75，合入时建议 grep
   `target_height|height_target|0.75|0.72` 复核；必要时由 lead 在 G1 覆盖层加显式覆盖。
4. **GMR URDF 部分连杆无碰撞体**（pelvis/hip_pitch/hip_roll/ankle_pitch/wrist）：
   平地速度谱任务可接受；涉摔/自碰撞任务前需补凸包。
5. **数据集占位**：`data/motions/g1_lab_v1` 尚不存在，AMP demo 采样为空权重惰性——
   C1（纯任务先验）设计使然，非遗漏。
