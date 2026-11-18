# g1_sim2sim_package — sim2sim 验证链路 G1 化交付包

四件套（任务 grp-82bda2d73286c659--t1 产出）：

| 文件 | 内容 |
|---|---|
| `g1_sim2sim.py` | A. G1 版 MuJoCo 回放器：300 维 obs（term-major×3，clock=[sinL,sinR,cosL,cosR]）、PRG twin（none/hard/soft，prg_clock.PRGState 同参数重建）、npz/ONNX/zero/random 四策略通道、--pd-json 运行时 PD 覆盖、内置 gait_metrics 语义指标（步频/duty/摆动/滑移/相位-接触同步误差/存活）落 JSON |
| `g1_joint_order.py` | B. MuJoCo actuator ↔ lab 29DoF 映射表（显式双表 + import 时 assert；当前恒等映射，按名字解析，附 URDF/live-model 验证函数与 X1→G1 角色映射） |
| `g1_pd_calibration.md` | C. G1 首版 PD/armature 方案（G1_CFG 推导 + 来源分级 + native-mode 数值稳定性实测结论 + 静态站立增益带 + 待辨识清单） |
| `smoke_report.md` | D. 冒烟结果（6 场景实测矩阵 + 2 套单测 + Isaac 差异清单 D1-D10 + TODO） |

运行依赖（随包分发，均来自输入 tar 未改动）：
- `prg_clock.py` — 训练同源 PRG 状态机（直接复用，未重写）
- `g1_mocap_29dof.xml` — GMR G1 MuJoCo 模型（运行时 strip-mesh 预处理）
- `gait_metrics.py` — X1 指标参考（G1 关节名差异见 smoke_report D10）

测试：`tests/test_g1_twin.py`（PRG twin 6 用例）、`tests/test_g1_metrics.py`（合成步态指标语义验证）。

冒烟证据（work/smoke/ 复件）：s1-s6 JSON、npz log、pd_servostand.json、
random.policy.npz / random.onnx（通道测试策略）、s1 mp4。

快速上手：
```bash
python3 g1_sim2sim.py --ckpt zero --xml g1_mocap_29dof.xml --cmd 0 0 0 \
  --duration 4.5 --prg-mode none --actuator native \
  --pd-json smoke_evidence/pd_servostand.json --json out.json --video out.mp4
```
环境：mujoco>=3.2、numpy、imageio(+ffmpeg)、onnxruntime（可选）。
