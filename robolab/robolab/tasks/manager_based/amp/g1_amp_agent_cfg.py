import os

from isaaclab.utils import configclass

from ..agents.x1_amp_agent_cfg import X1RslRlOnPolicyRunnerAmpCfg


# G1-AMP runner: inherits the X1 AMP PPO/AMP hyperparameters verbatim (29-DoF
# action/obs dims are identical: 96-dim frame + 4-dim gait clock, 50 Hz).
# Only the experiment identity changes (separate logs / checkpoints), matching
# the F1W pattern. X1_* env vars (X1_DISC_BUFFER / X1_STYLE_SCALE /
# X1_TASK_LERP / X1_CONTROL_HZ ...) keep their meaning unchanged.


@configclass
class G1RslRlOnPolicyRunnerAmpCfg(X1RslRlOnPolicyRunnerAmpCfg):
    experiment_name = "g1_amp"
    wandb_project = "g1_amp"
