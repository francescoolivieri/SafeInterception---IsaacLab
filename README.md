<div align="center">

# SafeInterception

**Safe target interception for a quadrotor in cluttered arenas, built on Isaac Lab**

Francesco Olivieri

![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-2.3.2-76b900)
![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.1.0-76b900)
![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-ee4c2c)

[Weights](data/checkpoints) · [Evaluation suites](data/benchmark_suite)

</div>

A quadrotor has to catch a moving target in an arena full of static and moving pillars,
without ever touching them. This repository is the code behind my thesis on that problem.
It contains the task, an offline scenario generator, five ways of keeping the pursuer safe,
the fixed evaluation suites and the trained weights.

The five methods go from a classical controller with a safety filter to safety learned
end to end:

| | Policy | How safety is handled |
|---|---|---|
| **M1** | LQR pursuit | CBF-QP filter on the velocity command |
| **M2** | PPO, obstacle-blind | same CBF-QP filter |
| **M3** | PPO, obstacle-blind | learned policy neural CBF (PNCBF) as filter |
| **M4** | graph PPO | penalty for every violation in the reward |
| **M5** | DG-PPO, graph | constraint critic and discrete CBF condition in the policy update |

All policies output a world-frame velocity and yaw-rate command. Observations go through a
measurement model with delays, biases and noise, and every command reaches the vehicle 10 ms late.

## The vehicle model is not included

The quadrotor itself (asset, motor model, low-level velocity controller and its physical
randomisation) is not part of this release. Without it the policies can be neither trained nor
run in simulation, including the shipped weights. What works out of the box is the scenario
side: generating the training pools and the evaluation suites, which runs offline on the CPU.

Everything up to the vehicle is included: `PerturbedBaseEnv` scales the action, delays it and
runs the measurement model, then passes the command to `cfg.robot`,
`_apply_vehicle_command(command)` and `_reset_vehicle(env_ids)`. A vehicle model plugs in by
subclassing `TargetInterceptionEnv` and implementing these three; the command arrives at
500 Hz as a world-frame `[vx, vy, vz, yaw_rate]`. Without them, training and evaluation stop
at scene setup and say which hooks are missing.

## Installation

The code was developed on Ubuntu 22.04 with these versions:

| Isaac Sim | Isaac Lab | Python | PyTorch | skrl | PyTorch Geometric |
|---|---|---|---|---|---|
| 5.1.0 | 2.3.2 | 3.11 | 2.7.0 (CUDA 12.8) | 2.1.0 | 2.7.0 |

Install Isaac Sim and Isaac Lab first, following the
[Isaac Lab 2.3.2 installation guide](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/index.html).
It lists the hardware and driver requirements (an RTX GPU, Ubuntu 22.04 or Windows 11) and the
different ways to install (pip, binaries, Docker).

<details>
<summary>The pip setup used here</summary>

```bash
conda create -n safe_interception python=3.11
conda activate safe_interception
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

git clone --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && ./isaaclab.sh --install && cd ..
```

</details>

Then install this repository in the same environment. This also pins skrl to 2.1.0:

```bash
git clone https://github.com/francescoolivieri/SafeInterception.git
cd SafeInterception
pip install -e .
```

## Evaluation

All methods are compared on fixed suites of trials rather than on random episodes. A trial
fixes the obstacle layout, the start of the pursuer and the route and speed of the target, so
every method faces exactly the same situations. It ends when the target is caught, at the first
collision or arena violation, or after 10 s. Two suites ship in `data/benchmark_suite/`:

| Suite | Arena | Trials |
|---|---|---|
| `thesis_eval_large_final_id` (in distribution) | the large training arena | 3000: three clutter levels (3/2, 6/4 and 10/5 static/moving pillars) with 200 layouts and four target speeds from 1.2 to 2.2 m/s, plus 600 with a stationary target |
| `thesis_eval_room_final_ood` (out of distribution) | a smaller room with walls | 600: a wall, a narrow passage and moving obstacles, 200 each |

### Running an evaluation

Every method has a config in `configs/` that names the task, the seed-0 weights and the safety
filter with its thesis settings. A run plays the whole suite once, with the deterministic
action and evaluation seed 0:

```bash
# M5 on the in-distribution suite
python scripts/evaluate.py --experiment configs/m5_dgppo.yaml --headless

# M5 on the out-of-distribution suite
python scripts/evaluate.py --experiment configs/m5_dgppo.yaml --ood-room --headless
```

Results land in `outputs/`: `metrics.json` has the capture, collision and timing metrics
(overall and per scenario family) and `episodes.pt` the trajectories.

The thesis reports two seeds per method, and both are in `data/checkpoints/`: 0 and 42. `--checkpoint` (and `--filter-checkpoint` for the M3
certificate) evaluates the second seed with the same settings. M1 has nothing to train, so its
two seeds are evaluation seeds, set with `--evaluation-seed 0` and `--evaluation-seed 42`:

```bash
python scripts/evaluate.py --experiment configs/m5_dgppo.yaml --headless \
    --checkpoint data/checkpoints/m5_dgppo_s42/checkpoints/agent_800000.pt
```

To compare all methods, `benchmark.py` runs every config in both arenas and writes a markdown
table, a CSV and two plots. With existing results it only rebuilds the tables:

```bash
python scripts/benchmark.py --headless
python scripts/benchmark.py --aggregate-only --output_dir outputs/benchmark
```

### Building new suites

The suites are generated offline on the CPU, without the simulator. The shipped ones are the
suites used in the thesis, and generating with the default name overwrites them, so give a
new suite its own name and select it with `--suite-name` when evaluating:

```bash
python scripts/generate_benchmark_suite.py --task TargetInterception-State-Large-v0 --suite-name my_large_suite
python scripts/generate_benchmark_suite.py --task TargetInterception-State-v0 --suite-name my_room_suite

python scripts/evaluate.py --experiment configs/m5_dgppo.yaml --suite-name my_large_suite --headless
```

The task picks the arena. `--clutter-layouts`, `--challenge-layouts` and `--seed` change the
size and the draw of the suite.

## Training

Training does not sample scenarios on the fly. Layouts and target routes come from pools
generated offline on the CPU and grouped by curriculum phase, and as training goes on the
environment moves from the easy phases to the hard ones. So create the pools first; they are
written to `data/scenario_pools/`, and their default size per phase is the one used in the thesis:

```bash
python scripts/generate_scenarios.py --task TargetInterception-Nominal-Large-v0 --pool-name clutter_nominal_pool_large
python scripts/generate_scenarios.py --task TargetInterception-State-Large-v0 --pool-name clutter_train_pool_large
python scripts/generate_scenarios.py --task TargetInterception-State-Penalty-Large-v0 \
    --curriculum-profile penalty_staged --pool-name penalty_large_staged
```

The agent configs in `target_interception/task/agents/` hold the full recipe of every method:
training budget, reward weights, curriculum and network. A run only needs the task, the
algorithm, the number of environments and the seed (the thesis used 4096 environments and
seeds 0 and 42):

```bash
# nominal PPO, the policy behind M2 and M3
python scripts/train.py --task TargetInterception-Nominal-Large-v0 --algorithm PPO --num_envs 4096 --seed 0 --headless

# M3: certificate of a trained nominal policy
python scripts/train.py --task TargetInterception-State-Large-v0 --algorithm PNCBF --num_envs 4096 --seed 0 --headless \
    --nominal-checkpoint data/checkpoints/nominal_ppo_s0/checkpoints/agent_150000.pt

# M4
python scripts/train.py --task TargetInterception-State-Penalty-Large-v0 --algorithm PPO_GRAPH --num_envs 4096 --seed 0 --headless

# M5
python scripts/train.py --task TargetInterception-State-Large-v0 --algorithm DGPPO --num_envs 4096 --seed 0 --headless
```

The shipped M3 certificates come from four fits of policy iteration: each fit after the first
collects data with the nominal policy filtered by the previous certificate, passed with
`--policy-filter-checkpoint`. For several GPUs, launch with `torchrun` and add `--distributed`.

Single values of the agent config can be changed from the command line with Hydra, for
example `agent.trainer.timesteps=1000` for a short test run. The PPO-based recipes log to
Weights & Biases; `agent.agent.experiment.wandb=false` turns that off. Every run is
written to `logs/projects/<task>/<suite>/<timestamp>`, with its resolved configuration in
`params/` next to the checkpoints.

| Task | Arena | Used for |
|---|---|---|
| `TargetInterception-Nominal-Large-v0` | large, no obstacles | nominal PPO (M2, M3) |
| `TargetInterception-State-Large-v0` | large, up to 11 static and 5 moving pillars | M3, M5 and evaluation |
| `TargetInterception-State-Penalty-Large-v0` | same, with the safety penalty in the reward | M4 |
| `TargetInterception-State-v0` | room | out-of-distribution evaluation |

## Repository structure

```
SafeInterception/
├── target_interception/
│   ├── sim/          base environment: action interface, command delay, measurement model
│   ├── task/         interception environment, observations, rewards, curriculum
│   │   └── agents/   training recipe of every method
│   ├── scenarios/    scenario generation, training pools, evaluation suites
│   ├── algorithms/   LQR (M1), PNCBF (M3), graph PPO (M4), DG-PPO (M5)
│   ├── safety/       CBF-QP and PNCBF filters, safety constraints
│   ├── nn/           relation graph, GNN and network presets
│   ├── policies.py   checkpoint loading for evaluation
│   └── metrics.py    evaluation metrics
├── scripts/          training, evaluation, benchmark, pool and suite generation
├── configs/          evaluation config of every method
└── data/
    ├── checkpoints/      trained weights of both seeds, each with its training configuration
    └── benchmark_suite/  the in-distribution and out-of-distribution suites
```
