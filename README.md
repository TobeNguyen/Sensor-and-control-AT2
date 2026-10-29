# Track A: Visual Waypoint Navigation and Docking for a TurtleBot3

41014 Sensors and Control in Mechatronics Systems, Integrated Project

| Name | Student ID |
|---|---|
| Thien Phu Le | 25661048 |
| Tran Minh Thi Nguyen | 14621635 |
| Nguyen Hoang Tung Dao |  |

---

## 1. Overview

A TurtleBot3 Waffle starts at an unknown pose in a room containing obstacles. It has to find a charging dock marked with an ArUco marker, drive to it without collisions and dock in front of it. The robot is given no map: marker poses are learned at run time and obstacles are seen only through the LiDAR.

The system combines four components, each one assessed separately:

| Component | Implementation |
|---|---|
| Model | Unicycle kinematics, marker measurement model, linearised tracking-error model |
| Sensing | Wheel encoders, 360-beam LiDAR, monocular camera with ArUco detection |
| Estimation | Extended Kalman Filter on the planar pose |
| Control | Discrete-time LQR path tracker, LiDAR avoidance, visual-servoing docking |

The software never uses ground truth for control. The simulator exposes ground truth in a single place, which feeds the evaluation metrics only. The robot software talks to the robot through one interface, so the same pipeline can run in simulation and on the real robot.

---

## 2. Pipeline

```
SURVEY -> EXPLORE (only if the dock was not seen) -> NAVIGATE (AIM) -> DOCK -> DONE
                         ^                                  |            |
                         +--------------- RECOVER <---------+------------+
                                (any phase can end in FAILED)
```

| Phase | Behaviour |
|---|---|
| **SURVEY** | Rotates in small steps at the start pose, stopping between steps. Markers are added to the map only while the robot is stationary, because the image is then sharp and not delayed. |
| **EXPLORE** | Entered if the dock was not seen. Repeats *pick direction, turn, move, stop-and-scan*. Directions are scored by free LiDAR space, distance from places already scanned and the stand-off points of known tags. Ends when the dock pose has been learned from enough observations. |
| **NAVIGATE (AIM)** | A *look, turn, move* loop: stand still and observe the dock, turn towards it, drive a short segment with LQR tracking and LiDAR avoidance, repeat. If the dock is not seen, fall back to the dock approach point, then to each known tag in turn, with a limited (half-turn) scan at every point. |
| **DOCK** | Visual servoing on range, bearing and marker view angle. The EKF estimate bridges short marker losses, and a real detection always overrides the map estimate. |
| **RECOVER** | Started when the position uncertainty exceeds a threshold. Spins to look for a tag, drives to a tag stand-off point, and finally follows the learned map to the dock approach point. |

Hand-over rules worth noting:
- AIM hands over to DOCK when the dock is seen on its axis at close range.
- A docking attempt that takes too long, or fails, returns to AIM (up to three retries).
- DOCKED is declared only after consecutive real detections inside the tolerances, never from the estimate alone.

---

## 3. Repository layout

| File | Role |
|---|---|
| `main.py` | State-machine loop (`run_pipeline`) and command line |
| `config.py` | All parameters of the simulation |
| `robot_model.py` | Motion and measurement models, Jacobians, tracking-error model, controllability and observability, `TagMap`, run result types |
| `estimator.py` | EKF, NEES and chi-square helpers |
| `controller.py` | LQR tracker, command limits, front-obstacle speed limiter |
| `navigation.py` | Waypoint navigator with LiDAR avoidance, AIM controller |
| `docking.py` | Visual-servoing docking, dock re-learning from close views |
| `recovery.py` | Tag recovery, stop-go scan, EXPLORE |
| `sensing.py` | Sensor frames, ArUco detection, LiDAR utilities |
| `calibration.py` | Camera intrinsics and extrinsics, frame-convention checks |
| `calibrate_odometry.py` | Wheel radius and wheel base calibration |
| `safety.py` | Command filter and latched emergency stop |
| `environment.py` | Robot interface shared by simulator and real robot |
| `simulation.py` | MuJoCo scene, sensors with imperfections |
| `visualization.py` | Live monitor, final report, route record |
| `evaluation.py` | Systematic evaluation (criteria A to E) |
| `hardware.py` | Real-robot driver and entry point |

---

## 4. Modelling (criterion A)

### 4.1 Frames and conventions

- **Estimator frame:** origin at the robot start pose. Learned marker poses and all control quantities live in this frame.
- **World frame:** used only to draw results and to compare against ground truth.
- Bearing is positive to the left. The marker view angle is zero when the marker faces the robot head-on. The camera frame is X right, Y down, Z forward. A set of automatic checks (`check_frame_conventions`) verifies that points ahead, to the left and above the camera project to the expected pixels, that the camera-robot transform round-trips, and that a head-on marker gives zero bearing and view angle.

### 4.2 Robot kinematics

State `x = [x, y, theta]`. The wheel encoders give the input `(ds, dtheta)`:

```
ds     = r (dl + dr) / 2
dtheta = r (dr - dl) / b                (r: wheel radius, b: wheel base)

x'     = x + ds cos(theta + dtheta/2)
y'     = y + ds sin(theta + dtheta/2)
theta' = theta + dtheta
```

### 4.3 Marker measurement

For a marker at `(mx, my)` with yaw `psi`:

```
range   = sqrt((mx - x)^2 + (my - y)^2)
bearing = atan2(my - y, mx - x) - theta
view    = psi - pi - theta
```

The camera supplies the three quantities `[range, bearing, view]`. The view angle depends only on the heading, which is what makes the full pose observable from a single marker frame.

### 4.4 Tracking-error model for the LQR

The error is expressed in the robot frame, linearised at the reference speeds and discretised with an Euler step (`tracking_error_model`):

```
e_dot = A e + B u,   A = [[0, w_ref, 0], [-w_ref, 0, v_ref], [0, 0, 0]]
                     B = [[-1, 0], [0, 0], [0, -1]]
```

### 4.5 Controllability and observability

Ranks are computed numerically (`analyse_model`) at four operating points, with the marker near and far (the ranks did not depend on the distance). The observability Gramian over a 100-step (2 s) trajectory adds a measure of *how well* a direction is observed, which a rank cannot give.

| Operating point | Controllability rank | Observability rank (range + bearing) | Observability rank (+ view angle) | Smallest Gramian eigenvalue, without view | Smallest Gramian eigenvalue, with view |
|---|---|---|---|---|---|
| stopped (v = 0, w = 0) | 2 | 2 | 3 | about 0 (|λ| < 1e-9) | 8.56e4 |
| straight (v = 0.2) | 3 | 3 | 3 | about 0 (|λ| < 1e-9) | 8.59e4 |
| turn in place (w = 0.5) | 3 | 2 | 3 | about 0 (|λ| < 1e-9) | 8.56e4 |
| arc (v = 0.15, w = 0.4) | 3 | 3 | 3 | about 0 (|λ| < 1e-9) | 8.58e4 |

What the numbers show:

- **Controllability.** With `v = w = 0` the matrix `A` of the error model is zero and no input acts on the lateral error, so the rank is 2. As soon as the robot translates or rotates, `A` couples the along-track error into the lateral one and the rank is 3. The LQR is designed at a non-zero cruise speed (0.15 m/s), where the pair is fully controllable.
- **Observability from range and bearing.** The rank is 2 when the robot is stopped or turns in place, because the position does not change and one viewpoint gives two independent measurements for three states. It is 3 once the robot translates.
- **Observability with the view angle.** The rank is 3 at every operating point, including standing still. A single marker frame is enough to observe the full pose.
- **Rank versus Gramian.** Without the view angle the smallest Gramian eigenvalue is zero to numerical precision (the small negative values are rounding errors) even where the rank test returns 3. Over 2 s the robot moves at most about 0.4 m, which is too little for the third direction to be measurably constrained. The rank result is structural, whereas the Gramian shows that the heading is only weakly observed without the view angle. With the view angle the smallest eigenvalue is about 8.6e4. This is the quantitative reason for using the view angle as a third camera measurement.

The Jacobians `F`, `G` and `H` agree with central finite differences to better than 1.2e-10. The LQR gain at the design point is `K = [[3.06, 0, 0], [0, 3.12, 1.39]]` (control `u = -K e`). The closed-loop pole magnitudes are 0.939, 0.984 and 0.988, all inside the unit circle. At the 50 Hz control rate they correspond to time constants of about 0.32 s, 1.2 s and 1.7 s.

### 4.6 Assumptions

Unicycle kinematics with no slip inside the controller and estimator, additive Gaussian noise, markers static once learned, planar motion, obstacle avoidance outside the LQR (reactive, LiDAR based). The simulator violates the first assumption on purpose (wheel slip, wheel-size errors), to test the robustness of the design.

---

## 5. Sensing and calibration (criterion B)

### 5.1 Sensors

| Sensor | Characteristics used by the software |
|---|---|
| Wheel encoders | Cumulative wheel angle. Noise modelled per step. |
| LiDAR | 360 beams, 0.12 to 3.5 m. Invalid returns are masked. |
| Camera | 640 x 480, processed at 10 Hz. ArUco `DICT_4X4_50`. |

Marker detection uses the Machine Vision Toolbox. Each marker pose is converted into `(range, bearing, view angle)` in the robot frame, and grazing views (beyond 70 degrees) are rejected. A frame in which the toolbox fails is treated like a dropped frame. Camera measurements are used by the EKF only when the robot is not rotating (rotation rate below 0.05 rad/s for at least 0.3 s), to avoid motion blur and latency errors.

### 5.2 Camera intrinsics

The intrinsic matrix and distortion come from a chessboard calibration (OpenCV). After the first fit, the view with the largest reprojection error is removed repeatedly (while more than seven views remain) until the RMS reprojection error is at most 1 px. A calibration above 1 px is rejected. Images are undistorted before marker detection. In simulation the lens has no distortion, so the distortion terms are fixed to zero.

### 5.3 Camera extrinsics

The camera position (forward and lateral offset) and yaw relative to the robot are estimated from chessboard views at known robot-frame poses:

1. For each placement, the board centre and its normal are measured in the camera frame (PnP, intrinsics from 5.2) and in the robot frame (tape measure on the real robot, known robot pose in simulation).
2. The camera yaw is the circular mean of the difference between the two normal directions.
3. The camera position is the mean of `p_robot - R(yaw) q_camera` over all placements.

The residual spread (position RMS and yaw standard deviation) is reported with the result. In simulation 27 placements are used (three distances, three lateral offsets, three headings). The simulator hides a small camera offset error, so the estimate can be compared with the true value.

### 5.4 Odometry calibration

The wheel radius and wheel base are not trusted from the data sheet. `calibrate_odometry.py` runs two manoeuvres, with an external reference playing the role of a tape measure:

- **Straight run** of 1 m at 0.15 m/s: `r = driven distance / mean wheel rotation`.
- **Two full turns** in place at 0.5 rad/s: `b = r (right - left wheel rotation) / body rotation`.

The estimated values are effective values: they also absorb systematic wheel slip, because the reference measures what the body really did. The robot's belief (not the world) is updated with the calibrated values before the run.

### 5.5 Noise handling

- **Per-measurement gating:** chi-square gate on the normalised innovation (section 6).
- **Latency compensation:** a marker measured `camera_delay` seconds ago is re-expressed from the current pose using the odometry recorded since then.
- **LiDAR:** beam validity masking, and age checks in the safety layer.
- **Map building:** observations are weighted by `1 / range^2`, observations beyond a maximum range are ignored, and a tag needs a minimum number of observations (three) before it enters the map.

### 5.6 Results

**Calibration.**
- *Camera intrinsics.* In the evaluation run (seed 100) three chessboard views had to be dropped (reprojection error 5.01, 3.24 and 2.67 px) before the 1 px limit was met. In the `main.py` run (seed 42) the final RMS error was 0.967 px with a focal length of 407.4 px.
- *Odometry (seed 42).* Estimated wheel radius 0.0321 m and wheel base 0.3004 m, against nominal 0.033 m and 0.287 m. The wheel base is 4.7 % above nominal, more than the hidden ±2 % scale error of the simulator. This is expected: the calibrated value is an effective one and absorbs the rotation slip (up to 5 %).
- *Camera extrinsics.* Lateral offset and yaw were estimated as 0.0002 to −0.0031 m and 0.00 degrees, which is correct because the simulator only perturbs the forward offset. The forward offset was estimated with an error of 7 mm in one run (0.2516 m against a hidden 0.2444 m) and 29 mm in another (seed 100: 0.2641 m against 0.2350 m). A possible reason is that the intrinsic fit of seed 100 was the worst (three views dropped): a focal-length error scales the depth of the board and biases the forward offset. This was not verified.

**Detection.** The dock marker was detected in 54 to 60 of 60 camera frames (90 to 100 %) at every tested range from 0.6 to 2.5 m, before and after calibration. During a complete docking run the dock marker was accepted in only 18 to 22 % of the camera frames (any marker in 65 %), because obstacles and the field of view hide the dock for most of the approach.

**Marker and LiDAR noise.** Range bias and the standard deviation of range, bearing and view angle versus distance, before and after extrinsic calibration, are in `results/evaluation/sensing_marker_noise.csv` and `.png`. The LiDAR bias and noise versus distance are in `lidar_noise.csv`.

**Effect on the closed loop (ablation, seed 100, one run each).**

| Configuration | Position RMSE [m] | NEES | Dock lateral error [m] |
|---|---|---|---|
| full system | 0.020 | 4.1 | 0.037 |
| no extrinsic calibration | 0.025 | 5.2 | 0.047 |
| no camera calibration (intrinsics and extrinsics) | 0.106 | 32.3 | 0.011 |
| no odometry calibration | 0.083 | 98.2 | 0.027 |

Odometry calibration reduces the position RMSE by a factor of 4 and the NEES from 98 to 4. Camera calibration has the largest effect on accuracy. Extrinsic calibration adds a small improvement (0.025 to 0.020 m). The dock lateral error does not follow the same order, because docking uses the real marker measurement directly and not the estimate. With one run per row, the small differences must not be over-interpreted.

---

## 6. Estimation (criterion C)

### 6.1 Filter

Extended Kalman filter on `[x, y, theta]`.

**Prediction.** The mean is propagated through the motion model. The covariance uses `P = F P F^T + G Q_odom G^T`, where `Q_odom` is derived from the encoder noise by error propagation:

```
sigma_ds     = r * sigma_enc / sqrt(2)
sigma_dtheta = r * sigma_enc * sqrt(2) / b
```

**Correction.** Marker measurement `[range, bearing, view angle]` (3 degrees of freedom). The bearing and view angle residuals are wrapped to `(-pi, pi)`. The covariance update uses the Joseph form. Each marker is looked up by its identifier in the learned map, so the dock and the extra tags share the same update.

**Gating.** The normalised innovation squared (NIS) is compared with a chi-square threshold (99 %, 2 degrees of freedom, rescaled for other sizes). Rejected measurements are counted. After ten consecutive rejections the filter assumes that its own estimate is wrong: the covariance is inflated until the measurement is consistent, and the measurement is accepted (lock-out recovery).

**Optional LiDAR update.** Scalar updates of a subset of beams against the known room geometry, gated. It needs the world geometry, so it is disabled in the default system and used only in the evaluation as a comparison.

### 6.2 Initial uncertainty and loss of localisation

The filter starts with a small initial covariance because the robot defines its own frame at the start. The position uncertainty `sqrt(trace(P_xy))` is monitored: above `tags.lost_std` the robot is declared lost and the recovery phase starts. After a failed docking the covariance is inflated deliberately so that the next tag detections are accepted instead of being gated out.

### 6.3 Consistency metrics

- **NEES** `e^T P^-1 e` with ground truth, averaging around 3 for a consistent 3-state filter (evaluation only).
- **Mean NIS per degree of freedom**, averaging around 1 (usable on a real robot, no ground truth needed).
- A chi-square acceptance interval on the average is used for warnings in the final report.

### 6.4 Tuning

Three parameters were swept (seed 100, one run per value). The default setting is `R = diag(0.15², 0.05²)` for range and bearing, a view-angle variance of 0.15², a gate of 9.21 and an odometry covariance scale of 1.

| `R` scale | 0.01 | 0.03 | 0.1 | 0.25 | 0.5 | **1 (default)** | 2 | 4 |
|---|---|---|---|---|---|---|---|---|
| Position RMSE [m] | 0.084 | 0.020 | 0.017 | 0.020 | 0.021 | 0.020 | 0.020 | 0.018 |
| NEES (ideal 3) | 157 | 22.2 | 14.6 | 9.3 | 6.3 | 4.1 | 2.1 | 1.1 |
| NIS per dof (ideal 1) | 1.74 | 0.92 | 1.14 | 0.53 | 0.28 | 0.15 | 0.07 | 0.04 |

| Odometry covariance scale | 0.5 | **1 (default)** | 2 | 5 |
|---|---|---|---|---|
| Position RMSE [m] | 0.022 | 0.020 | 0.019 | 0.016 |
| NEES | 4.35 | 4.08 | 2.88 | 1.99 |
| NIS per dof | 0.15 | 0.15 | 0.15 | 0.13 |

The gate (6, 9.21, 13.8 or none) gave identical results because no measurement was rejected in this run, so this sweep carries no information at this seed.

Reading the sweeps:
- **The two consistency metrics disagree on `R`.** NIS per dof is close to 1 at `R` scales of 0.03 to 0.1, which says that the real measurement noise is much smaller than the assumed one. NEES, which is measured against the true pose, is closest to 3 at scales of 1 to 2. Shrinking `R` therefore makes the filter overconfident (NEES 22 at 0.03) even though it makes the innovations look consistent.
- **Likely reason (not tested).** The learned marker positions have errors of 2 to 9 cm (section 9.4 of the log of `main.py`: tag map errors of 0.019 to 0.090 m). These errors are not in the filter covariance. A small `R` makes the filter trust the biased map more, so the true error grows while the innovations stay small.
- **Position accuracy hardly depends on `R`** between scales of 0.03 and 4 (0.017 to 0.021 m). It degrades only at 0.01 (0.084 m).

Decision: the default `R` is kept, because the NEES against ground truth is the stricter test and the tuned value (0.03) made the filter overconfident. The odometry covariance scale of 2 is the setting that brings the NEES closest to 3 (2.88) without a loss of accuracy. The pipeline default has not been changed to it. It would need to be confirmed with more seeds.


### 6.5 Results

Estimator variants (seed 100, one run each):

| Variant | Docked | Position RMSE [m] | NEES | NIS per dof |
|---|---|---|---|---|
| EKF, camera only (pipeline default) | yes | 0.020 | 3.5 to 4.1 | 0.14 to 0.15 |
| Dead reckoning (no marker update) | **no** (explore failed) | 0.023 | 0.12 | n/a |
| EKF, camera + LiDAR map update | yes | 0.003 | 136 | 0.20 |
| EKF, `R` scale 0.03 | yes | 0.020 | 22.2 | 0.92 |
| UKF | yes | 0.020 | 3.7 | 0.15 |
| UKF, `R` scale 0.03 | yes | 0.020 | 20.9 | 1.18 |

- **Camera is required.** Dead reckoning cannot learn the dock, so it never docks. Its position error is small only because the robot hardly moves (final distance to the dock 1.34 m).
- **LiDAR map update.** It gives the smallest position error (0.003 m) but a NEES of 136, so the filter is overconfident by a large factor, and the heading tracking error is large (0.40 rad RMS). It also needs the room geometry in the world frame, so it is unavailable on a real robot and is not part of the default system.
- **UKF versus EKF.** The two are equivalent in accuracy and consistency, and both become overconfident in the same way when `R` is reduced. The nonlinearity of the measurement model is therefore not the limiting factor here. The measurement noise is small compared with the range, so the linearisation is accurate. The inconsistency comes from the measurement model and the map errors.
- **Ten seeds, partial configuration.** `python main.py --trials 10 --seed 100` docked in 10 of 10 runs (95 % Wilson interval 72 to 100 %). This run used odometry calibration but, because `run_trials` does not pass them, **no camera calibration and no delay compensation**. Results: position RMSE 0.053 ± 0.050 m, NEES 12.4 ± 14.2, NIS per dof 0.57 ± 0.72, dock lateral error 0.013 ± 0.009 m, dock marker accepted in 22.2 ± 7.5 % of the camera frames, 8.1 ± 10.6 gate rejections per run and no tag recovery. The mean run time was 62.5 s (median about 54 s). NEES values above 29 occurred in three of the ten seeds (100, 103 and 105), the same seeds with the largest position errors (0.09 to 0.145 m). With camera calibration a single run gave a NEES of 4.1, so the multi-seed consistency of the complete system is not yet measured.

---

## 7. Control (criterion D)

### 7.1 Path tracking

A discrete-time LQR is computed from the algebraic Riccati equation on the tracking-error model of section 4.4, at the cruise speed. The cost weights are `Q = diag(10, 10, 1)` for the along-track, cross-track and heading errors and `R = diag(1, 1)` for the speed and turn-rate inputs. The controller output is `u = ref + (-K e)`, then limited to the robot velocity limits. The pole magnitudes of the closed loop are checked to be inside the unit circle.

The reference is the closest point on the current segment between waypoints. The feed-forward speed decreases near the end of the segment (slow-down radius 0.4 m, minimum 0.03 m/s).

### 7.2 Obstacle avoidance

The navigator switches between two modes:

- **TRACK:** follow the segment with the LQR.
- **AVOID:** when the free-path test along the goal direction fails (corridor of the body half-width plus a margin, 0.6 m look-ahead), choose the free heading closest to the goal from a 15 degree grid, with hysteresis towards the previous choice. The mode ends only after the goal path has been clear for 0.5 s in a row.

The forward speed is also scaled down as the front LiDAR range decreases. If the robot is stuck in AVOID near the last waypoint for too long, control is handed over to the docking controller.

### 7.3 Docking

Docking uses a separate visual-servoing law on the measured marker: the forward speed follows the range error and is reduced when the marker is off-centre or the robot is off the marker axis, and the turn rate combines the bearing with the marker view angle to converge onto the marker axis. If the robot arrives at the right distance but is off the axis for more than 2 s, it backs off and re-approaches. Docking succeeds when the range, bearing and axis angle are inside the tolerances (target distance 0.55 m +- 0.05 m, bearing 0.08 rad, axis 0.15 rad) for two consecutive real detections.

### 7.4 Results

Path tracking (seed 100, one run each; the error is measured against the reference segment in the estimator frame, so it is how well the robot follows the path it believes in):

| Setting | Cross-track RMS [m] | Cross-track max [m] | Heading error RMS [rad] | Time [s] | Dock lateral error [m] |
|---|---|---|---|---|---|
| Q × 0.1 (identical to R × 10) | 0.055 | 0.078 | 0.130 | 52.3 | 0.028 |
| **default (Q = diag(10, 10, 1), R = I)** | **0.006** | **0.012** | **0.016** | 51.7 | 0.037 |
| Q × 10 (identical to R × 0.1) | 0.002 | 0.003 | 0.004 | 52.3 | 0.051 |
| Proportional (Stanley-style) baseline | 0.016 | 0.023 | 0.063 | 51.4 | 0.034 |
| LQR designed at 0.15 m/s, cruise 0.10 m/s | 0.006 | 0.012 | 0.018 | 54.6 | 0.067 |
| LQR redesigned for 0.10 m/s | 0.003 | 0.005 | 0.026 | 54.8 | 0.065 |
| LQR designed at 0.15 m/s, cruise 0.22 m/s | 0.004 | 0.009 | 0.016 | 50.7 | 0.028 |
| LQR redesigned for 0.22 m/s | 0.005 | 0.006 | 0.020 | 51.0 | 0.029 |

- **Only the ratio of Q to R matters.** The rows for Q × 0.1 and R × 10 are identical, and so are those for Q × 10 and R × 0.1, because the LQR gain depends on the ratio of the weights.
- **Weights.** Raising the ratio by a factor of 100 reduces the cross-track RMS from 0.055 m to 0.002 m and the heading error from 0.130 to 0.004 rad, with no change in the total time. The control effort and saturation of each setting are in `control_runs.csv` (columns `effort` and `sat_frac`) and were not tabulated here.
- **LQR versus the proportional baseline.** The LQR has a cross-track RMS about 2.7 times smaller (0.006 against 0.016 m) and a heading error about 4 times smaller (0.016 against 0.063 rad). Total time and docking error are the same.
- **Redesigning the LQR for the actual cruise speed** has no clear effect at 0.10 or 0.22 m/s.
- **Speed barely changes the total time** (50.7 to 54.8 s), because the robot spends most of the run in the stop-and-go phases (survey, exploration, look-turn-move) and only a small part in continuous path tracking. The docking error differs from run to run (0.028 to 0.067 m) and does not follow a clear order, so with one run per setting no conclusion about it is drawn.

---

## 8. Safety layer

Every velocity command goes through `SafetyGuard` before it reaches the actuators. It is deliberately independent of the planners.

| Rule | Value |
|---|---|
| LiDAR data older than | 0.5 s: stop |
| LiDAR data older than | 2.0 s: **latched E-STOP** (needs an operator reset) |
| Fewer than 30 valid LiDAR beams | LiDAR treated as invalid: stop |
| Translation | The sector of +-60 degrees in the direction of motion (forward or reverse) must be farther than the body radius + 0.05 m |
| Rotation | The whole 360 degree ring must be farther than the body radius + 0.01 m |
| Position uncertainty | Above 0.30 m the forward speed is scaled down, above 1.00 m translation stops (rotation stays allowed) |
| Acceleration | Speeding up is limited (0.5 m/s^2 linear, 3 rad/s^2 angular). Braking is never delayed. |
| Invalid command | A non-finite command stops the robot |

A latched E-STOP ends the run and is reported as a failure, never as success.

---

## 9. Simulation and robustness (criterion E)

### 9.1 What is simulated

The simulator samples a hidden world for every seed. It differs from what the robot believes:

- dock position (along the wall and in depth) and yaw
- true camera field of view and camera mounting offset
- true wheel radius and wheel base
- wheel slip (translation and, more strongly, rotation)

and it adds the imperfections of real hardware:

- actuator acceleration limits and motor dead band
- camera latency (quantised to the camera period), dropped frames, image noise, brightness changes, motion blur while rotating
- LiDAR delay, range-dependent noise and dropped beams

The seed changes the noise and the hidden world. The obstacle layout is fixed in `config.py`.

### 9.2 Experiments

| Experiment | Question |
|---|---|
| Disturbance sweeps | How much encoder noise, LiDAR noise, LiDAR dropout, camera frame loss, camera delay (with and without compensation), dock displacement, wheel slip and all noises combined can the system tolerate? |
| Ablations | What does each feature contribute? Tag recovery, odometry calibration, camera calibration, extrinsic calibration, delay compensation, LiDAR map update. |
| Estimator variants | Dead reckoning, camera only, camera + LiDAR, EKF and UKF (default and tuned `R`), sweeps of `R`, gate and odometry covariance. |
| Controller comparison | LQR weights, a proportional (Stanley-style) baseline on the same error model, and the LQR designed at the default or at the actual cruise speed (0.10 and 0.22 m/s). Cross-track and heading error are logged. |
| Random layouts | Random obstacle positions and random start headings, one layout per seed. |

Every experiment is repeated over several seeds. For each setting the evaluation reports the success rate, the mean and standard deviation of time, position and heading RMSE, NEES, NIS per degree of freedom, docking error, obstacle and wall clearance, control effort and path efficiency, and a count of failure causes.

### 9.3 Results

All numbers are from one seed (seed 100) per setting unless stated. A single run per setting has a 95 % Wilson interval of 21 to 100 % for a success and 0 to 79 % for a failure, so these results show where the system *can* work, not how *reliably* it does. The same configuration (default settings, seed 100) produced identical numbers in four different experiments (docked in 51.7 s, RMSE 0.020 m, NEES 4.08), which shows that the simulation is deterministic and that the runs are reproducible.

Of 71 runs, 68 docked and 3 failed (all *explore failed*). No collision occurred in any docked run (a collision ends the run as a failure).

| Disturbance | Levels | Result |
|---|---|---|
| Encoder noise × | 0.5, 1, 2, 4 | docked up to ×2 (×2 took 95.7 s against 51.7 s); **failed at ×4** |
| LiDAR noise × | 0.5, 1, 2, 4 | docked at all levels (51 to 63 s) |
| LiDAR dropout | 0, 5, 15, 30 % | docked at all levels |
| Camera frame loss | 0, 10, 30, 50 % | docked at all levels; at 50 % it took 92.2 s and the RMSE rose to 0.033 m |
| Camera delay, not compensated | 0, 0.1, 0.2, 0.4 s | docked at all levels; at 0.4 s it took 86.8 s, RMSE 0.056 m |
| Camera delay, compensated | 0, 0.1, 0.2, 0.4 s | docked at all levels; at 0.4 s it took 94.6 s, RMSE 0.062 m |
| Dock displacement × | 0, 1, 2 | docked at all levels |
| Wheel slip × | 0, 1, 2, 3 | docked at all levels (×2 took 93.0 s, ×3 took 52.5 s) |
| All noises combined × | 1, 2, 3 | docked at ×1 and ×2; **failed at ×3** |

- **Delay compensation showed no benefit** in this single run at any level, and at 0.4 s the compensated run was slightly worse (RMSE 0.062 against 0.056 m). The simulated delay is quantised to the 0.1 s camera period, so the effective delay may differ from the value assumed by the compensation. No claim about the benefit of the compensation is made.
- **Run time is bimodal.** Most runs take 51 to 55 s. Several take 87 to 98 s (encoder noise ×2, 50 % frame loss, delay 0.4 s, slip ×2, and two of the ten trial seeds). This suggests one discrete extra event of about 40 s (for example an extra search or a docking retry) rather than a gradual slowdown. The time does not increase monotonically with the disturbance (slip ×3 took 52.5 s, slip ×2 took 93.0 s). This was not verified with the event logs.

Ablation (seed 100, one run each; all docked):

| Configuration | Position RMSE [m] | NEES | NIS per dof | Cross-track RMS [m] | Time [s] |
|---|---|---|---|---|---|
| full system | 0.020 | 4.08 | 0.15 | 0.006 | 51.7 |
| no tag recovery | 0.020 | 4.08 | 0.15 | 0.006 | 51.7 |
| no odometry calibration | 0.083 | 98.2 | 0.31 | 0.011 | 50.2 |
| no camera calibration | 0.106 | 32.3 | 1.91 | 0.039 | 51.8 |
| no extrinsic calibration | 0.025 | 5.24 | 0.15 | 0.006 | 53.0 |
| no delay compensation | 0.020 | 1.79 | 0.15 | 0.004 | 52.4 |
| LiDAR map update on | 0.003 | 151 | 0.22 | 0.006 | 52.2 |

- **Tag recovery is identical to the full system** because recovery was never triggered (see section 10). The ablation therefore does not demonstrate its benefit.
- **Delay compensation** made no visible difference in position error, and the NEES was lower without it (1.79 against 4.08). With one run this is not a significant difference.

### 9.4 Failure analysis

Three runs failed, all with *explore failed*: dead reckoning (estimator variant), encoder noise ×4, and all noises combined ×3 (which includes encoder noise ×3). In each case the run ended at about 133 s, which is the end of the survey (13.2 s) plus the 120 s limit of the exploration, with the robot 1.29 to 1.47 m from the dock.

| Failed run | Position RMSE [m] | NEES | Interpretation |
|---|---|---|---|
| Dead reckoning | 0.023 | 0.12 | the filter was accurate, and honest (reported uncertainty is much larger than the error) |
| Encoder noise ×4 | 0.083 | 2.02 | accurate and consistent |
| Combined noise ×3 | 0.058 | 1.07 | accurate and consistent |

The position errors were small and the filters consistent, so these failures are not caused by a wrong estimate. The mechanism, inferred from the code and these metrics and not yet confirmed with the `last_event` column of `robust_runs.csv`, is the following. The exploration stops and re-scans instead of driving when the *reported* position uncertainty is above 0.10 m, and adds a tag to the map only when it is below 0.08 m. With strong odometry noise (or no marker updates) the reported uncertainty grows faster than the markers can reduce it, so the robot stays in the scanning state and never learns the dock. The cause therefore lies in the planner thresholds and their dependence on the reported covariance, not in the controller or the sensors. The run with encoder noise ×2 still docked but needed 95.7 s, which places the limit of the system between ×2 and ×4 the nominal encoder noise. Possible remedies are to allow the robot to move to a known tag to reduce its uncertainty before scanning, or to make the thresholds depend on the odometry noise.

---

## 10. Limitations

- **Statistics.** Most results come from a single seed per setting. The 95 % intervals are very wide (21 to 100 % for a success), and a success rate of 100 % does not mean certainty. Only one experiment has ten seeds, and it is for a partial configuration (section 6.5).
- **The complete system has no multi-seed result.** The ten-seed run did not include camera calibration and delay compensation. The consistency of the full system (NEES 4.1 in one run) is not confirmed over several seeds.
- **Tag recovery is never exercised in the nominal scenario.** There were no recoveries in ten trials, and the ablation without recovery is identical to the full system. Its benefit is not demonstrated.
- **Delay compensation shows no demonstrated benefit** in the sweeps.
- **Extrinsic calibration error.** The forward camera offset is estimated with an error of 7 to 29 mm in the two evaluation runs, and the cause of the larger one was not verified.
- **Consistency depends on the map.** The learned marker poses have errors of 2 to 9 cm that are not in the filter covariance. They are the likely reason why the NEES is above 3 when `R` is reduced.
- **Explore failures.** The exploration logic depends on the reported covariance and fails under strong odometry noise (section 9.4).
- **Fixed obstacle layout.** The default layout is fixed, so generalisation to other layouts is not evaluated (random layouts are available as a separate experiment).
- The LiDAR map update needs the room geometry, so it is not part of the default system.
- The controller and the estimator assume planar motion and a static scene.
- The camera is used only while the robot is nearly stationary, which limits how fast the robot can move through a scene.
- The safety layer assumes that a LiDAR scan arrives at least every 0.5 s.
- Cross-track error is measured against the reference segment in the estimator frame, so it describes the path the robot believes in, not its true error.
- The simulator and the estimator share some models, so good simulation results do not by themselves show that the system works on a real robot.

---

## 11. Real robot

> **TODO:** write after the hardware tests.

Planned contents:
- Robot, network and rosbridge interface; what differs from the simulation configuration
- Source of the odometry (wheel encoders or dead reckoning from the commands)
- Calibration on the real hardware (intrinsics, extrinsics, odometry)
- Safety procedure and pre-flight check
- Measurement of the final docking pose (no ground truth on the robot)
- Simulation versus real comparison and the sim-to-real gap

---

### 12. Reproducibility and testing

Every figure and table is produced by `evaluation.py` and written to `results/evaluation/` as CSV, JSON and PNG files. Raw runs are saved per experiment, and the summary and figures are rebuilt from them. The seeds used are printed at the start of each run and are fixed by the `--seed0` argument.

The test suite covers the models, estimator, controllers, pipeline and safety rules. One test replaces the simulator's ground truth with garbage and checks that the run does not change, which shows that control does not depend on ground truth.

Main libraries: NumPy 1.26.4, SciPy 1.17.1, MuJoCo (3.x), OpenCV contrib 4.11.0.86, Machine Vision Toolbox 2.4.0, Matplotlib 3.10.9, pytest 8.4.2 (Python 3.12).

---

### 13. Acknowledgements

The pipeline adapts material from the weekly tutorials of the course. Third-party libraries: MuJoCo, OpenCV, the Robotics and Machine Vision Toolboxes, NumPy, SciPy and Matplotlib. The real-robot interface uses the rosbridge client provided by UTS-RI.