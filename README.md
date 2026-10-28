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

Both ranks are computed numerically (`analyse_model`) at several operating points: stopped, straight line, rotation in place, arc, with the marker near and far. The design point of the LQR has non-zero speed because the error model is controllable only if the robot is moving (a stopped robot cannot correct a lateral error with the available inputs). Range and bearing alone give a 2-dimensional measurement; adding the view angle makes the pose fully observable from one frame.

The Jacobians `F`, `G` and `H` are checked against central finite differences.

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

> **TODO:** noise of the marker measurement versus range, bias before and after extrinsic calibration, detection rate, calibration residuals. Source: `results/evaluation/sensing_marker_noise.csv`.

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

> **TODO:** document the chosen measurement noise `R`, the gate, and the odometry covariance scale, with their justification (marker noise measured in section 5.6 and the consistency metrics above).

### 6.5 Results

> **TODO:** RMSE, NEES and NIS per degree of freedom, comparison of dead reckoning, camera only and camera + LiDAR, sweeps over `R`, gate and odometry covariance, accuracy versus consistency trade-off. Source: `results/evaluation/estimation_runs.csv`, `estimation_tradeoff.png`.

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

> **TODO:** cross-track and heading error, control effort, saturation, and the effect of the LQR weights. Source: `results/evaluation/control_runs.csv`, `control_gains.png`.

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
| Disturbance sweeps | How much encoder noise, LiDAR noise, LiDAR dropout, camera frame loss, camera delay (with and without compensation), dock displacement and wheel slip can the system tolerate? |
| Ablations | What does each feature contribute? Tag recovery, odometry calibration, camera calibration, extrinsic calibration, delay compensation, LiDAR map update. |
| Estimator variants | Dead reckoning, camera only, camera + LiDAR, and sweeps of the filter parameters. |
| Controller sweep | Effect of the LQR weights on success, docking error, control effort and time. |

Every experiment is repeated over several seeds. For each setting the evaluation reports the success rate, the mean and standard deviation of time, position and heading RMSE, NEES, NIS per degree of freedom, docking error, obstacle and wall clearance, control effort and path efficiency, and a count of failure causes.

### 9.3 Results

> **TODO:** fill in from the multi-seed run (`results/evaluation/summary.md`, `robustness_success.png`, `ablation.png`). Do not use single-seed numbers.

### 9.4 Failure analysis

> **TODO:** list each observed failure with its cause traced to the model, sensing, estimator or controller. Example already seen in early runs: *explore failed* at high encoder noise, caused by the rule that tags are learned only while the position uncertainty is small, which high odometry noise never allows.

---

## 10. Limitations

> **TODO:** complete after the final evaluation. Known points:
> - The obstacle layout is fixed, so generalisation to other layouts is not evaluated.
> - The LiDAR map update needs the room geometry and is not part of the default system.
> - The controller and estimator assume planar motion and a static scene.
> - The camera measurement is used only while the robot is nearly stationary, which limits how fast the robot can move through a scene.
> - The safety layer assumes that a LiDAR scan arrives at least every 0.5 s.
> - Simulation results do not by themselves show that the system works on a real robot.

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

## 12. Reproducibility and testing

Every figure and table of the report is produced by `evaluation.py` and written to `results/evaluation/` as CSV, JSON and PNG files, together with the list of seeds used. The test suite covers the models, the estimator, the controllers, the pipeline and the safety rules, including a test that replaces the simulator's ground truth with garbage and checks that the run does not change (control does not depend on ground truth).

> **TODO:** final test count, seeds used for the reported figures, versions of the main libraries.

---

## 13. Acknowledgements

TODO (course material adapted from the weekly tutorials, third-party libraries, licence).