| experiment | setting | level | n | success | time [s] | RMSE [m] | NEES (~3) | NIS/dof (~1) | dock lat [m] | failures |
|---|---|---|---|---|---|---|---|---|---|---|
| ablation | full system | - | 1 | 100% | 92.7 | 0.022 | 4.93 | 0.06 | 0.012 | {} |
| ablation | no recovery | - | 1 | 100% | 92.7 | 0.022 | 4.93 | 0.06 | 0.012 | {} |
| ablation | no odometry calibration | - | 1 | 100% | 79.7 | 0.044 | 35.62 | 0.25 | 0.011 | {} |
| ablation | no camera calibration | - | 1 | 100% | 94.1 | 0.016 | 1.85 | 0.25 | 0.026 | {} |
| ablation | no extrinsics calibration | - | 1 | 100% | 93.1 | 0.013 | 3.03 | 0.18 | 0.027 | {} |
| ablation | no delay compensation | - | 1 | 100% | 92.8 | 0.042 | 8.91 | 0.07 | 0.008 | {} |
| ablation | LiDAR map update on | - | 1 | 100% | 93.6 | 0.003 | 116.35 | 0.15 | 0.011 | {} |
| control | Q x0.1 | - | 1 | 100% | 93.0 | 0.011 | 0.56 | 0.16 | 0.009 | {} |
| control | Q x1 (default) | - | 1 | 100% | 92.7 | 0.022 | 4.93 | 0.06 | 0.012 | {} |
| control | Q x10 | - | 1 | 100% | 85.9 | 0.013 | 0.82 | 0.16 | 0.002 | {} |
| control | R x10 | - | 1 | 100% | 93.0 | 0.011 | 0.56 | 0.16 | 0.009 | {} |
| control | R x0.1 | - | 1 | 100% | 85.9 | 0.013 | 0.82 | 0.16 | 0.002 | {} |
| estimation | pipeline default | - | 1 | 100% | 55.4 | 0.015 | 1.81 | 0.16 | 0.022 | {} |
| estimation | dead reckoning | - | 1 | 0% | 133.0 | 0.022 | 0.17 | nan | 0.977 | {"explore failed": 1} |
| estimation | camera only | - | 1 | 100% | 55.2 | 0.015 | 1.58 | 0.16 | 0.025 | {} |
| estimation | camera + LiDAR | - | 1 | 100% | 54.8 | 0.003 | 153.24 | 0.17 | 0.027 | {} |
| estimation | sweep R x | 0.25 | 1 | 100% | 55.4 | 0.013 | 4.39 | 0.60 | 0.022 | {} |
| estimation | sweep R x | 0.5 | 1 | 100% | 55.4 | 0.016 | 4.43 | 0.31 | 0.029 | {} |
| estimation | sweep R x | 2 | 1 | 100% | 55.5 | 0.015 | 0.99 | 0.08 | 0.024 | {} |
| estimation | sweep R x | 4 | 1 | 100% | 55.3 | 0.014 | 0.49 | 0.04 | 0.027 | {} |
| estimation | sweep gate | 6.0 | 1 | 100% | 55.2 | 0.015 | 1.58 | 0.16 | 0.025 | {} |
| estimation | sweep gate | 13.8 | 1 | 100% | 55.2 | 0.015 | 1.58 | 0.16 | 0.025 | {} |
| estimation | sweep gate | None | 1 | 100% | 55.2 | 0.015 | 1.58 | 0.16 | 0.025 | {} |
| estimation | sweep odom cov x | 0.5 | 1 | 100% | 55.3 | 0.017 | 3.78 | 0.16 | 0.024 | {} |
| estimation | sweep odom cov x | 2 | 1 | 100% | 54.9 | 0.015 | 1.68 | 0.15 | 0.028 | {} |
| estimation | sweep odom cov x | 5 | 1 | 100% | 55.4 | 0.013 | 1.35 | 0.14 | 0.014 | {} |
| robust | encoder noise x | 0.5 | 1 | 100% | 95.8 | 0.008 | 0.55 | 0.25 | 0.007 | {} |
| robust | encoder noise x | 1 | 1 | 100% | 92.7 | 0.022 | 4.93 | 0.06 | 0.012 | {} |
| robust | encoder noise x | 2 | 1 | 100% | 54.9 | 0.012 | 0.29 | 0.03 | 0.008 | {} |
| robust | encoder noise x | 4 | 1 | 0% | 133.0 | 0.008 | 0.18 | nan | 1.417 | {"explore failed": 1} |
| robust | LiDAR noise x | 0.5 | 1 | 100% | 93.7 | 0.010 | 0.79 | 0.14 | 0.008 | {} |
| robust | LiDAR noise x | 1 | 1 | 100% | 92.7 | 0.022 | 4.93 | 0.06 | 0.012 | {} |
| robust | LiDAR noise x | 2 | 1 | 100% | 53.5 | 0.013 | 0.66 | 0.01 | 0.007 | {} |
| robust | LiDAR noise x | 4 | 1 | 100% | 90.5 | 0.011 | 0.45 | 0.02 | 0.013 | {} |
| robust | LiDAR dropout | 0.0 | 1 | 100% | 93.9 | 0.012 | 0.64 | 0.05 | 0.019 | {} |
| robust | LiDAR dropout | 0.05 | 1 | 100% | 92.3 | 0.023 | 5.33 | 0.06 | 0.010 | {} |
| robust | LiDAR dropout | 0.15 | 1 | 100% | 93.3 | 0.018 | 2.69 | 0.05 | 0.010 | {} |
| robust | LiDAR dropout | 0.3 | 1 | 100% | 80.4 | 0.013 | 0.74 | 0.14 | 0.011 | {} |
| robust | camera frame drop | 0.0 | 1 | 100% | 92.7 | 0.023 | 4.80 | 0.06 | 0.011 | {} |
| robust | camera frame drop | 0.1 | 1 | 100% | 96.5 | 0.015 | 1.06 | 0.05 | 0.000 | {} |
| robust | camera frame drop | 0.3 | 1 | 100% | 126.2 | 0.029 | 4.34 | 0.19 | 0.007 | {} |
| robust | camera frame drop | 0.5 | 1 | 100% | 40.5 | 0.009 | 1.28 | 0.07 | 0.003 | {} |
| robust | camera delay s (not compensated) | 0.0 | 1 | 100% | 92.3 | 0.021 | 4.60 | 0.06 | 0.009 | {} |
| robust | camera delay s (not compensated) | 0.1 | 1 | 100% | 92.9 | 0.032 | 7.32 | 0.07 | 0.008 | {} |
| robust | camera delay s (not compensated) | 0.2 | 1 | 100% | 92.8 | 0.042 | 8.91 | 0.07 | 0.008 | {} |
| robust | camera delay s (not compensated) | 0.4 | 1 | 100% | 41.0 | 0.073 | 61.67 | 0.14 | 0.026 | {} |
| robust | camera delay s (compensated) | 0.0 | 1 | 100% | 92.3 | 0.021 | 4.60 | 0.06 | 0.009 | {} |
| robust | camera delay s (compensated) | 0.1 | 1 | 100% | 91.9 | 0.023 | 5.39 | 0.06 | 0.015 | {} |
| robust | camera delay s (compensated) | 0.2 | 1 | 100% | 92.7 | 0.022 | 4.93 | 0.06 | 0.012 | {} |
| robust | camera delay s (compensated) | 0.4 | 1 | 100% | 41.0 | 0.047 | 54.04 | 0.11 | 0.026 | {} |
| robust | dock shift x | 0.0 | 1 | 100% | 54.3 | 0.010 | 0.50 | 0.02 | 0.023 | {} |
| robust | dock shift x | 1 | 1 | 100% | 92.7 | 0.022 | 4.93 | 0.06 | 0.012 | {} |
| robust | dock shift x | 2 | 1 | 100% | 52.0 | 0.011 | 0.32 | 0.02 | 0.010 | {} |
| robust | wheel slip x | 0.0 | 1 | 100% | 52.0 | 0.014 | 0.94 | 0.17 | 0.003 | {} |
| robust | wheel slip x | 1 | 1 | 100% | 92.7 | 0.022 | 4.93 | 0.06 | 0.012 | {} |
| robust | wheel slip x | 2 | 1 | 100% | 95.1 | 0.015 | 0.72 | 0.11 | 0.010 | {} |
| robust | wheel slip x | 3 | 1 | 100% | 96.0 | 0.008 | 0.91 | 0.33 | 0.011 | {} |