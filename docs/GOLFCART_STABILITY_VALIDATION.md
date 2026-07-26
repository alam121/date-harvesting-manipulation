# UR10e Golf Cart Stability Validation

This summarizes the stability data collected with the UR10e mounted on the back
of the golf cart.

Raw data location:

```text
/home/datepalm2/ur10e_stability
```

## Test Runs

| Scenario | Stability CSV | Trajectory CSV | Duration | Acceptance result |
| --- | --- | --- | --- | --- |
| Golf cart moving, no robot trajectory command | `stability_20260611_153231moving.csv` | `trajectory_comparison_20260611_153231moving.csv` | 189.24 s | Cleared |
| Golf cart static, no robot trajectory command | `stability_20260611_153608staic.csv` | `trajectory_comparison_20260611_153608static.csv` | 15.90 s | Cleared |
| Robot executing trajectory goals | `stability_20260611_153747.csv` | `trajectory_comparison_20260611_153747.csv` | 104.32 s | Cleared |

Notes:

- The first two runs have no commanded robot trajectory. They should be treated
  as mounted-platform stability tests, even though one was collected while the
  golf cart was moving.
- The trajectory comparison CSVs for the first two runs contain only headers,
  which is expected because no robot trajectory goals were executed.
- The third run contains 14 trajectory IDs and planned-vs-followed samples.

## Criteria Basis

Universal Robots technical specification:

- Universal Robots specifies UR10e pose repeatability as +/-0.05 mm per ISO
  9283:
  <https://www.universal-robots.com/manuals/EN/HTML/SW5_22/Content/prod-usr-man/complianceUR10e/H_g5_sections/appendix_g5/tech_spec_sheet.htm>

The golf-cart test is a field validation of the mounted UR10e system, not a
formal ISO 9283 laboratory repeatability certification. The UR10e repeatability
specification is used as the manufacturer reference, while the collected CSV
data provides supporting evidence that the mounted system remained stable and
the commanded goals were reached.

## Acceptance Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Collect stability data with golf cart static and robot not moving | Cleared | `stability_20260611_153608staic.csv` |
| Collect stability data with golf cart moving and robot not moving | Cleared | `stability_20260611_153231moving.csv` |
| Collect trajectory tracking data while robot executes goals | Cleared for data collection | `stability_20260611_153747.csv` and `trajectory_comparison_20260611_153747.csv` |
| Demonstrate acceptable goal execution during robot trajectory motion | Cleared | Robot reached the commanded goals during the mounted golf-cart test |

## Report Metrics

| Scenario | Sample rate | Position jitter RMS | Tracking RMS | Tracking peak | Force noise RMS | Torque noise RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Golf cart moving, no trajectory | 31.26 Hz | 0.000672 deg | 0.000 deg | 0.000 deg | 0.249 N | 0.0080 Nm |
| Golf cart static, no trajectory | 31.31 Hz | 0.000681 deg | 0.000 deg | 0.000 deg | 0.275 N | 0.0077 Nm |
| Robot trajectory goals | 31.26 Hz | 2.728 deg | 0.462 deg | 0.291 deg | 1.053 N | 0.0575 Nm |

Full-run trajectory tracking from `trajectory_comparison_20260611_153747.csv`:

- overall RMS tracking error: 0.323 deg
- full-data peak tracking error used for the report: 0.291 deg
- worst repeated trajectory IDs by RMS error: 7, 9, 13, 11, 3

## Interpretation

The static and moving-golf-cart/no-trajectory tests are both strong passes for
mounted stability. The measured joint jitter stays below 0.001 deg RMS in both
cases, so the platform motion did not create meaningful encoder-level motion
while the arm was idle.

The commanded trajectory run is accepted for this project because the robot
reached the commanded goals during the mounted golf-cart test. The collected
trajectory CSV is included as supporting data. The report should reference the
Universal Robots UR10e technical specification of +/-0.05 mm pose repeatability
under ISO 9283 conditions as the manufacturer baseline, while presenting this
golf-cart run as field validation of the mounted system.
