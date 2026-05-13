# Patch Forecaster Models

This folder provides two model variants for the new motion-target pipeline.

## Files

- `patch_forecaster.py`
  - `MotionPatchPretrainForecaster`
  - `MotionPatchCompletionForecaster`
  - `MotionPatchConfig`

## Variants

### 1. `MotionPatchPretrainForecaster`

Pretraining / no-completion version.

Pipeline:

1. Split the history sequence into fixed patches
2. Build point features with time embedding
3. Encode each patch into a patch token
4. Run patch graph attention + transformer over patch tokens
5. Use the last context token and last observation to predict future normalized motion

This is the simpler backbone-only forecasting model.

### 2. `MotionPatchCompletionForecaster`

Completion / recovery version.

Pipeline:

1. Same patch backbone as the pretrain model
2. Student branch masks one valid patch
3. Teacher branch sees the full patch sequence
4. Forecast future normalized motion from the student contextual tokens
5. Add patch recovery loss on the masked patch
6. Optionally add patch completion loss

This is the teacher-student masked-patch version.

## Input Batch Contract

Both models currently expect:

- `history`: `(B, L, C)`
- `history_mask`: `(B, L)`
- `future_dt`: `(B, H)`
- `future_pos`: `(B, H, 2)`
- `future_motion_norm`: `(B, H, 2)`

Notes:

- `L` is expected to be `npatch * patch_len`
- if not, the current implementation pads or truncates automatically
- motion reconstruction uses `target_mode` in config:
  - `velocity`
  - `displacement`

## Current Status

This implementation follows the repository's current trajectory-forecasting
pipeline and is intended to remain easy to inspect, extend, and reproduce.
