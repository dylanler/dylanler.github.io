+++
title = 'Coordinates for an Unseen Camera'
date = 2026-03-17T07:42:00-07:00
lastmod = 2026-08-27T10:00:00-07:00
draft = false
tags = ["AI", "video", "synthetic data", "experiments", "discovery"]
+++

A camera moves left. Or perhaps the subject moves right. The pixels alone do not tell us which coordinate system the sentence meant.

That ambiguity became the experimental question for this month:

> Does adding explicit camera coordinates make a movement label meaningfully more reconstructable than ordinary cinematic language?

The larger camera dataset project in this repository proposes generated environments, depth estimation, scene reconstruction, scripted camera paths, and captions derived from those paths. There is no completed video model benchmark in the repository yet, so I did not manufacture one. Instead, I tested the assumption underneath the pipeline: whether a label can preserve the pose that created it.

## Hypothesis

A vague label such as “pan left and tilt up” identifies a region of motion. A coordinate label such as “pan negative 17 degrees, tilt positive 6 degrees” identifies a pose. If the dataset is meant to teach controllable cinematography, reconstruction error should fall sharply when the numeric state is retained.

## Method

I enumerated 15,617 camera poses on a deterministic grid. Pan ranged from negative 40 to positive 40 degrees. Tilt ranged from negative 24 to positive 24 degrees. Both axes advanced in half degree increments.

Each pose was encoded two ways.

1. The vague encoder emitted left, center, or right for pan and up, level, or down for tilt. Reconstruction used the center of each named region.

2. The precise encoder rounded each axis to the nearest whole degree.

The primary metric was mean absolute angular error. The second metric was the percentage of poses reconstructed within two degrees on both axes. The complete [reproduction script](https://github.com/dylanler/dylanler.github.io/blob/main/experiment-tools/frontier_camera_label_eval.py) uses only the Python standard library.

```python
def encode_vague(pan, tilt):
    pan_hat = -24 if pan < -8 else 24 if pan > 8 else 0
    tilt_hat = -14.5 if tilt < -5 else 14.5 if tilt > 5 else 0
    return pan_hat, tilt_hat

def encode_precise(pan, tilt):
    return round(pan), round(tilt)

pan_hat, tilt_hat = encode_precise(pan, tilt)
error = abs(pan - pan_hat) + abs(tilt - tilt_hat)
```

This is a label reconstruction experiment, not a claim about generated video quality. It isolates whether the annotation itself contains enough information to recover the intended camera state.

## Results

| Label scheme | Pan MAE | Tilt MAE | Both axes within 2° |
|---|---:|---:|---:|
| Vague directions | 7.20° | 4.29° | 4.67% |
| Rounded coordinates | 0.25° | 0.25° | 100.00% |

![Camera label reconstruction error](/images/frontier-camera-error.svg)

The coordinate label reduced pan error by 96.6 percent and tilt error by 94.2 percent. More importantly, the joint tolerance rate moved from 4.67 percent to 100 percent.

## Rate distortion, not just accuracy

Technical annotation systems trade representation size against reconstruction error. I reran the grid as a simple rate distortion experiment with three codecs. The rate is the base two logarithm of the number of states the label can express. Distortion is Euclidean angular error:

```text
D(pose, reconstruction) = sqrt((pan - pan_hat)^2 + (tilt - tilt_hat)^2)
R(codec) = log2(number of representable states)
```

| Codec | States | Rate | Mean error | P95 error | P99 error |
|---|---:|---:|---:|---:|---:|
| Vague 3 by 3 vocabulary | 9 | 3.17 bits | 9.032° | 15.882° | 17.219° |
| Whole degree coordinates | 3,969 | 11.95 bits | 0.424° | 0.707° | 0.707° |
| Half degree coordinates | 15,617 | 13.93 bits | 0.000° | 0.000° | 0.000° |

![Camera annotation rate distortion curve](/images/frontier-camera-rate-distortion.svg)

An additional 8.78 bits per pose moves the P95 error from 15.882 degrees to 0.707 degrees. The next 1.98 bits eliminate quantization error on this grid. That is a concrete storage and supervision tradeoff, not merely an argument for “more detail.”

The new [technical audit script](https://github.com/dylanler/dylanler.github.io/blob/main/experiment-tools/frontier_technical_audit.py) generates this table and writes the complete machine readable [audit result](https://github.com/dylanler/dylanler.github.io/blob/main/experiment-tools/results/frontier_technical_audit.json).

The result is almost embarrassingly strong, but that is useful. It means the first uncertainty in the project is resolved. A directional word is not a sufficient training target for precise control.

{{< frontier mode="camera" id="march-camera" >}}

Move the controls above. The readout changes because the prompt is tied to a physical state. That relationship is what the dataset needs to preserve.

## What this does not prove

This experiment does not show that a video model will obey the coordinates. It does not test occlusion, subject motion, focal length, camera roll, acceleration, or whether a human would prefer the resulting shot. It only shows that the label no longer destroys the pose before training begins.

That limitation determines the next experiment. Render trajectories from Blender or another scene engine, hide the source parameters, and ask a pose estimator to recover them from the clip. Then compare four annotation families:

| Condition | Information retained | Expected failure |
|---|---|---|
| Cinematic phrase | Movement category | Large endpoint variance |
| Phrase plus duration | Category and time | Unknown magnitude |
| Start and end pose | Geometry | Unknown velocity profile |
| Full trajectory | Geometry and rhythm | Caption complexity |

The right dataset may need both human language and machine coordinates. Language carries intention. Coordinates carry accountability.

## Dataset contract and leakage controls

The training record should carry the physical state separately from the natural language realization. A compact schema looks like this:

```json
{
  "scene_id": "atrium_0042",
  "trajectory_id": "arc_0187",
  "fps": 24,
  "poses": [[0.0, 1.6, 4.2, 0.0, -6.0, 0.0], [0.03, 1.6, 4.18, 0.5, -5.8, 0.0]],
  "caption": "Arc right while the camera rises and keeps the subject centered",
  "subject_screen_path": [[0.50, 0.51], [0.50, 0.50]]
}
```

The split must be grouped by `scene_id`, not by rendered clip. Otherwise the same reconstructed environment can appear in training and evaluation with only a different camera path. That leakage would let a model memorize scene geometry and inflate control scores.

For evaluation I would report translation error in meters, rotation geodesic error in degrees, dynamic time warping over the full pose sequence, and subject screen path error. Endpoint error alone cannot distinguish a smooth arc from a camera that teleports to the correct final pose.

## The argument the data supports

Synthetic data is often described as a way to create more examples. This experiment suggests a more interesting purpose. It lets us create examples whose hidden causes are known.

When a virtual camera moves, the renderer knows every pose along the path. The caption should not throw that knowledge away. If we preserve it, a generated clip can be evaluated against the exact journey requested, not merely whether it “looks cinematic.”

The first discovery on an uncharted path is sometimes a coordinate system. Once we can name where the camera went, we can finally ask whether the model followed.
