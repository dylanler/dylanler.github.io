+++
title = 'Coordinates for an Unseen Camera'
date = 2026-03-17T07:42:00-07:00
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

## The argument the data supports

Synthetic data is often described as a way to create more examples. This experiment suggests a more interesting purpose. It lets us create examples whose hidden causes are known.

When a virtual camera moves, the renderer knows every pose along the path. The caption should not throw that knowledge away. If we preserve it, a generated clip can be evaluated against the exact journey requested, not merely whether it “looks cinematic.”

The first discovery on an uncharted path is sometimes a coordinate system. Once we can name where the camera went, we can finally ask whether the model followed.
