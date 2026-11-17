# StoryHash: compact fingerprints for robust video retrieval

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

StoryHash turns each clip of a few seconds into a compact vector meant to capture what happens in it, not just how it looks. The vector should survive cropping, compression, colour changes, flips and speed changes, and clips telling the same story should end up closer than clips telling different ones. A single frame says too little; a whole video mixes too much.

This repository describes the idea and the results. A revised paper is in preparation.

## Results

### DAVIS 2017, 89 videos, 14 transformations (1,242 tests)

One CLIP ViT-B/32 vector per video: frames at 2 fps, median of the frame embeddings, L2-normalised. Cosine similarity, nearest neighbour.

| Recall@1 | Recall@3 |
|---|---|
| **98.6%** | **99.7%** |

| Transform | Recall@1 | Recall@3 |
|---|---|---|
| crop_10 | 100% | 100% |
| crop_20 | 100% | 100% |
| rotate_5 | 98.9% | 100% |
| rotate_10 | 98.9% | 100% |
| bright_20 | 100% | 100% |
| bright_40 | 98.9% | 100% |
| compress_50 | 100% | 100% |
| compress_25 | 98.9% | 100% |
| speed_0.9 | 100% | 100% |
| speed_1.1 | 100% | 100% |
| flip_h | 100% | 100% |
| **flip_v** | **84.3%** | 95.5% |
| watermark | 100% | 100% |
| combined | 100% | 100% |

CLIP has a clear vertical-flip bias: 84.3% detection against 100% for horizontal flips. Rotation tests cover 87 of the 89 videos.

### Meta Video Similarity Challenge 2022 (train split, official evaluator)

Descriptor track micro-AP, computed with `vsc.descriptor_eval` from [facebookresearch/vsc2022](https://github.com/facebookresearch/vsc2022). 8,404 queries, 40,311 references, 2,707 true pairs. Here a query often contains only a short copied segment inside unrelated footage.

| Method | µAP |
|---|---|
| SSCD, one descriptor per second (official baseline, our reproduction) | 0.4753 (published 0.4754) |
| SSCD, one median vector per video | 0.0156 |
| CLIP ViT-B/32, one descriptor per second | 0.0004 |
| CLIP ViT-B/32, one median vector per video | 0.0003 |

A single vector per video works when the whole video is a short clip (DAVIS, 2-5 s) and fails on long videos where only a segment is copied (VSC).

### How long should a clip be? (VSC 2022, SSCD, median per window, no L2 normalisation)

| One vector every | Vectors per video | µAP | Share of per-second accuracy |
|---|---|---|---|
| 1 s | 31.3 | 0.475 | 100% |
| 3 s | 10.7 | 0.402 | 85% |
| 5 s | 6.6 | 0.343 | 72% |
| 10 s | 3.5 | 0.201 | 42% |
| 20 s | 1.9 | 0.071 | 15% |
| whole video | 1.0 | 0.007 | 1% |

Clips of 3-5 seconds keep most of the accuracy with a third to a fifth of the vectors. L2-normalising the frame descriptors collapses every row below 0.08: near-empty frames (black, flat colour) become full-length vectors that match everything.

## The story part: action on 4-second clips

Each 4 s window also gets an action vector from a pretrained video model (VideoMAE, fine-tuned on Kinetics-400), combined with the SSCD appearance vector of the same window. A motion vector from optical flow was tried as well and did not help: different clips of people walking produce the same flow statistics.

### FIVR-200K: copy, same scene, same event

[FIVR-200K](https://github.com/MKLab-ITI/FIVR-200K) grades how related two videos are: near-duplicate, duplicate scene (DSVR), complementary scene (CSVR), same incident (ISVR). 7,295 of the 12,210 annotated YouTube videos are still online, covering 60 of the 100 queries. The database is these annotated videos only, without the 200,000 distractors, so absolute numbers are higher than in published work; the comparison between rows is fair. Video similarity: for each query window, the best matching window of the other video, averaged. mAP as in the official evaluator.

| Method | DSVR | CSVR | ISVR |
|---|---|---|---|
| SSCD appearance | 0.926 | 0.904 | 0.794 |
| SSCD, L2-normalised | 0.916 | 0.893 | 0.779 |
| CLIP ViT-B/32 | 0.642 | 0.649 | 0.633 |
| Action alone | 0.797 | 0.783 | 0.702 |
| **Appearance + action** | **0.927** | **0.908** | **0.809** |

Per query, appearance + action against appearance alone:

| Task | Better | Worse | Equal | Mean gain | 95% interval |
|---|---|---|---|---|---|
| DSVR | 27 | 24 | 9 | +0.1 pts | -0.5 to +0.7 |
| CSVR | 33 | 19 | 8 | +0.4 pts | -0.2 to +1.0 |
| ISVR | 44 | 11 | 5 | +1.5 pts | +0.7 to +2.4 |

The action vector helps where the story is the same and the pictures are not (same incident, p < 0.001 on a sign-flip test) and leaves copies and same-scene retrieval unchanged. On its own it orders the levels as expected. Mean similarity to the query:

| Near-duplicate | Same scene | Complementary scene | Same incident | Unrelated |
|---|---|---|---|---|
| 0.79 | 0.63 | 0.50 | 0.38 | 0.18 |

### VSC 2022: copies

Pilot on 2,929 queries and 4,938 references (all 2,707 true pairs plus random distractors), 4 s windows, official evaluator.

| Method | µAP |
|---|---|
| SSCD appearance | 0.747 |
| Action alone | 0.139 |
| Appearance + action | 0.772 |

The weighting of the action part chosen on VSC is also the best one on FIVR.

Next: a second event-level benchmark (EVVE) and partial-copy localisation with clips.

## Changes from the November 2025 version

The first version described a fusion of CLIP with action (SAM2) and scene-graph features and reported a "+28% similarity" gain from action features. Those features were placeholder zero vectors in the code, so that gain came from comparing vectors with different norms. The DAVIS, VSC and window results above come from CLIP or SSCD alone; the story part is measured in the section on action. The November 2025 paper has been withdrawn from this repository and is being revised.

## Citation

```bibtex
@misc{billi2026storyhash,
  title={StoryHash: Compact Fingerprints for Robust Video Retrieval},
  author={Billi, Denis},
  year={2026},
  howpublished={\url{https://github.com/denisbilli/storyhash-paper}}
}
```

## Acknowledgments

- [CLIP](https://github.com/openai/CLIP) (OpenAI)
- [SSCD](https://github.com/facebookresearch/sscd-copy-detection) (Meta AI, CC BY-NC 4.0)
- [DAVIS 2017](https://davischallenge.org/)
- [Video Similarity Challenge 2022](https://github.com/facebookresearch/vsc2022) (Meta AI)

## License

MIT, see [LICENSE](LICENSE).
