# Optional heavier tracking: investigation

Status: research only. No new tracking mode or replacement footage is deployed.

## Recommendation

Keep three distinct choices: **Off** (detection only), **Standard** (the current browser motion/appearance tracker), and an eventual **Enhanced** mode combining dedicated person ReID with stronger association. Keep detector size independent: YOLO26x improves detections but does not itself decide persistent identities.

The first enhanced candidate should use OSNet person embeddings with BoT-SORT-style motion association, a gallery of clean appearance observations, and confidence/occlusion-aware gallery updates. Test joint motion/appearance costs and rejection of contradictory appearance evidence. Do not blindly reject appearance mismatches from heavily occluded crops: that can create new IDs instead of resolving a crossing. The existing browser tracker already freezes histogram updates during overlap; a neural mode should preserve and evaluate that protection.

## Experiments on the actual showcase continuation

Environment: isolated Ultralytics 8.4.148, PyTorch 2.7.0+cu118, RTX 3060. Source: OpenCV `vtest.avi`, frames 404–553 inclusive, 150 frames at 10 fps. Source frame numbers are zero-based. Review frame 100 means source frame 504.

YOLO26x at image size 1280 and confidence 0.05 recovered additional partial detections around the pole compared with earlier runs. Some are duplicate head/body boxes. Better recall alone does not establish correct tracking.

| Candidate | Observed result | Decision |
| --- | --- | --- |
| YOLO26x/1280 + BoT-SORT + YOLO26n ReID | Initial IDs 1–4 persisted after confirmation; later ID 7 changed from the blonde person to another entrant between review frames 100 and 140. | Fails identity gate |
| Same cached detector output + BoT-SORT + OSNet x1.0 | Blonde person has ID 7 at frame 100, ID 9 at frame 140; ID 7 is on the other entrant at frame 140. | Fails identity gate |
| Same cached detector output + Deep OC-SORT + OSNet x1.0 | Blonde person has ID 7 at frame 100 and ID 10 at frame 140. | Fails identity continuity gate |

OSNet used the author's MSMT17 combine-all weights, RGB crops resized to 256×128, ImageNet normalization, and normalized 512-dimensional embeddings. Both OSNet trackers received exactly the same cached detections and embeddings. High/low/new track thresholds were 0.30/0.08/0.50, buffer 30. Other settings were Ultralytics defaults, except Deep OC-SORT enabled `use_byte`. BoT-SORT used sparse optical-flow camera compensation; Deep OC-SORT used its default, so this is a comparison of configured pipelines, not an isolated association ablation.

OSNet crop preparation, GPU embeddings and CPU transfer took about **2.49 seconds for 150 frames** in one run. This excludes detector inference, tracking and rendering; it is not end-to-end FPS or a browser/mobile benchmark. The separate YOLO ReID ONNX experiment fell back to CPU because its installed GPU runtime could not load required CUDA libraries, so its total runtime is not comparable.

These failures are directly visible and sufficient to reject the candidates. They are not an exhaustive ground-truth annotation, HOTA/IDF1 measurement or a ranking of the underlying algorithms. Distinct ID counts alone are not accuracy measurements.

## Why heavier appearance features did not solve everything

In the installed BoT-SORT implementation, `get_dists` combines IoU and appearance costs using their minimum. A strong box-overlap match can therefore survive poor appearance agreement. Updating the appearance template after a mistaken assignment can reinforce the mistake. These are mechanisms worth testing, not a proven causal diagnosis of every observed switch.

## Other options

- **Deep OC-SORT:** observation-centric motion and adaptive appearance matching; tested here, but its configured run did not maintain the blonde person's identity.
- **StrongSORT:** another appearance-heavy comparison candidate. Evaluate its online tracker separately from StrongSORT++ offline linking/interpolation.
- **BoostTrack:** useful next comparison for confidence and association handling. Benchmark on identical cached detections before claiming a gain.
- **Offline bidirectional tracking/tracklet linking:** appropriate for generating a reviewed portfolio recording, but uses future frames and must not be described as the live browser algorithm. Do not manually rewrite identities or fabricate detections to pass the gate.

## Browser integration and acceptance

The Python trackers tested here are not drop-in browser components. An enhanced browser mode would need ONNX person-ReID inference and tested JavaScript association logic. Load the ReID model only when requested, batch person crops, and measure frame latency and memory on desktop and mobile. Turning it off should stop the extra inference and release the additional session. Preserve local video processing; a server GPU mode would be a separate explicit data-flow choice.

Before shipping, evaluate both the crossing and independent footage with annotated identities. Measure switches, fragmentation, misses, extra boxes and latency. Only replace the portfolio recording after every frame of at least 15 continuous seconds passes the existing promotion gate. No tested continuation currently qualifies.

## Sources

- [Ultralytics tracking and optional ReID](https://docs.ultralytics.com/modes/track)
- [OSNet / Torchreid](https://github.com/KaiyangZhou/deep-person-reid), [author's weights](https://huggingface.co/kaiyangzhou/osnet)
- [BoT-SORT](https://github.com/NirAharon/BoT-SORT)
- [Deep OC-SORT](https://github.com/GerardMaggiolino/Deep-OC-SORT)
- [StrongSORT](https://github.com/dyhBUPT/StrongSORT)
- [BoostTrack](https://github.com/vukasin-stanojevic/BoostTrack)
