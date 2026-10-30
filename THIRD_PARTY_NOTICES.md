# Third-party notices

Supplemental evaluation images and annotations: [Microsoft COCO val2017](https://cocodataset.org/#download). The sample subset retains original image IDs, source URLs, annotation IDs, Flickr URLs and individual image license records in `source-subset.json`. Weather degradations are deterministic synthetic additions; the subset is not the original GridFormer restoration test split. No prediction-derived pseudo labels are used as ground truth.

GridFormer source and pretrained checkpoints: [TaoWangzj/GridFormer](https://github.com/TaoWangzj/GridFormer), Tao Wang and collaborators. The model is described in *GridFormer: Residual Dense Transformer with Grid Structure for Image Restoration in Adverse Weather Conditions*, IJCV 2024. The project downloads original architecture/checkpoint from a pinned commit/official Drive link; preserve upstream license notices and model terms when redistributing. No claim of authorship over GridFormer is made.

YOLOv8 and Ultralytics: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics), AGPL-3.0 / alternative commercial licensing from the copyright holder. Export tooling refers to official pretrained YOLOv8n and YOLOv8n-seg. Preserve applicable source disclosure, license and attribution obligations when distributing derivatives or weights.

Google LiteRT/TensorFlow Lite: [google-ai-edge/LiteRT](https://github.com/google-ai-edge/LiteRT), Apache-2.0. Bundled Android Maven artifacts retain their own notices. AndroidX, Kotlin, JUnit, PyTorch, NumPy, Pillow, einops, gdown and conversion dependencies retain their upstream licenses. Downloaded pretrained weights are not packaged into the source ZIP or APK by default.
