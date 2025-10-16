# Jotunn

A petabyte scale data processing framework for AI models using Daft + Ray.

This framework is still under active development and breaking changes may be introduced.

## Installation
```python
uv pip install jotunn
```
Install specific multimodal components
```python
# Image
uv pip install jotunn[image]

# Text
uv pip install jotunn[text]

# Video
uv pip install jotunn[video]

# Audio
uv pip install jotunn[audio]

# Everything
uv pip install jotunn[all]
```

## Community
[Join our Discord community](https://discord.gg/Fh4DfwQGhd)

## Examples

A variety of examples can be found [here](https://github.com/teraflop-ai/jotunn/tree/main/examples).

## Coverage

Qwen Image
- Stage 1
    - [x] Image Resizing
    - [x] Broken File Filter - Daft returns null if can not decode image
    - [x] File Size Filter
    - [x] Resolution Filter
    - [x] Deduplication Filter - requires using the [Frigg](https://github.com/teraflop-ai/frigg) library
    - [x] NSFW Filter
- Stage 2
    - [x] Rotation Filter
    - [x] Clarity Filter
    - [x] Luma Filter
    - [x] Saturation Filter
    - [x] Entropy Filter
    - [] Texture Filter - It is unclear to me what is used for this.
- Stage 3
    - [x] Raw Caption Split - requires the [Frigg](https://github.com/teraflop-ai/frigg) library.
    - [x] Recaption Split - requires the [Frigg](https://github.com/teraflop-ai/frigg) library.
    - [x] Fused Caption Split - requires the [Frigg](https://github.com/teraflop-ai/frigg) library.
    - [x] CLIP and SIGLIP Filter
    - [x] Token Length Filter
    - [x] Invalid Caption Filter
- Stage 4
    - [] Synthetic Text Rendering
    - [x] Intensive Text Filter
    - [] Small Character Filter
    - [] Image Categorization and Tagging
- Stage 5
    - [x] Image Quality Filter - It is unclear whether this is a combination of different heuristics. 
    - [x] Resolution Filter - Same as Stage 1.
    - [x] Aesthetic Filter - Should be improved through a community effort.
    - [] Abnormal Element Filter - Needs to be trained but the data is being collected.
- Stage 6
    - [] Face Mosaic Filter
    - [] Image Categorization and Tagging
    - [x] Synthetic Captioning
- Stage 7
    - [x] Resolution Filter - Same as Stage 1.
    - [] Image Categorization and Tagging 

## Citation
```bibtex
@misc{shippole2025jotunn,
    title   = {Jotunn: Distributed Multimodal Data Processing at Petabyte Scale},
    author  = {Enrico Shippole},
    year    = {2025},
}
```