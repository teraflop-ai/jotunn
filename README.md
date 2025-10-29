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

Step-Video-T2V
- Stage 1
    - [x] Video Scene Segmentation
    - [x] Frame Extraction
    - [x] Duration Filter
    - [x] Clip Score Filter
- Stage 2
    - [x] Aesthetic Filter
    - [x] NSFW Filter
    - [x] Resolution Filter
    - [x] Watermark Filter
- Stage 3
    - [x] Motion Filter - partial
    - [x] Blur Filter
- Stage 4
    - [] Source Filter - unclear what this is to me
    - [x] Saturation Filter - partial
- Stage 5
    - [x] Subtitle Filter
    - [] Black Border Filter
    - [x] Video Captioning
- Stage 6
    - [x] Concept Balancing Filter - partial

HunyuanVideo
- Stage 1
    - [x] Video Scene Segmentation
    - [x] Deduplication
    - [x] Embedding
    - [x] Concept Balancing - partial
    - [x] Optical Flow
    - [x] Duration Filter
    - [x] Resolution Filter
    - [x] Transnet v2
- Stage 2
    - [x] Laplacian Blur Filter
    - [x] Intensive Text Filter
    - [] Visual Blur Filter - needs to be trained
- Stage 3
    - [] Dover - not sure this is worth adding. only for completeness
    - [] YOLOX Watermark, Borders, Logos - needs to be trained
- Stage 4
    - [x] Captioning

Wan
- [x] Intensive Text Filter
- [x] Aesthetic Filter 
- [x] NSFW Filter
- [x] Watermark Filter
- [] Logo Filter - needs to be trained
- [] Black Border Detection
- [x] Exposure Filter
- [] Synthetic Image Filter - needs to be trained
- [x] Blur Filter
- [x] Duration Filter
- [x] Resolution Filter
- [x] Clustering
- [] Scoring - needs to be trained
- [x] Motion Quality - partial
- [x] Captioning 


## Citation
If you find this framework useful, please feel free to cite:
```bibtex
@misc{shippole2025jotunn,
    title   = {Jotunn: Distributed Multimodal Data Processing at Petabyte Scale},
    author  = {Enrico Shippole},
    year    = {2025},
}
```
## Acknowledgements
```bibtex
@misc{eva02_large_patch14_448_dbv4_full_2024,
  title        = {EVA02-Large-Patch14-448 Anime Image Tagger},
  author       = {narugo1992 and Deep Generative anime Hobbyist Syndicate (DeepGHS)},
  year         = {2024},
  howpublished = {\url{https://huggingface.co/animetimm/eva02_large_patch14_448.dbv4-full}},
  note         = {A large-scale anime-style image classification model based on EVA02-Large architecture, fine-tuned on Danbooru dataset for multi-label tagging with 12,476 tags including general, character, and rating categories. Model parameters: 316.8M, input resolution: 448×448.},
  license      = {GPL-3.0}
}
```
```bibtex
@misc{ma2025stepvideot2vtechnicalreportpractice,
      title={Step-Video-T2V Technical Report: The Practice, Challenges, and Future of Video Foundation Model}, 
      author={Guoqing Ma and Haoyang Huang and Kun Yan and Liangyu Chen and Nan Duan and Shengming Yin and Changyi Wan and Ranchen Ming and Xiaoniu Song and Xing Chen and Yu Zhou and Deshan Sun and Deyu Zhou and Jian Zhou and Kaijun Tan and Kang An and Mei Chen and Wei Ji and Qiling Wu and Wen Sun and Xin Han and Yanan Wei and Zheng Ge and Aojie Li and Bin Wang and Bizhu Huang and Bo Wang and Brian Li and Changxing Miao and Chen Xu and Chenfei Wu and Chenguang Yu and Dapeng Shi and Dingyuan Hu and Enle Liu and Gang Yu and Ge Yang and Guanzhe Huang and Gulin Yan and Haiyang Feng and Hao Nie and Haonan Jia and Hanpeng Hu and Hanqi Chen and Haolong Yan and Heng Wang and Hongcheng Guo and Huilin Xiong and Huixin Xiong and Jiahao Gong and Jianchang Wu and Jiaoren Wu and Jie Wu and Jie Yang and Jiashuai Liu and Jiashuo Li and Jingyang Zhang and Junjing Guo and Junzhe Lin and Kaixiang Li and Lei Liu and Lei Xia and Liang Zhao and Liguo Tan and Liwen Huang and Liying Shi and Ming Li and Mingliang Li and Muhua Cheng and Na Wang and Qiaohui Chen and Qinglin He and Qiuyan Liang and Quan Sun and Ran Sun and Rui Wang and Shaoliang Pang and Shiliang Yang and Sitong Liu and Siqi Liu and Shuli Gao and Tiancheng Cao and Tianyu Wang and Weipeng Ming and Wenqing He and Xu Zhao and Xuelin Zhang and Xianfang Zeng and Xiaojia Liu and Xuan Yang and Yaqi Dai and Yanbo Yu and Yang Li and Yineng Deng and Yingming Wang and Yilei Wang and Yuanwei Lu and Yu Chen and Yu Luo and Yuchu Luo and Yuhe Yin and Yuheng Feng and Yuxiang Yang and Zecheng Tang and Zekai Zhang and Zidong Yang and Binxing Jiao and Jiansheng Chen and Jing Li and Shuchang Zhou and Xiangyu Zhang and Xinhao Zhang and Yibo Zhu and Heung-Yeung Shum and Daxin Jiang},
      year={2025},
      eprint={2502.10248},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.10248}, 
}
```
```bibtex
@article{kong2024hunyuanvideo,
  title={Hunyuanvideo: A systematic framework for large video generative models},
  author={Kong, Weijie and Tian, Qi and Zhang, Zijian and Min, Rox and Dai, Zuozhuo and Zhou, Jin and Xiong, Jiangfeng and Li, Xin and Wu, Bo and Zhang, Jianwei and others},
  journal={arXiv preprint arXiv:2412.03603},
  year={2024}
}
```
```bibtex
@misc{wan2025wanopenadvancedlargescale,
      title={Wan: Open and Advanced Large-Scale Video Generative Models}, 
      author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and Yingya Zhang and Yitong Huang and Yong Li and You Wu and Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and Yupeng Shi and Yutong Feng and Zeyinzi Jiang and Zhen Han and Zhi-Fan Wu and Ziyu Liu},
      year={2025},
      eprint={2503.20314},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.20314}, 
}
```