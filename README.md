
<div align="center">
<h1>AI Papers</h1>
<p>A collection of awesome AI papers.</p>
</div>

## Table of Contents
- [Dataset](#dataset)
  - [Dataset for LLM](#dataset-for-llm)
- [Speech and Audio AI](#speech-and-audio-ai)
  - [Speech to Text](#speech-to-text)
  - [Text to Speech](#text-to-speech)
- [Vision](#vision)
  - [Image Enhancement](#image-enhancement)

## Dataset

### Dataset for LLM
- Liu, Yang, et al. "**Datasets for large language models: A comprehensive survey.**" arXiv preprint arXiv:2402.18041 (2024). [paper](https://arxiv.org/pdf/2402.18041)
  ```
  Discusses different datasets for training LLMs
  ```

## Speech and Audio AI

### Speech to Text

- Wei, Kun, et al. "**Conversational speech recognition by learning audio-textual cross-modal contextual representation.**" IEEE/ACM Transactions on Audio, Speech, and Language Processing 32 (2024): 2432-2444. [paper](https://arxiv.org/pdf/2310.14278)
  ```text
  Describes the limitations and challenges of ASR in conversational systems and presents a 
  novel ASR systems based on an attention-based Conformer architecture, Conditional VAE.
  ```
- Nozaki, Jumon, and Tatsuya Komatsu. "**Relaxing the conditional independence assumption of CTC-based ASR by conditioning on intermediate predictions.**" arXiv preprint arXiv:2104.02724 (2021). [paper](https://www.isca-archive.org/interspeech_2021/nozaki21_interspeech.pdf)
  ```text
  CTC-based ASR suffer from the "Conditional Independence Assumption", which affects its accuracy.
  This paper proposes intermediate prediction to relax the assumption problem.
  ```
- Watanabe, Shinji, et al. "**Hybrid CTC/attention architecture for end-to-end speech recognition.**" IEEE Journal of Selected Topics in Signal Processing 11.8 (2017): 1240-1253. [paper](https://merl.com/publications/docs/TR2017-190.pdf)
  ```text
  This paper discusses CTC & Transformer-based ASR architectures and proposes hybrid CTC/attention
  end-to-end ASR system for better accuracy and performance.
  ```

### Text to Speech

- Tan, Xu, et al. "**A survey on neural speech synthesis.**" arXiv preprint arXiv:2106.15561 (2021). [paper](https://arxiv.org/pdf/2106.15561)
  ```text
  Talks about history of text to speech, speech synthesis. Focuses mostly on
  TTS based on neural networks.
  ```
- Ren, Yi, et al. "**Fastspeech 2: Fast and high-quality end-to-end text to speech.**" arXiv preprint arXiv:2006.04558 (2020). [paper](https://arxiv.org/pdf/2006.04558v8)
  ```text
  Discusses rapid progress of TTS, describes limitations of autoregressive TTS models.
  Presents non-autoregressive TTS models as a solution.
  ```

## Vision 

### Image Enhancement

- Liu, Fangxue, and Lei Fan. "**A review of advancements in low-light image enhancement using deep learning.**" arXiv preprint arXiv:2505.05759 (2025). [paper](https://www.arxiv.org/pdf/2505.05759)
  ```text
  This paper discusses issues of low-light conditions in CV tasks and explains
  limitations of hardware-based methods and discusses different methods to address this.
  ```
- Liang, Jingyun, et al. "**Swinir: Image restoration using swin transformer.**" Proceedings of the IEEE/CVF international conference on computer vision. 2021. [paper](https://arxiv.org/pdf/2108.10257)
  ```
  It presents SwinIR for image restoration based on Swin Transformer. It achieves
  SOTA performance on restoration, denoising and compression tasks.
  ```
- Wang, Zhihao, Jian Chen, and Steven CH Hoi. "**Deep learning for image super-resolution: A survey.**" IEEE transactions on pattern analysis and machine intelligence 43.10 (2020): 3365-3387. [paper](https://arxiv.org/pdf/1902.06068)
  ```
  This paper discusses the recent development in image super-resolution with
  deep learning. 
  ```
- Yang, Fuzhi, et al. "**Learning texture transformer network for image super-resolution.**" Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020. [paper](https://arxiv.org/pdf/2006.04139)
  ```
  It describes limitations of existing approaches and presents texture Transformer
  Network for Image SuperResolution, optimized for image super-resolution tasks.
  ```
- Ledig, Christian, et al. "**Photo-realistic single image super-resolution using a generative adversarial network.**" Proceedings of the IEEE conference on computer vision and pattern recognition. 2017. [paper](https://arxiv.org/pdf/1609.04802)
  ```
  It presents SRGAN, capable of inferring photo realistic images for 4x upscaling.
  Also proposes a perpetual loss function.
  ```
- Dong, Chao, Chen Change Loy, and Xiaoou Tang. "**Accelerating the super-resolution convolutional neural network.**" European conference on computer vision. Cham: Springer International Publishing, 2016. [paper](https://arxiv.org/pdf/1608.00367)
  ```
  This paper highlights how they accelerate SRCNN by introducing deconvolution layer
  at the end of the network, achieving acceleration of more than 40x. 
  ```
- Dong, Chao, et al. "**Image super-resolution using deep convolutional networks.**" IEEE transactions on pattern analysis and machine intelligence 38.2 (2015): 295-307. [paper](https://arxiv.org/pdf/1501.00092)
  ```
  It presents a deep learning method for single image super resolution tasks
  called Super-Resolution Convolution Neural Network or SRCNN.
  ```
