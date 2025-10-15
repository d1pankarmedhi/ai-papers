
<div align="center">
<h1>AI Papers</h1>
<p>A collection of awesome AI papers.</p>
</div>

## Table of Contents
- [Dataset](#dataset)
  - [Dataset for LLM](#dataset-for-llm)
- [LLM](#llm)
  - [Recursion](#recursion)
  - [Diffusion Transformer](#diffusion-transformer)
- [Speech and Audio AI](#speech-and-audio-ai)
  - [Speech to Text](#speech-to-text)
  - [Text to Speech](#text-to-speech)
- [Vision](#vision)
  - [Image Enhancement](#image-enhancement)
  - [Image Generation](#image-generation)
- [Optimization](#optimization)
  - [Flash Attention](#flash-attention)
- [AI&ML Engineering](#aiml-engineering)

## Dataset

### Dataset for LLM
- Liu, Yang, et al. "**Datasets for large language models: A comprehensive survey.**" arXiv preprint arXiv:2402.18041 (2024). [paper](https://arxiv.org/pdf/2402.18041)
  ```
  Discusses different datasets for training LLMs
  ```

## LLM

### Recursion 
- Jolicoeur-Martineau, Alexia. "**Less is More: Recursive Reasoning with Tiny Networks.**" arXiv preprint arXiv:2510.04871 (2025). [paper](https://arxiv.org/pdf/2510.04871v1)
  ```
  Introduces TRM, a Tiny Recursion Model (7M), uses a recursion reasoning approach,
  achieving a higher generalisation score than HRML.
  ```

  <table>
    <tbody>
      <tr>
        <td> <img width="100" alt="image" src="https://github.com/user-attachments/assets/81e8688a-9762-4d0f-b0d2-7e8b433fecba" /></td>
        <td> <img width="300" alt="image" src="https://github.com/user-attachments/assets/b3962712-e3ec-4828-a748-f9dfe0dfb45a" /></td>
        <td> <img width="250" alt="image" src="https://github.com/user-attachments/assets/835959ae-2be8-44b6-84cd-ff67bcf64877" /></td>
      </tr>
    </tbody>
  </table>


### Diffusion Transformer 
- Nie, Shen, et al. "**Large language diffusion models.**" arXiv preprint arXiv:2502.09992 (2025). [paper](https://arxiv.org/pdf/2502.09992)
  ```
  This paper discusses LLaDA, a diffusion model trained for language modelling,
  competing with auto-regressive language models.
  ```
  <table>
    <tbody>
      <tr>
        <td> <img width="300" height="364" alt="image" src="https://github.com/user-attachments/assets/ef501f29-97c3-4309-b82b-539fc85fc5cc" /></td>
        <td><img width="850" height="276" alt="image" src="https://github.com/user-attachments/assets/afede1ba-3574-4ea3-a7b3-e72641666208" /></td>
      </tr>
    </tbody>
  </table>


## Speech and Audio AI

### Speech to Text

- Wei, Kun, et al. "**Conversational speech recognition by learning audio-textual cross-modal contextual representation.**" IEEE/ACM Transactions on Audio, Speech, and Language Processing 32 (2024): 2432-2444. [paper](https://arxiv.org/pdf/2310.14278)
  ```text
  Describes the limitations and challenges of ASR in conversational systems and presents a 
  novel ASR systems based on an attention-based Conformer architecture, Conditional VAE.
  ```
  <table>
    <tbody>
      <tr>
        <td> <img width="300"  alt="image" src="https://github.com/user-attachments/assets/39d97b9a-215b-403d-a42d-f3e41e981073" /></td>
        <td><img width="600"  alt="image" src="https://github.com/user-attachments/assets/da6bc287-e747-4b6d-adc1-123c0ed00339" /></td>
      </tr>
    </tbody>
  </table>

- Nozaki, Jumon, and Tatsuya Komatsu. "**Relaxing the conditional independence assumption of CTC-based ASR by conditioning on intermediate predictions.**" arXiv preprint arXiv:2104.02724 (2021). [paper](https://www.isca-archive.org/interspeech_2021/nozaki21_interspeech.pdf)
  ```text
  CTC-based ASR suffer from the "Conditional Independence Assumption", which affects its accuracy.
  This paper proposes intermediate prediction to relax the assumption problem.
  ```
  <table>
    <tbody>
      <tr>
        <td> <img width="900" alt="image" src="https://github.com/user-attachments/assets/78410117-fec3-47aa-9956-c23cbb1a8d67" /></td>
        <td><img width="300"  alt="image" src="https://github.com/user-attachments/assets/a0ef47ed-ab77-43dc-a6cc-b8755bbfaf8e" /></td>
      </tr>
    </tbody>
  </table>
  
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

### Image Generation

- Peebles, William, and Saining Xie. "**Scalable diffusion models with transformers**." Proceedings of the IEEE/CVF international conference on computer vision. 2023. [paper](https://arxiv.org/pdf/2212.09748)
  ```
  This paper introduces Diffusion models based on Transformer architectures,
  replacing the UNet with a Transformer that operates on latent patches.
  ```

  <table>
    <tbody>
      <tr>
        <td> <img width="400" alt="image" src="https://github.com/user-attachments/assets/e44389b8-f8a0-4109-8d6d-94b134733ee6" /></td>
        <td><img width="400"  alt="image" src="https://github.com/user-attachments/assets/45113673-e82b-4ed8-81bd-fa6e38034b95" /></td>
      </tr>
    </tbody>
  </table>
  
- Zhang, Lvmin, Anyi Rao, and Maneesh Agrawala. "**Adding conditional control to text-to-image diffusion models.**" Proceedings of the IEEE/CVF international conference on computer vision. 2023. [paper](https://arxiv.org/pdf/2302.05543)
  ```
  This paper presents ControlNet, a neural network architecture for controlling Text-to-Image
  models. It describes "zero convolution", connects trainable diffusion model to original, locked model. 
  ```

  <table>
    <tbody>
      <tr>
        <td> <img width="200" alt="image" src="https://github.com/user-attachments/assets/8d56ba59-5b67-4580-9e06-ddf1314bbfa4" /></td>
        <td><img width="250"  alt="image" src="https://github.com/user-attachments/assets/8d8ce4c2-744a-4aff-8ad3-2d3aeca822a7" /></td>
        <td><img width="500"  alt="image" src="https://github.com/user-attachments/assets/d8192c53-3b1b-44f0-8a0a-26eb8e4c8204" /></td>
      </tr>
    </tbody>
  </table>
  
- Ramesh, Aditya, et al. "**Hierarchical text-conditional image generation with clip latents.**" arXiv preprint arXiv:2204.06125 1.2 (2022): 3. [paper](https://arxiv.org/pdf/2204.06125)
  ```
  This paper introduces a two-stage model for generating images. The first stage generates
  CLIP image embeddings, the second stage generates an image from these embeddings.
  ```
  
  <table>
    <tbody>
      <tr>
        <td> <img width="500" alt="image" src="https://github.com/user-attachments/assets/05b2d7d5-35a8-4749-865d-779c709b3c35" /></td>
        <td><img width="350"  alt="image" src="https://github.com/user-attachments/assets/8851777b-bceb-4bc0-93ae-01fd71e7b8ea" /></td>
      </tr>
    </tbody>
  </table>

## Optimization

### Flash Attention

- Dao, Tri, et al. "**Flashattention: Fast and memory-efficient exact attention with io-awareness.**" Advances in neural information processing systems 35 (2022): 16344-16359. [paper](https://papers.nips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf)
  ```
  This paper introduces FlashAttention, an algorithm that reduces the
  number of memory read/write operations between GPU HBM and SRAM.
  ```
  <table>
    <tbody>
      <tr>
        <td> <img width="500" alt="image" src="https://github.com/user-attachments/assets/f53b9a98-0443-4283-9054-6d6754f9b76a" /></td>
        <td><img width="500"  alt="image" src="https://github.com/user-attachments/assets/66f0b74e-1805-4137-aa08-3665cccab9e2" /></td>
      </tr>
    </tbody>
  </table>
  
## AI&ML Engineering

- Markov, Igor L., et al. "**Looper: An end-to-end ML platform for product decisions.**" Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2022. [paper](https://arxiv.org/pdf/2110.07554)
  ```
  This paper introduces Looper, an end-to-end ML platform for
  real-time smart decision making.
  ```
  <table>
    <tbody>
      <tr>
        <td> <img width="500" alt="image" src="https://github.com/user-attachments/assets/08a67cde-eab4-4c2c-8f87-ac00c14f0b5d" /></td>
        <td><img width="500"  alt="image" src="https://github.com/user-attachments/assets/c40399b8-abd9-4677-98cc-a1c434996d82" /></td>
      </tr>
    </tbody>
  </table>
