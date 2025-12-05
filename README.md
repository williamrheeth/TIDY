<div align="center">
  <p align="center">
  <img src="assets/logo.png" width="150">
</p>
  
<h1>
  <b>TIDY</b> :
  Thermal Infrared Image Denoising via <br>
  Wavelet Domain Entropy and Directional Stripe Index
</h1>


[**Tai Hyoung Rhee**](https://scholar.google.com/citations?user=PF8EfdYAAAAJ&hl=en&oi=ao)<sup>1</sup> · [**Dong-Guw Lee**](https://scholar.google.com/citations?user=u6VDnlgAAAAJ&hl=ko)<sup>1</sup> · [**Ayoung Kim**](https://ayoungk.github.io/)<sup>1&dagger;</sup>

<sup>1</sup>Seoul National University


**Under-Review**

<a href='https://williamrheeth.github.io/mirage_web/'><img src='https://img.shields.io/badge/TIDY-Project_Page-purple?logo=github' alt='Project Page'></a>
<a href="https://rpm.snu.ac.kr"><img src='https://img.shields.io/badge/TIDY-arXiv-red?logo=arxiv' alt='ArXiv Link'></a>
<a href="https://rpm.snu.ac.kr"><img src='https://img.shields.io/badge/TIDY-Weights-yellow?logo=huggingface' alt='Weights'></a>
<a href="https://rpm.snu.ac.kr"><img src='https://img.shields.io/badge/SCaN--TIR-Dataset-blue?logo=googledrive' alt='SCaN-TIR Dataset'></a>
</div>



## News
- ⚡(2025-12-03): TIDY repo opening

  
<hr />

> **Abstract:** *Thermal infrared (TIR) imaging enhances robotic perception in adverse conditions but is fundamentally degraded by severe structured and random noise. Existing denoising methods not only depend on self-supervision via synthetic noise resulting in poor generalization, but also employ computationally heavy models that are impractical for real-time robotics. We address both limitations with TIDY via explicitly exploiting the wavelet domain in a novel manner: by leveraging wavelet entropy with a novel directional stripe index into the loss function with wavelet transform as an encoder/decoder, it isolates stochastic noise and orientation-specific stripe artifacts, achieving 20–30% improvements in PSNR and SSIM compared to the strongest baselines, while reducing computational complexity by an order of magnitude. To enable supervised learning on real sensor noise, we further introduce SCaN-TIR, the first real stereo clean-noisy paired dataset. Extensive evaluations show that TIDY not only delivers state-of-the-art denoising, even in zero-shot conditions, but also substantially enhances downstream robotics tasks such as thermal inertial odometry and monocular depth estimation, all while sustaining real-time efficiency. Both TIDY and the SCaN-TIR dataset will be publicly released..* 
<hr />

## TIDY & SCaN-TIR Overview


### Network Architecture

![representatives](assets/model.jpg)

### SCaN-TIR Dataset Overview Table

<details>
  <summary>Click to Expand</summary>

<table style="border-collapse: collapse; width: 80%;">
  <tr>
    <th style="border-bottom: 2px solid black;">Sequence</th>
    <th style="border-bottom: 2px solid black;"># Image Pairs</th>
    <th style="border-bottom: 2px solid black;">Duration</th>
    <th style="border-bottom: 2px solid black;">Scene</th>
  </tr>

  <tr><td><code>300_floor5</code></td><td>494</td><td>19.8s</td><td>Indoor</td></tr>
  <tr><td><code>300_to_303</code></td><td>4,698</td><td>187s</td><td>Outdoor</td></tr>
  <tr><td><code>301_118</code></td><td>4,329</td><td>173s</td><td>Indoor</td></tr>
  <tr><td><code>301_floor1</code></td><td>4,548</td><td>183s</td><td>Indoor</td></tr>
  <tr><td><code>301_floor12</code></td><td>5,973</td><td>241s</td><td>Indoor</td></tr>
  <tr><td><code>303_floor2</code></td><td>3,287</td><td>131s</td><td>Indoor</td></tr>
  <tr><td><code>303_floor5</code></td><td>3,045</td><td>121s</td><td>Indoor</td></tr>
  <tr><td><code>303_floor7</code></td><td>3,799</td><td>151s</td><td>Indoor</td></tr>
  <tr><td><code>303_bridge</code></td><td>2,217</td><td>88s</td><td>Outdoor</td></tr>
  <tr><td><code>test_300</code></td><td>132</td><td>5.3s</td><td>Indoor</td></tr>
  <tr><td><code>test_303</code></td><td>55</td><td>2.2s</td><td>Indoor</td></tr>

  <tr>
    <td><b>Total</b></td>
    <td><b>32,577</b></td>
    <td></td>
    <td></td>
  </tr>
</table>
</details>


### Zero-shot Results

<details>
  <summary>Click to Expand</summary>
![zeroshot](assets/results_zeroshot.jpg)
</details>

### Downstream Enhancement

- Thermal-Inertial Odometry based on VINS-Mono
> <details>
>   <summary>Click to Expand</summary>
> 
>   <img src="assets/results_from_vins_mono.jpg" />
> 
> </details>


- Monocular Depth Estimation based on Depth Anything V2
> <details>
>   <summary>Click to Expand</summary>
> ![results_depth](assets/results_depth.jpg)
> </details>

---

## Installation

Refer to the link below for download.

[Google Drive](https://rpm.snu.ac.kr/)

Docker.

---

## Dataset Structure

```
  SCaN-TIR
  ├── {$SEQUENCE_NAME}
  |   ├── Clean
  |   |   ├── 0.png
  |   |   └── ...
  |   └── Noisy
  |       ├── 0.png
  |       └── ...
  ├── ...
  ├── ViVID
  |   ├── img_campus_day1
  |   |   ├── RGB
  |   |   |   ├── 000001.png
  |   |   |   └── ...
  |   |   └── TIR
  |   |       ├── 000001.png
  |   |       └── ...
  |   ├── ...
  ├── ...
  ```

---

## Benchmark Evaluation

### State-of-the-Art Image Translation Model Evaluation

#### RGB-to-TIR ###
<details>
  <summary>Click to Expand</summary>

It's a gif file. Changes every 5 seconds
  
![rgb2tir](https://github.com/user-attachments/assets/c17ff001-8e35-43eb-be29-6d643d114854)

</details>

#### TIR-to-RGB ###

<details>
  <summary>Click to Expand</summary>

It's a gif file. Changes every 5 seconds
  
![tir2rgb](https://github.com/user-attachments/assets/ccfbfe0f-2bf8-4134-8183-b51197c03487)

</details>

### Zero-shot Performance of MIRAGE

<details>
  <summary>Click to Expand</summary>
  
Zero-shot performance of PID trained on MIRAGE in comparison to PID trained with other benchmarks
  
![zero_shot](https://github.com/user-attachments/assets/b99b2a94-0e03-431a-b583-f403c252e92d)

</details>


## TODO List

- [x] Open git repository
- [ ] Upload dataset
- [ ] Upload curation framework
- [ ] add more baselines


---

## Citation
If you found our work useful, please cite
```
@inproceedings{mirage2025,
title={MIRAGE: Large-Scale Aligned RGB-Thermal Infrared Dataset for Scalable Multispectral Translation},
  author={Lee, Dong-Guw and Jang, Hyunsoo and Rhee, TaiHyoung and Shin, UkCheol and Kim, Ayoung},
  booktitle={ArXiv},
  year={2025}
}

}
```
