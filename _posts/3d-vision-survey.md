---
title: 'Literature Survey: Recent Advances in Generative 3D Vision'
date: 2024-08-14
permalink: /posts/2024/08/3d-vision-survey/
tags:
  - 3D Generation
  - Novel View Synthesis
  - Diffusion Models
  - Gaussian Splatting
  - Literature Survey
---

This post provides a survey of recent and seminal papers in the rapidly advancing field of generative 3D vision. We will explore key innovations across several domains, including novel view synthesis, 360° panorama generation, and full 3D world creation, highlighting the shift towards diffusion-based models and real-time interactive systems.

## 1. Novel View Synthesis (NVS)

Novel View Synthesis (NVS) aims to synthesize photorealistic images of a scene from new camera viewpoints, given one or more input images. The primary challenges lie in handling occlusions and maintaining strict 3D geometric consistency while generating novel content.

### GenWarp: Single Image to Novel Views with Semantic-Preserving Generative Warping (2024)
*   **Key Innovation**: Proposes a "Semantic-Preserving Generative Warping" framework (GenWarp) that unifies the traditional two-step "warping-and-inpainting" process into a single generative model. It teaches a diffusion model "where to warp" and "where to generate," addressing artifacts and semantic loss from inaccurate depth maps.
*   **Implementation**: Utilizes a dual-stream architecture (semantic-preserving network and diffusion model) sharing a U-Net structure. "Warped coordinate embeddings" are introduced as a condition to implicitly guide the diffusion model's geometric warping. Its core is an **enhanced self-attention mechanism** that concatenates cross-view attention from the source view with self-attention in the target view, allowing the model to automatically balance warping pixels versus generating new content.
*   **Pipeline**:
    1.  **Input**: A single source image, its estimated depth map, and the target camera pose.
    2.  **Warped Coordinate Generation**: The source view's standard 2D coordinate map is geometrically transformed based on depth and target pose to create a warped coordinate map for the target view.
    3.  **Dual-Stream Generation**: A semantic-preserving network encodes source image features. Guided by these features and the warped coordinate map, the diffusion model progressively denoises the target view using the enhanced self-attention mechanism.
*   **Evaluation**: Assessed on RealEstate10K and ScanNet datasets, it outperforms baselines like GeoGPT, Photometric-NVS, and SD-Inpainting methods in FID and PSNR metrics, especially for medium and long-range viewpoint changes.

### Generative Novel View Synthesis with 3D-Aware Diffusion Models (GeNVS) (CVPR 2023)
*   **Key Innovation**: First to integrate explicit 3D geometric priors (in the form of a 3D feature volume) into the backbone of a 2D diffusion model. This enables the model to generate diverse and geometrically consistent new views, even when facing ambiguities like occlusions.
*   **Implementation**: The core is a 3D-aware diffusion model that introduces a latent 3D feature field to capture the scene's geometric distribution.
*   **Pipeline**:
    1.  **Input**: One or more source images with corresponding camera poses, and a target camera pose.
    2.  **Feature Construction**: An encoder extracts features from each source image, which are then back-projected to construct one or more 3D feature volumes.
    3.  **Aggregation & Rendering**: Multiple feature volumes are aggregated (e.g., via average pooling) and then rendered into a 2D feature map for the target view using a small MLP decoder and volume rendering.
    4.  **Denoising Generation**: This feature map is concatenated with the noisy target view image and fed into a U-Net denoiser to produce the final output.
*   **Evaluation**: Achieved SOTA performance on multiple datasets including ShapeNet, Matterport3D, and CO3D. Its geometric consistency, verified via COLMAP point cloud reconstruction, far exceeds that of pure generative models lacking geometric priors.

### ZeroNVS: Zero-Shot 360-Degree View Synthesis from a Single Image (2024)
*   **Key Innovation**: Designed specifically for 360° view synthesis from a single in-the-wild image. It trains a unified diffusion prior on a large, diverse dataset by introducing a novel camera parameterization and normalization strategy. It also proposes "SDS anchoring" to address the lack of diversity in complex backgrounds when using standard Score Distillation Sampling (SDS).
*   **Implementation**: A two-stage approach: first, a 2D conditional diffusion model is trained, and then a 3D representation (NeRF) is distilled via SDS. Its "6DoF+1" camera conditioning includes relative pose and field-of-view, with a viewer-centric normalization to handle scale ambiguity. SDS anchoring pre-generates several "pseudo-ground-truth" images from different viewpoints using DDIM sampling to guide the SDS process, enhancing background diversity.
*   **Pipeline**:
    1.  **Training**: A 2D diffusion model is trained on a mixed-scene dataset, conditioned on the source image and the novel camera parameters.
    2.  **Inference (3D Distillation)**:
        a. (Optional) "Anchor" views are generated at key orientations using the DDIM sampler.
        b. A NeRF model is optimized using an SDS loss, guided by the input image and these anchors, to produce the final 3D scene representation.
*   **Evaluation**: Achieved SOTA LPIPS scores in a zero-shot setting on the DTU dataset, even outperforming methods fine-tuned on it.

### MVDiffusion++: A Dense High-resolution Multi-view Diffusion Model for Single or Sparse-view 3D Object Reconstruction (2024)
*   **Key Innovation**: Achieves the generation of **dense (32 views)**, **high-resolution (512x512)** multi-view image sequences from single or sparse input images, all **without camera pose information**. The core ideas are its "pose-free architecture" and "view dropout" training strategy.
*   **Implementation**: Leverages a latent space inpainting model split into a conditioning branch and a generation branch. 3D consistency is learned implicitly via a **global self-attention mechanism** across all view features, without any explicit camera poses or projection formulas. "View dropout" drastically reduces memory consumption by randomly dropping a majority of generated views during training, enabling high-resolution dense view generation.
*   **Pipeline**:
    1.  **Input**: 1 to 10 object images without pose information.
    2.  **Multi-branch Denoising**: A conditioning branch processes inputs, while a generation branch handles target views (initially blank). All branches share U-Net weights.
    3.  **Global Self-Attention**: At each U-Net layer, features from all views (conditional + generative) are concatenated and processed together in a single self-attention block to learn global 3D consistency.
    4.  **Output**: 32 high-resolution, multi-view consistent images suitable for downstream 3D reconstruction (e.g., using NeuS).
*   **Evaluation**: On the Google Scanned Objects (GSO) dataset, it achieved a Volumetric IoU 0.1552 higher than SyncDreamer for single-view reconstruction and a PSNR 8.19 higher than LEAP for sparse-view NVS, reaching SOTA levels.

#### NVS Summary & Comparison

| Method (Method)     | Core Innovation                                       | Key Technology                                   | Input                      | Output                | Year/Conference |
| :---------------- | :--------------------------------------------- | :----------------------------------------- | :------------------------ | :------------------ | :-------- |
| **GenWarp**       | Unified generative warping, resolving artifacts and semantic loss | Enhanced self-attention, warped coordinate embeddings | Single image + depth, target pose | Single new view image      | 2024      |
| **GeNVS**         | Integrates 3D feature volumes into 2D diffusion for diverse, consistent views | 3D latent feature field, volume rendering, conditional diffusion | 1-N images + poses, target pose | Single/sequence of new views | CVPR 2023 |
| **ZeroNVS**       | Zero-shot 360° scene generation, improving background diversity | Novel camera parameterization, SDS anchoring | Single in-the-wild image   | Complete 3D scene (NeRF) | 2024      |
| **MVDiffusion++** | Generates dense, high-res multi-views without pose input | Global self-attention, View Dropout training | 1-N pose-free images     | 32 dense high-res views    | 2024      |

## 2. Panorama/360° Image Generation

### MVDiffusion: Enabling Holistic Multi-view Image Generation with Correspondence-Aware Diffusion (NeurIPS 2023)
*   **Key Innovation**: Proposes a method for generating multi-view images in parallel to synthesize a panorama, rather than using traditional iterative inpainting. It enforces inter-view consistency with a **Correspondence-Aware Attention (CAA)** mechanism.
*   **Implementation**: A panorama is decomposed into 8 overlapping 90° perspective images. The model is a multi-branch U-Net where each branch processes one view but shares weights. After each U-Net block, a CAA module allows each view's feature map to perform cross-attention with its neighbors, enforcing local consistency.
*   **Pipeline**:
    1.  **Parallel Input**: Noisy latent variables for all 8 views are fed into the multi-branch U-Net in parallel.
    2.  **Correspondence-Aware Denoising**: At each network layer, information is exchanged between adjacent branches via the CAA module.
    3.  **Stitching**: After generating 8 consistent perspective views, they are stitched into a complete panorama.
*   **Evaluation**: Trained and evaluated on Matterport3D, it outperforms Text2Light and inpainting methods on FID, IS, and CS metrics, resolving the boundary discontinuity issues of traditional approaches.

### PanFusion: Taming Stable Diffusion for Text to 360° Panorama Image Generation (CVPR 2024)
*   **Key Innovation**: Introduces a novel **dual-branch diffusion model** that denoises in both the panoramic and perspective domains simultaneously. This combines the global layout guidance of the panoramic branch with the powerful generative capabilities and distortion mitigation of the perspective branch.
*   **Implementation**: The core is the **Equirectangular-Perspective Projection-Aware Attention (EPPA)** mechanism, which establishes a projection-aware correspondence between the two branches' feature maps. The panoramic branch's U-Net uses circular padding to ensure seamless left-right boundaries.
*   **Pipeline**:
    1.  **Dual-Branch Input**: A text prompt is fed to both branches. A noisy latent variable is sampled on the panorama and projected to multiple perspective views for joint initialization.
    2.  **Synergistic Denoising**: At each denoising step, information is exchanged and fused between branches via the EPPA module.
    3.  **Output**: The panoramic branch directly outputs the final panorama, requiring no stitching.
*   **Evaluation**: Evaluated on Matterport3D, its FAED (a panorama-specific FID metric) significantly outperforms Text2Light and MVDiffusion. It produces images with better global consistency and fewer artifacts.

### DiffPano: Scalable and Consistent Text to Panorama Generation with Spherical Epipolar-Aware Diffusion (NeurIPS 2024)
*   **Key Innovation**: Achieves scalable and consistent **multi-view panorama generation**, enabling transitions from one room to another. It is the first to derive and apply **spherical epipolar geometry constraints** suitable for panoramic images.
*   **Implementation**: Trained on a custom large-scale panoramic video-text dataset. The model includes a single-view text-to-panorama model (fine-tuned from Stable Diffusion with LoRA) and a **spherical epipolar-aware** multi-view diffusion model. This multi-view model uses an attention module to ensure 3D consistency across a series of panoramas generated at different camera poses.
*   **Pipeline**:
    1.  **Dataset Construction**: A large-scale dataset of continuous panoramic keyframes, depths, poses, and text descriptions is generated using the Habitat simulator.
    2.  **Single-View Model**: Stable Diffusion is fine-tuned with LoRA to create a high-quality single text-to-panorama model (Pano-SD).
    3.  **Multi-View Model**: Building on Pano-SD, a spherical epipolar-aware attention module is added and trained on the dataset to generate view-consistent panoramic sequences.
*   **Evaluation**: Compared to a modified MVDream, it shows superior performance on FID, IS, and CS metrics. Critically, it demonstrates a unique "room-switching" capability, generating coherent transitions between scenes.

#### Panorama Generation Summary & Comparison

| Method (Method)   | Core Innovation                             | Key Technology                       | Input          | Output                     | Year/Conference    |
| :-------------- | :----------------------------------- | :----------------------------- | :------------ | :----------------------- | :----------- |
| **MVDiffusion** | Parallel multi-view generation for local consistency | Correspondence-Aware Attention (CAA) | Text          | 8 perspective views (stitched) | NeurIPS 2023 |
| **PanFusion**   | Synergistic panoramic-perspective generation for global consistency | Equirectangular-Perspective Projection-Aware Attention (EPPA) | Text/Layout     | Single complete panorama | CVPR 2024    |
| **DiffPano**    | Scalable, multi-view panorama generation (scene switching) | Spherical Epipolar-Aware Attention | Text + Camera Pose | Multi-view/sequence of panoramas | NeurIPS 2024 |

The evolution of panorama generation technology clearly shows a progression towards better handling of its unique geometry. **MVDiffusion** was a pioneering attempt, solving the basic problem of circular consistency. **PanFusion** offered a more elegant solution with its dual-branch architecture, improving both quality and consistency. Finally, **DiffPano** elevated the task from generating a single image to a navigable panoramic sequence by integrating physical 3D geometry constraints into the generative model.

## 3. 3D World/Scene Generation

### LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes (CVPR 2024)
*   **Key Innovation**: The first **domain-free** pipeline for generating general 3D scenes. It expands a single RGB(D) image or text prompt into a complete, navigable 3D Gaussian scene by alternating between "Dreaming" and "Alignment" steps, overcoming the limitations of domain-specific training data.
*   **Implementation**:
    *   **Dreaming**: Moves along a predefined camera trajectory, projects the current point cloud to a new view to get an incomplete image, uses Stable Diffusion's inpainting to complete it, and then "lifts" the new pixels to 3D points using monocular depth estimation.
    *   **Alignment**: A differentiable optimizer adjusts the depth of new points to smoothly stitch new and old point clouds together.
    *   **Gaussian Splatting**: The final point cloud initializes a 3D-GS model, which fills holes and outputs a high-fidelity scene.
*   **Evaluation**: Significantly outperforms RGBD2 across all metrics (CLIP-Score, CLIP-IQA), demonstrating superior content alignment, visual quality, and clarity even on domains RGBD2 was trained on.

### DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting (CVPR 2024)
*   **Key Innovation**: The first framework for **unconstrained 360° text-to-3D scene** generation. It uses a text-to-panorama model to create a globally consistent 2D panorama, then lifts it in one go to a 3D Gaussian scene, avoiding the repetition and distortion of piecemeal inpainting.
*   **Implementation**:
    *   **Text → Panorama**: Uses a combination of diffusion models and GPT-4V self-refinement to generate a high-quality panorama.
    *   **Panorama → 3D**: The panorama is projected into 20 perspective views with estimated depth. A learnable global geometry field aligns these depths into a dense point cloud, which then initializes a 3D-GS model.
    *   **Virtual View Regularization**: Uses **semantic loss** (DINOv2) and **geometric loss** (DPT depth covariance) on synthesized virtual views to guide the optimization of occluded regions.
*   **Evaluation**: Outperforms LucidDreamer on text-image alignment, perceptual quality, and realism metrics (CLIP Distance, NIQE, BRISQUE), providing complete 360° coverage with globally consistent semantics and style.

### WonderWorld: Interactive 3D Scene Generation from a Single Image (2025)
*   **Key Innovation**: The first to achieve **interactive, low-latency** 3D scene generation. A user can start from a single image and control the generation of new scene parts in real-time by moving a camera and providing text prompts, seeing results in under **10 seconds**.
*   **Implementation**: The core is a novel scene representation called **FLAGS (Fast LAyered Gaussian Surfels)** and a rapid generation algorithm. FLAGS divides the scene into layers (foreground, background, sky), represented by Gaussian surfels. A **guided depth diffusion** module ensures new content connects smoothly with the existing scene.
*   **Evaluation**: Speed is a key metric: a new scene chunk is generated in **9.5 seconds**. It surpasses all baselines in quantitative metrics (CLIP Score, Consistency, IQA) and won an overwhelming **~98%** preference rate in user studies against other SOTA methods.

### WonderTurbo: Generating Interactive 3D World in 0.72 Seconds (arXiv 2025)
*   **Key Innovation**: Achieves truly **real-time** interactive 3D scene generation, reducing the generation time per interaction to just **0.72 seconds**—a 15x speedup over WonderWorld. This is accomplished via a new framework of three highly efficient modules for geometry, appearance, and depth.
*   **Implementation**:
    *   **Geometry (StepSplat)**: A **feed-forward** 3D Gaussian generation method that uses a feature memory module and a Depth Guided Cost Volume for fast, accurate geometry creation.
    *   **Appearance (FastPaint)**: A custom **2-step diffusion model** for real-time inpainting/outpainting.
    *   **Depth (QuickDepth)**: A lightweight network for fast depth completion of newly generated appearance.
*   **Evaluation**: Achieves a revolutionary speed of **0.72s** per generation. Despite the speed, it surpasses all online and offline baselines in quantitative quality metrics and achieved a **69.43%** win rate against WonderWorld in user studies.

### HunyuanWorld 1.0: Generating Immersive, Explorable, and Interactive 3D Worlds from Words or Pixels (2025)
*   **Key Innovation**: A comprehensive framework that combines 2D panorama generation and 3D reconstruction to create worlds with **semantic layering**, **exportable standard meshes**, and **interactive, separable objects**.
*   **Implementation**: Uses an "Agentic World Layering" module to automatically decompose a high-quality panorama into semantic layers (sky, background, foreground objects). Each layer is then independently lifted to 3D and reconstructed as a mesh.
*   **Evaluation**: Achieved SOTA results in both panorama generation and 3D world generation tasks. Its layered mesh output provides a new level of interactivity and compatibility with traditional graphics pipelines.

#### 3D World Generation Summary & Comparison

| Method (Method)     | Core Innovation                   | Key Technology                                  | Scalability/Interactivity          | Output Format         | Year/Conference |
| :---------------- | :------------------------- | :---------------------------------------- | :----------------------- | :--------------- | :-------- |
| **LucidDreamer**  | General domain-free 3D scene generation | Alternating Dreaming + Alignment; 3D-GS optimization | Single text/image input | 3D Gaussian Scene      | 2023      |
| **DreamScene360** | Unconstrained text-to-360° 3D scene | GPT-4V self-refinement; Panoramic geometry field | Any text prompt           | 360° 3D Gaussian Scene | 2024      |
| **WonderWorld**   | Real-time interactive scene generation | FLAGS representation, guided depth diffusion | Highly interactive, real-time | FLAGS (Gaussian Surfels) | 2025      |
| **WonderTurbo**   | Real-time (<1s) interactive world generation | StepSplat + FastPaint + QuickDepth modular pipeline | Extremely interactive, real-time | 3D Gaussian Scene      | 2025      |
| **HunyuanWorld**  | Semantic, layered, interactive 3D world | Agentic World Layering, layered 3D reconstruction | Scalable, object-level interaction | Layered 3D Mesh        | 2025      |

## 4. Other Foundational Works

*   **MVDREAM: Multi-view Diffusion for 3D Generation (2024)**
    *   **Summary**: A foundational model focused on generating multi-view consistent images of **objects** from text. By "inflating" 2D self-attention to 3D, it enforces consistency across views. While not a direct 3D generator, its output serves as a powerful prior for SDS-based 3D optimization (e.g., NeRF), significantly mitigating the multi-faced Janus problem.

*   **InFusion: Inpainting 3D Gaussians via Learning Depth Completion from Diffusion Prior (2024)**
    *   **Summary**: A "depth-first" paradigm for 3D inpainting. It uses a conditional diffusion model to accurately complete the depth map of a missing region first. This provides precise 3D locations for new points, simplifying the complex 3D inpainting task into a brief fine-tuning process. It is over **20x faster** than baselines with SOTA quality.