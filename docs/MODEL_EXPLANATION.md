# Neural Network Model Explanation: Pix2Pix & CycleGAN
**Project:** Cross-Sensor Image Translation (Optical-to-SAR)  
**Target:** ISRO / URSC Research Demonstration  

---

## 1. Pix2Pix Architecture (Conditional GAN)

Pix2Pix is the primary deep learning baseline implemented in this application for paired optical-to-SAR translation.

```
       [ Optical Image X (3x256x256) ]
                     │
                     ▼
         ┌───────────────────────┐
         │    U-Net Generator    │
         │ (Encoder-Decoder with │
         │   Skip Connections)   │
         └───────────────────────┘
                     │
                     ▼
       [ Synthetic SAR Y_hat (1x256x256) ]
                     │
                     ├───► L1 Reconstruction Loss with Real SAR Y
                     │
                     ▼
         ┌───────────────────────┐
         │ PatchGAN Discriminator│ ◄─── Concatenation [X, Y] or [X, Y_hat]
         │  (70x70 Patch Map)    │
         └───────────────────────┘
                     │
                     ▼
         Adversarial Loss (Real / Fake)
```

### 1.1 Generator: U-Net 256
* **Encoder:** 8 downsampling convolutional blocks (stride 2, kernel 4, padding 1).
  - Layer 1: $3 \to 64$ (LeakyReLU 0.2)
  - Layer 2: $64 \to 128$ (BatchNorm + LeakyReLU 0.2)
  - Layer 3: $128 \to 256$ (BatchNorm + LeakyReLU 0.2)
  - Layer 4: $256 \to 512$ (BatchNorm + LeakyReLU 0.2)
  - Layers 5–8: $512 \to 512$ (BatchNorm + LeakyReLU 0.2)
* **Decoder:** 8 upsampling transposed convolutional blocks (stride 2, kernel 4, padding 1).
  - Layers 1–3: $512 \to 512$ (BatchNorm + Dropout 0.5 + ReLU) with skip concatenation
  - Layers 4–7: Upsampling with skip concatenation
  - Layer 8: Final ConvTranspose2d with **Tanh** activation mapping synthetic output to $[-1.0, 1.0]$.

### 1.2 Discriminator: $70 \times 70$ PatchGAN
* Takes concatenated 4-channel input: Optical condition ($3$ channels) + SAR image ($1$ channel).
* Uses 4 convolutional downsampling layers with LeakyReLU ($0.2$) and BatchNorm.
* Output is a $30 \times 30$ patch grid where each neuron has an effective receptive field of $70 \times 70$ pixels on the input image.

---

## 2. Loss Formulations

1. **Adversarial Objective:**
   $$\mathcal{L}_{\text{GAN}}(G, D) = \mathbb{E}_{x, y}[\log D(x, y)] + \mathbb{E}_{x}[\log(1 - D(x, G(x)))]$$
2. **Reconstruction L1 Objective:**
   $$\mathcal{L}_{L1}(G) = \mathbb{E}_{x, y}[\|y - G(x)\|_1]$$
3. **Full Joint Objective:**
   $$\arg\min_G \max_D \mathcal{L}_{\text{cGAN}}(G, D) + \lambda_{L1} \mathcal{L}_{L1}(G)$$
   Default $\lambda_{L1} = 100.0$.

---

## 3. CycleGAN Research Extension

CycleGAN is provided as an optional research module (`src/models/cyclegan/`) for scenarios where perfectly registered pairs are unavailable.
* Uses two generators ($G: X \to Y$, $F: Y \to X$) and two discriminators ($D_X, D_Y$).
* Enforces **Cycle Consistency Loss**:
  $$\mathcal{L}_{\text{cyc}}(G, F) = \mathbb{E}_{x}[\|F(G(x)) - x\|_1] + \mathbb{E}_{y}[\|G(F(y)) - y\|_1]$$
* **Comparison:** Pix2Pix is preferred for paired benchmark data (SEN1-2) because direct paired ground truth constrains geographic feature alignment without hallucinating unobserved ground structures.
