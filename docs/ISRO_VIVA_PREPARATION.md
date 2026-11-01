# ISRO / URSC Technical Viva Preparation Guide
**Topic:** Cross-Sensor Image Translation (Optical-to-SAR)  
**Candidate Evaluation Level:** Advanced Undergraduate / Internship Defense  
**Focus Areas:** Synthetic Aperture Radar, Optical Remote Sensing, Deep Generative Models, PyTorch, Evaluation Metrics, and Satellite Systems.

---

### Category 1: Remote Sensing Principles (SAR & Optical)

#### Q1: What is the fundamental difference between passive optical and active microwave (SAR) remote sensing?
**Answer:** Passive optical sensors rely on external illumination (the Sun) and measure reflected radiation in the visible ($0.4-0.7\,\mu\text{m}$), NIR ($0.7-1.1\,\mu\text{m}$), and SWIR ($1.1-2.5\,\mu\text{m}$) bands. They cannot see through clouds or operate at night. In contrast, SAR is an active microwave instrument ($1-10\,\text{GHz}$) that emits its own coherent pulses and records backscattered signal amplitude and phase, providing day-and-night observation capabilities that penetrate clouds, rain, and haze.

#### Q2: How does a Synthetic Aperture Radar achieve high spatial resolution in the azimuth direction?
**Answer:** Azimuth resolution for a real aperture antenna is $\Delta\theta \approx \lambda / L$ (where $L$ is antenna length). From satellite altitudes ($700\,\text{km}$), achieving a $10\,\text{m}$ resolution would require an antenna kilometers long. SAR overcomes this limitation by utilizing the platform's forward orbital velocity: it transmits a series of coherent pulses and records Doppler frequency shifts. By coherently synthesizing these echoes over the synthetic aperture length, the achievable azimuth resolution becomes independent of slant range: $\delta_{\text{az}} \approx L / 2$.

#### Q3: What is radar backscatter ($\sigma^0$) and why is it measured in decibels (dB)?
**Answer:** Backscatter coefficient $\sigma^0$ (sigma-nought) is the radar cross section per unit illuminated surface area. Because ground backscatter spans several orders of magnitude—from $-40\,\text{dB}$ for smooth calm water to $+10\,\text{dB}$ for urban double-bounce reflectors—it is represented on a logarithmic decibel scale: $\sigma^0_{\text{dB}} = 10 \cdot \log_{10}(\sigma^0_{\text{linear}})$.

#### Q4: What is the difference between VV and VH polarizations in Sentinel-1?
**Answer:** In VV (co-polarization), the electric field is transmitted vertically and received vertically. It is strongly influenced by surface roughness and water-land dielectric boundaries. In VH (cross-polarization), the signal is transmitted vertically and received horizontally. Depolarization occurs due to multiple internal scatterings, making VH particularly sensitive to volume scattering within vegetation canopies and forest branches.

#### Q5: What is the speckle phenomenon in SAR?
**Answer:** Speckle is a granular, deterministic interference pattern that appears in coherent imaging systems. Within a single resolution cell, multiple sub-scatterers reflect waves that interfere constructively and destructively depending on sub-wavelength phase path differences. While speckle resembles noise, it carries information about micro-relief.

#### Q6: How does radar wavelength impact penetration?
**Answer:** Penetration depth increases with longer wavelengths. X-band ($\approx 3\,\text{cm}$) scatters primarily off top canopy leaves; C-band ($\approx 5.6\,\text{cm}$, e.g. Sentinel-1 / RISAT-1) penetrates through foliage into small branches; L-band ($\approx 24\,\text{cm}$, e.g. NASA-ISRO NISAR) penetrates deep tree canopies and dry sand down to ground trunks and sub-surface moisture.

---

### Category 2: Deep Learning & Image Translation (Pix2Pix & GANs)

#### Q7: What is Conditional Generative Adversarial Network (cGAN) translation?
**Answer:** Standard GANs generate random samples $G(z)$ from a noise distribution $z \sim \mathcal{N}(0, I)$. Conditional GANs condition the generation on an observed input $x$: $G(x)$. In Pix2Pix, the generator transforms an optical image $x$ into a synthetic SAR image $G(x)$, while the discriminator determines whether pair $[x, y]$ is real or fake.

#### Q8: Why does Pix2Pix use a U-Net rather than an autoencoder?
**Answer:** In optical-to-SAR translation, fine geometric features (coastlines, agricultural parcel boundaries, urban street grids) must remain spatially aligned. Standard autoencoders compress all spatial features through a low-dimensional bottleneck, which destroys high-frequency coordinates. The U-Net connects mirrored encoder and decoder layers via skip connections, allowing high-resolution spatial details to bypass the bottleneck directly.

#### Q9: What is the role of PatchGAN?
**Answer:** Instead of mapping the entire image to a single real/fake probability, PatchGAN classifies $70 \times 70$ pixel sub-patches. This restricts the discriminator's focus to local high-frequency textures (texture realism and speckle realism) while leaving global low-frequency spatial structure to be governed by the L1 reconstruction loss.

#### Q10: What is the objective function of Pix2Pix?
**Answer:**
$$\mathcal{L}_{\text{Pix2Pix}} = \arg\min_G \max_D \mathcal{L}_{\text{cGAN}}(G, D) + \lambda_{L1} \mathcal{L}_{L1}(G)$$
where $\mathcal{L}_{L1} = \mathbb{E}[\|Y - G(X)\|_1]$. We set $\lambda_{L1} = 100.0$ to ensure strong spatial correspondence to the input optical scene.

#### Q11: Why is L1 loss preferred over L2 loss for the generator?
**Answer:** Minimizing L2 loss ($\|Y - G(X)\|_2^2$) produces the conditional mean of all possible outputs, causing significant visual blurring in uncertain regions. L1 loss produces the conditional median, preserving sharper visual edges.

#### Q12: How does CycleGAN differ from Pix2Pix?
**Answer:** Pix2Pix requires strictly coregistered pairs $(X_i, Y_i)$ and trains with paired supervised L1 loss. CycleGAN is designed for unpaired domains; it uses two generators ($G: X \to Y$ and $F: Y \to X$) and enforces a cycle consistency constraint $F(G(X)) \approx X$ to prevent mode collapse without needing paired data.

#### Q13: Why did you prioritize Pix2Pix over CycleGAN for this project?
**Answer:** Because remote sensing cross-sensor translation for disaster response requires geographic fidelity. Using verified paired benchmarks (SEN1-2) with Pix2Pix ensures that predicted radar responses correspond directly to real ground coordinates without the risk of hallucinated structures common in unpaired CycleGAN translation.

---

### Category 3: Dataset Engineering & Preprocessing

#### Q14: What is the SEN1-2 dataset?
**Answer:** SEN1-2 (Schmitt et al., 2019) is a global benchmark consisting of 282,384 paired $256 \times 256$ patches acquired by Sentinel-1 (SAR) and Sentinel-2 (optical), resampled to $10\,\text{m}$ spatial resolution across all seasons and various climatic zones.

#### Q15: How did you verify the accessibility of the three datasets?
**Answer:** 
1. **SEN1-2:** Verified direct HTTP streaming and range requests via an open Hugging Face repository mirror (`Sul06/sen12mscr-replica`), allowing selective downloading without full archive bloat.
2. **QXS-SAROPT:** Probed repository and discovered that download links require manual application submission via Wenjuan.com.
3. **SAR2Opt:** Inspected repository and verified that data is stored in monolithic zip archives on Google Drive (blocked by virus-scan redirects) and BaiduDisk.

#### Q16: How do you normalize SAR backscatter in decibels?
**Answer:** Typical Sentinel-1 GRD backscatter in dB ranges from $-35\,\text{dB}$ to $0\,\text{dB}$. Values are clipped to $[-35, 0]$ and scaled to $[0, 1]$, then mapped to $[-1.0, 1.0]$ via $x_{\text{norm}} = (x_{01} \cdot 2.0) - 1.0$ to match the generator's Tanh activation function.

#### Q17: How is data leakage prevented between training and evaluation?
**Answer:** Patches are partitioned geographically by region (e.g., Azerbaijan scenes for training, Gabon for validation, Germany for testing), ensuring that spatially adjacent patches from the same satellite pass never leak across splits.

---

### Category 4: Evaluation Metrics & Interpretation

#### Q18: What is PSNR and how is it calculated?
**Answer:** Peak Signal-to-Noise Ratio compares the maximum possible power of a signal to the power of corrupting noise:
$$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right)$$
For images normalized in $[-1, 1]$, $\text{MAX} = 2.0$. Higher values indicate lower reconstruction error.

#### Q19: What is SSIM and what does it measure?
**Answer:** Structural Similarity Index Measure (SSIM) evaluates image similarity based on three comparative components: luminance $l(x, y)$, contrast $c(x, y)$, and structure $s(x, y)$. Values range between $-1$ and $+1$, with $+1$ indicating identical structural topology.

#### Q20: What are the limitations of PSNR and SSIM for SAR imagery?
**Answer:** PSNR and SSIM were designed for human visual perception of natural optical photographs. They reward smooth images and penalize high-frequency noise. Because real SAR images contain natural multiplicative speckle, a model that artificially blurs speckle may achieve a higher PSNR despite being less scientifically realistic.

#### Q21: What is a residual error map?
**Answer:** It is a pixel-by-pixel absolute difference matrix $|Y_{\text{reference}} - \hat{Y}_{\text{synthetic}}|$ visualized as a thermal heatmap to pinpoint geographic zones where the translation model struggled (e.g. high-reflectance urban corners).

---

### Category 5: ISRO Missions & Space Applications

#### Q22: Which ISRO satellites carry SAR payloads?
**Answer:** RISAT-1, RISAT-2, RISAT-2B, EOS-04 (RISAT-1A, launched in 2022 operating in C-band), and the upcoming joint NASA-ISRO NISAR mission (carrying dual L-band and S-band SAR).

#### Q23: Which ISRO satellites carry optical payloads?
**Answer:** Resourcesat-2/2A (LISS-3, LISS-4, AWiFS), Cartosat-2/3 series (sub-meter panchromatic and multispectral optical), and EOS-06 (Oceansat-3).

#### Q24: How does this project support ISRO's disaster response initiatives?
**Answer:** During the Indian monsoon, severe floods occur across Assam, Bihar, and Kerala while clouds obscure optical satellites. If clear historical optical maps exist, cross-sensor translation generates baseline synthetic SAR maps that aid automated floodwater thresholding on real post-disaster SAR passes.

#### Q25: Can this model replace real radar sensors?
**Answer:** Absolutely not. A neural network translates learned visual correlations into expected radar backscatter. It cannot physically penetrate soils, determine real sub-surface dielectric moisture, or detect hidden structural defects that only physical microwave radiation can illuminate.
