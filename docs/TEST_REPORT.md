# Software Test Report: Automated Test Outcomes
**Project:** Cross-Sensor Image Translation (Optical-to-SAR)  
**Target:** ISRO / URSC Research Demonstration  
**Test Framework:** `pytest 8.3.4`  
**Execution Environment:** Windows (Python 3.13.5, PyTorch 2.15.0+cpu)  

---

## 1. Test Execution Summary

* **Total Test Cases:** 25
* **Passed:** 25
* **Failed:** 0
* **Execution Duration:** $13.42\,\text{s}$
* **Success Rate:** **100%**

---

## 2. Detailed Test Results by Subsystem

| Test File | Test Case Name | Result | Description |
| :--- | :--- | :--- | :--- |
| `test_data_validation.py` | `test_validate_missing_file` | **PASSED** | Validates graceful handling of missing files |
| `test_data_validation.py` | `test_validate_valid_tiff` | **PASSED** | Validates GeoTIFF parsing and channel validation |
| `test_data_validation.py` | `test_pair_correspondence_mismatch` | **PASSED** | Detects and flags dimension mismatches between pairs |
| `test_dataset_adapters.py` | `test_dataset_registry_listing` | **PASSED** | Confirms registration of SEN1-2, QXS-SAROPT, SAR2Opt |
| `test_dataset_adapters.py` | `test_sen12_metadata` | **PASSED** | Confirms Sentinel-1, Sentinel-2, and CC BY 4.0 metadata |
| `test_dataset_adapters.py` | `test_qxs_saropt_authentication_requirement` | **PASSED** | Confirms honest reporting of Wenjuan questionnaire requirement |
| `test_dataset_adapters.py` | `test_sar2opt_partial_accessibility` | **PASSED** | Confirms honest reporting of Google Drive monolithic link |
| `test_inference.py` | `test_output_processor` | **PASSED** | Tests visual PNG and calibrated GeoTIFF byte exports |
| `test_inference.py` | `test_predictor_forward_pass` | **PASSED** | Tests single optical image inference with checkpoint |
| `test_metrics.py` | `test_identical_images_metrics` | **PASSED** | Identical images produce MSE=0, MAE=0, SSIM=1, PSNR>99 |
| `test_metrics.py` | `test_different_images_metrics` | **PASSED** | Different images produce valid finite non-zero metrics |
| `test_metrics.py` | `test_difference_map` | **PASSED** | Residual difference heatmap calculation |
| `test_models.py` | `test_unet_generator_shape` | **PASSED** | U-Net 256 forward pass produces (B, 1, 256, 256) in [-1, 1] |
| `test_models.py` | `test_patchgan_discriminator_shape` | **PASSED** | PatchGAN discriminator outputs spatial patch logits |
| `test_models.py` | `test_cyclegan_models_shape` | **PASSED** | CycleGAN ResNet generator and discriminator test |
| `test_models.py` | `test_build_pix2pix_models` | **PASSED** | Factory builder and Gaussian weight initialization |
| `test_preprocessing.py` | `test_optical_normalization_roundtrip` | **PASSED** | Optical normalization roundtrip within tolerance |
| `test_preprocessing.py` | `test_sar_db_normalization` | **PASSED** | SAR dB clipping [-35, 0] mapped linearly to [-1, 1] |
| `test_preprocessing.py` | `test_sar_to_display_image` | **PASSED** | Conversion to 8-bit visual array in [0, 255] |
| `test_preprocessing.py` | `test_split_dataset` | **PASSED** | Dataset partitioning without leakage |
| `test_streamlit_app.py` | `test_page_files_exist` | **PASSED** | Confirms existence of all 8 application pages |
| `test_streamlit_app.py` | `test_main_app_exists` | **PASSED** | Confirms main `app.py` entry point |
| `test_training.py` | `test_gan_loss_computation` | **PASSED** | Tests BCE/LSGAN adversarial loss computation |
| `test_training.py` | `test_pix2pix_loss_computer` | **PASSED** | Combined Generator (GAN + L1) and Discriminator loss |
| `test_training.py` | `test_checkpoint_save_and_load` | **PASSED** | Model weight serialization and metadata preservation |
