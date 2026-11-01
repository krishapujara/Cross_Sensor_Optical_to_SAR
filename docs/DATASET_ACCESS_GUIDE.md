# Dataset Access & Selective Retrieval Guide
**Project:** Cross-Sensor Image Translation: Optical-to-SAR  
**Target:** ISRO / URSC Research Demonstration  

---

## 1. Remote Access Architecture

Rather than forcing the user to download monolithic hundreds-of-gigabytes dataset archives, this platform implements a modular **selective retrieval architecture**:
* Queries remote repositories via direct HTTP range requests.
* Retrieves only the user-specified count of paired patches (e.g. 10, 25, 50, 100).
* Validates each pair for radiometric integrity, channel count, and dimensions before caching.
* Indexes all downloaded samples in `data/cache/{dataset_name}/metadata_index.json`.

---

## 2. Dataset Specific Access Procedures

### 2.1 Dataset A: SEN1-2 (Verified Primary Benchmark)
* **Status:** `VERIFIED ACCESSIBLE`
* **Access Mechanism:** Direct HTTPS queries to verified Hugging Face mirror endpoints.
* **URLs:**
  - Sentinel-1 SAR: `https://huggingface.co/datasets/Sul06/sen12mscr-replica/resolve/main/{region}/s1/patch_{id:04d}.tif`
  - Sentinel-2 Optical: `https://huggingface.co/datasets/Sul06/sen12mscr-replica/resolve/main/{region}/s2_cloudfree/patch_{id:04d}.tif`
* **Credentials:** No private tokens or authentication required.
* **Geographic Coverage in Verified Mirror:** Azerbaijan, Gabon, Germany.

### 2.2 Dataset B: QXS-SAROPT (Authentication Required)
* **Status:** `AUTHENTICATION REQUIRED`
* **Official Repository:** `https://github.com/yaoxu008/QXS-SAROPT`
* **Access Barrier:** Author requires submitting an academic questionnaire at `https://www.wenjuan.com/s/UZBZJv5GwL/` to obtain a download link.
* **How to Ingest Local Data:**
  1. Once access is granted by the authors, download the image pairs.
  2. Place them into:
     `Cross_Sensor_Optical_to_SAR/data/cache/qxs_saropt/local_pairs/`
  3. The `QXSSAROPTDatasetAdapter.ingest_local_folder()` function will automatically detect, validate, and index all pairs.

### 2.3 Dataset C: SAR2Opt (Heterogeneous Benchmark)
* **Status:** `PARTIALLY ACCESSIBLE`
* **Official Repository:** `https://github.com/MarsZhaoYT/SAR2Opt-Heterogeneous-Dataset`
* **Access Barrier:** Hosted on Google Drive and BaiduDisk as large monolithic zip files.
* **How to Ingest Local Data:**
  1. Download the archive from the repository's Google Drive link.
  2. Extract $600 \times 600$ pairs into:
     `Cross_Sensor_Optical_to_SAR/data/cache/sar2opt/local_pairs/`
  3. The adapter will automatically crop/resize patches to $256 \times 256$ and index them.

---

## 3. Cache Management

* **Cache Directory:** `data/cache/`
* **Cache Inspection:** Available directly on Page 2 (`Dataset Manager`) under the **Local Cache Storage Manager** tab.
* **Cache Cleaning:** Users can safely purge cached data on demand with confirmation dialogs.
