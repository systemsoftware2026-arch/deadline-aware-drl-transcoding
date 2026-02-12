# deadline-aware-drl-transcoding

A deadline-aware transcoding scheduling pipeline for low-power edge clusters (e.g., Raspberry Pi), integrating Provisioning Ratio (PR)-based bitrate ladder generation and JSON-based job dispatch.

---

## Overview

This repository provides a three-stage pipeline:

1. Provisioning Ratio (PR) extraction  
2. Algorithm-specific bitrate ladder generation  
3. JSON-based transcoding dispatch on a Raspberry Pi cluster  

The system is designed to operate in a master–worker edge cluster architecture.

---

## Pipeline

### 1) Provisioning Ratio (PR) Extraction

`PR_Extraction_Module.py`

- Computes Provisioning Ratio (PR)
- Outputs: `PR_result.json`

This file contains provisioning-related information used to determine bitrate version selection under resource constraints.

---

### 2) Bitrate Ladder Generation

`Ladder_Generator.py`

- Input: `PR_result.json`
- Generates algorithm-specific bitrate ladder JSON files
- Produces dispatch-ready configuration files

---

### 3) Transcoding Dispatch (Master Node)

`dispatch_json.py` (Run on Raspberry Pi master node)

- Reads ladder/dispatch JSON
- Assigns transcoding tasks to worker nodes
- Executes FFmpeg-based transcoding remotely via SSH

---


## Execution Order

1. Run PR extraction
   ```bash
   python PR_Extraction_Module.py

2. Generate bitrate ladders
   python Ladder_Generator.py

3. Dispatch transcoding jobs (on master node)
   python dispatch_json.py .json


## Dataset

`transcoding_dataset_final.csv`

This dataset was generated for bitrate-wise video quality prediction using a Random Forest model.

- Each entry corresponds to a transcoding configuration.
- Includes bitrate-level quality metrics (e.g., VMAF/PSNR, if applicable).
- Used for training and validating the bitrate-specific quality predictor.

The dataset is utilized during provisioning and ladder generation stages.

