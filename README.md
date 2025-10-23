# Lane Marking Labeling and Segmentation

This repository contains Python scripts for lane marking labeling and segmentation in image sequences. The scripts facilitate manual labeling of lane points and automate the labeling process for subsequent frames. Additionally, segmentation masks are generated from the labeled points for training purposes.

## Files

### 1. `hybrid_labeling.py` (Recommended)
This script implements the **Hybrid Interactive and Automated Labeling Method** compatible with specialized neural network architectures such as CLRerNet. It combines manual point annotations with automated propagation across sequential image frames to significantly reduce manual effort.

#### **Key Features:**
- Implements Algorithm 1 from the research paper
- Manual annotation of critical lane points on selected frames (every σ frames)
- Automated propagation of annotations to intermediate frames using directional translation
- Generates binary segmentation masks compatible with CLRerNet architecture
- Significantly reduces manual effort compared to CDLEM and ALINA methods
- Higher annotation consistency and scalability

#### **Usage:**
```bash
python hybrid_labeling.py
```

Modify the configuration in the `main()` function:
- `input_dir`: Directory containing image sequence
- `output_dir`: Directory for annotation output
- `step_size`: Frames between manual annotations (σ parameter)
- `adjustment_factor`: Scaling factor for directional translation

#### **Instructions:**
- **Left click**: Select lane points
- **'k' key**: Save current lane and start a new one
- **Arrow keys**: Set direction (Left, Right, Up, Down, default: Straight)
- **'r' key**: Redo current frame
- **Enter**: Finish selection and proceed

#### **Algorithm Overview:**
The hybrid method processes image sequences in two phases:
1. **Interactive Labeling**: Manual annotation every σ frames with directional translation to intermediate frames
2. **Mask Generation**: Automatic creation of binary segmentation masks from all annotations

---

### 2. `manual_labeling.py` (Legacy)
This script allows for manual annotation of lane markings on images and automates the labeling process for subsequent frames. The user manually selects lane points, and the script generates `.lines.txt` files containing the labeled points. It also enables automatic adjustment of lane points for consecutive frames based on user-defined directions.

#### **Key Features:**
- Allows manual selection of lane points on an image.
- Supports setting lane directions (left, right, up, down, straight) using arrow keys.
- Saves labeled points in `.lines.txt` files.
- Automates lane point adjustments for subsequent frames.
- Displays annotated frames for visualization.

#### **Usage:**
Modify the `input_dir` and `output_dir` variables to match your dataset structure. Run:
```bash
python manual_labeling.py
```

#### **Instructions for Use:**
- **Mouse Left-click**: Select a lane point.
- **'k' key**: Save the current lane and start a new one.
- **Arrow keys**: Set direction (Left, Right, Up, Down, default is Straight).
- **'Enter' key**: Finish selection and proceed.

---

### 3. `make_seg.py` (Legacy)
This script generates segmentation masks from lane markings stored in `.lines.txt` files. It reads image paths from `train.txt`, processes each image, and creates segmentation masks by connecting lane points with lines.

#### **Key Features:**
- Reads labeled points from `.lines.txt` files.
- Generates binary lane segmentation masks.
- Saves masks in a new directory (`laneseg_label_w16/images`).
- Updates the training file with new image-mask paths.

#### **Usage:**
Modify the `base_dir` variable to your dataset location. Run:
```bash
python make_seg.py
```

#### **Output:**
- The masks are saved as `.png` files in `laneseg_label_w16/images`.
- The updated `x_train.txt` file contains new image-mask paths.

---

## Setup
Ensure you have the required dependencies installed:
```bash
pip install opencv-python numpy
```

## Hybrid Labeling Algorithm

The hybrid labeling method implements the following mathematical procedure:

**Algorithm 1: Hybrid Interactive and Automated Labeling Method**

**Input:** Image sequence I = {I₁, I₂, ..., Iₙ}  
**Output:** Lane annotations L and segmentation masks M compatible with CLRerNet

1. Initialize frame index i ← 1, step size σ
2. While i ≤ N:
   - Manually select lane points Pᵢ = {(xⱼ, yⱼ)} on image Iᵢ
   - Define lane direction δ ∈ {left, right, up, down, straight}
   - Save points Pᵢ and direction δ to annotation file
   - For k = i+1 to min(i + σ - 1, N):
     - Adjust points using directional translation: Pₖ = Pᵢ + (k - i) · γ(δ)
     - Propagate adjusted points Pₖ to frame Iₖ
     - Save annotations for frame Iₖ
   - i ← i + σ
3. For all annotated frames Iᵢ:
   - Generate segmentation masks Mᵢ by interpolating lane points
   - Mᵢ(x,y) = 1 if (x,y) in lane segment, 0 otherwise
   - Save segmentation masks Mᵢ

**Key Advantages:**
- **Reduced Manual Effort**: Only annotate every σ frames instead of every frame
- **Consistency**: Automated propagation ensures consistent annotations
- **Compatibility**: Output format optimized for CLRerNet architecture
- **Scalability**: Efficiently handles large datasets with diverse conditions

## Project Structure
```
repo/
│── hybrid_labeling.py   # Hybrid labeling algorithm (Recommended)
│── manual_labeling.py   # Legacy manual labeling script
│── make_seg.py          # Legacy segmentation mask generation script
│── validate_labeling    # Validation script for labeled data
│── datasets/
│   ├── assisttaxi2/
│   │   ├── train/
│   │   │   ├── images/  # Original images
│   │   │   ├── train.txt  # List of training images
│   │   │   ├── laneseg_label_w16/  # Output segmentation masks
│   │   ├── valid/
│   │   │   ├── images2/  # Validation images
│   │   │   ├── processed_images/  # Processed images with annotations
│   │   │   ├── segmentation_masks/  # Generated segmentation masks
```

## Notes
- Ensure images are stored in the correct directories before running the scripts.
- Adjust `step_size` and `adjustment_factor` in `manual_labeling.py` for optimal automation.
