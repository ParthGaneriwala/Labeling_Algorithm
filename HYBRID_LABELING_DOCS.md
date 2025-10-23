# Hybrid Labeling Algorithm Documentation

## Overview

The Hybrid Interactive and Automated Labeling Method is a novel approach for lane marking annotation that combines manual point selection with automated propagation across sequential image frames. This method was specifically designed to be compatible with specialized neural network architectures such as CLRerNet.

## Motivation

### Limitations of Previous Methods

#### CDLEM (Contour-based Lane Marking Detection)
- **Manual Contour Adjustments**: Required frequent manual adjustments with scene variations
- **High Manual Overhead**: Every significant change in the scene required manual intervention
- **Inconsistency**: Manual adjustments could lead to inconsistent annotations across frames

#### ALINA (Automated Lane Identification and Annotation)
- **Initial ROI Definition**: Required manual definition of region of interest
- **Color Normalization Dependency**: While it improved consistency through color normalization, it still had limitations
- **Manual Overhead**: Initial setup for each sequence remained time-consuming

### Hybrid Method Advantages

The hybrid labeling method addresses these limitations by:

1. **Reduced Manual Effort**: Manual annotation required only every σ frames (default: 60 frames)
2. **Automated Propagation**: Intermediate frames are automatically annotated using directional translation
3. **Higher Consistency**: Mathematical propagation ensures consistent annotations across frames
4. **Scalability**: Efficiently handles large datasets with diverse conditions
5. **CLRerNet Compatibility**: Output format optimized for training segmentation models

## Algorithm Description

### Mathematical Formulation

The algorithm implements the following procedure:

**Input:** Image sequence I = {I₁, I₂, ..., Iₙ}  
**Output:** Lane annotations L and segmentation masks M

**Parameters:**
- σ: Step size (frames between manual annotations)
- γ(δ): Directional translation function
- δ: Lane direction ∈ {left, right, up, down, straight}

**Procedure:**

1. Initialize frame index i ← 1
2. While i ≤ N:
   a. Manually select lane points Pᵢ = {(xⱼ, yⱼ)} on image Iᵢ
   b. Define lane direction δ for the lane
   c. Save points Pᵢ and direction δ
   d. For each frame k from i+1 to min(i + σ - 1, N):
      - Compute adjusted points: **Pₖ = Pᵢ + (k - i) · γ(δ)**
      - Save annotations for frame Iₖ
   e. Increment: i ← i + σ
3. For all annotated frames:
   a. Generate segmentation mask Mᵢ by interpolating lane points
   b. Set Mᵢ(x,y) = 1 if (x,y) in lane segment, 0 otherwise

### Directional Translation Function γ(δ)

The function γ(δ) maps lane directions to translation vectors:

```python
γ('left')     = (-α, 0)   # Move left by α pixels per frame
γ('right')    = (α, 0)    # Move right by α pixels per frame
γ('up')       = (0, -α)   # Move up by α pixels per frame
γ('down')     = (0, α)    # Move down by α pixels per frame
γ('straight') = (0, 0)    # No movement
```

Where α is the adjustment factor (default: 1.5).

## Implementation Details

### Class: HybridLabeler

The `HybridLabeler` class provides a complete implementation of the algorithm.

#### Key Methods

1. **`gamma(direction)`**: Implements the directional translation function
2. **`adjust_points(points, k, i, direction)`**: Adjusts points for frame k
3. **`manual_annotation(image, output_file)`**: Interactive point selection
4. **`propagate_annotations(image, i, k, output_file)`**: Automated propagation
5. **`process_image_sequence(input_dir, output_dir)`**: Main processing loop
6. **`generate_segmentation_masks(...)`**: Creates binary masks

### Usage Example

```python
from hybrid_labeling import HybridLabeler

# Initialize labeler
labeler = HybridLabeler(
    step_size=60,           # Annotate every 60th frame
    adjustment_factor=1.5   # Translation scaling
)

# Phase 1: Process image sequence
labeler.process_image_sequence(
    input_dir='path/to/images',
    output_dir='path/to/annotations'
)

# Phase 2: Generate segmentation masks
labeler.generate_segmentation_masks(
    base_dir='path/to/base',
    annotations_dir='path/to/annotations',
    output_dir='path/to/masks'
)
```

## Interactive Controls

During manual annotation:

| Input | Action |
|-------|--------|
| Left Click | Select a lane point |
| 'k' key | Save current lane and start new one |
| Left Arrow | Set direction to 'left' |
| Right Arrow | Set direction to 'right' |
| Up Arrow | Set direction to 'up' |
| Down Arrow | Set direction to 'down' |
| 'r' key | Redo current frame |
| Enter | Finish selection and proceed |

**Default Direction**: 'straight' (no translation)

## Output Format

### Annotation Files (.lines.txt)

Each line in a `.lines.txt` file represents one lane marking:
```
x1 y1 x2 y2 x3 y3 ... xn yn
```

Example:
```
100.00 200.00 150.00 210.00 200.00 220.00
250.00 200.00 300.00 205.00 350.00 210.00
```

### Segmentation Masks (.png)

Binary images where:
- Lane pixels have value > 0 (specific to lane ID)
- Background pixels have value 0
- Each lane gets a unique ID for multi-lane scenarios

### Output List File

Format: `<image_path> <mask_path>`

Example:
```
images/frame_0001.jpg segmentation_masks/images/frame_0001.png
images/frame_0002.jpg segmentation_masks/images/frame_0002.png
```

## Performance Characteristics

### Manual Effort Reduction

With step size σ = 60:
- Traditional methods: 100% of frames require manual annotation
- Hybrid method: ~1.67% of frames require manual annotation
- **Reduction: 98.33%**

### Consistency

- Automated propagation ensures mathematical consistency
- No subjective variations between similar frames
- Predictable and reproducible results

### Scalability

- Linear time complexity: O(N) where N = number of frames
- Manual effort: O(N/σ) where σ is step size
- Suitable for large-scale datasets (thousands of frames)

## CLRerNet Compatibility

The output format is specifically designed for compatibility with CLRerNet:

1. **Binary Segmentation Masks**: CLRerNet expects binary masks separating lanes from background
2. **Multi-lane Support**: Each lane is assigned a unique ID
3. **Interpolated Segments**: Lane points are connected to form continuous segments
4. **Standard Format**: PNG masks with matching image dimensions

## Comparison with Previous Methods

| Feature | CDLEM | ALINA | Hybrid Method |
|---------|-------|-------|---------------|
| Manual Effort | High | Medium | Low |
| Consistency | Low | Medium | High |
| Scene Adaptability | Manual | Semi-Auto | Automated |
| Scalability | Limited | Medium | High |
| CLRerNet Ready | No | No | Yes |
| ROI Definition | Required | Required | Not Required |

## Best Practices

1. **Step Size Selection**:
   - Smaller σ (e.g., 30): Better for rapidly changing scenes
   - Larger σ (e.g., 100): Suitable for stable highway footage
   - Default σ = 60: Good balance for most scenarios

2. **Adjustment Factor**:
   - Depends on vehicle speed and frame rate
   - Higher speeds → larger adjustment factor
   - Start with default (1.5) and adjust based on results

3. **Direction Selection**:
   - Observe lane movement across frames
   - 'straight': For mostly parallel lanes with minimal perspective change
   - 'left'/'right': For curved roads or camera pan
   - 'up'/'down': For significant elevation changes

4. **Quality Control**:
   - Review automated frames periodically
   - If propagation drift occurs, reduce step size σ
   - Use validation script to check annotation quality

## Troubleshooting

### Issue: Annotations Drift Over Frames

**Solution**: Reduce step size σ or adjust the adjustment factor

### Issue: Points Outside Image Boundaries

**Solution**: The algorithm clips points to image boundaries automatically

### Issue: Inconsistent Lane Directions

**Solution**: Review direction selection; use 'straight' for ambiguous cases

## References

This implementation is based on the hybrid labeling methodology described in the research paper addressing lane marking annotation for CLRerNet architecture training.

## Future Enhancements

Potential improvements:
1. Automatic direction detection using optical flow
2. Adaptive step size based on scene complexity
3. Machine learning-based point adjustment
4. Real-time annotation preview
5. Multi-annotator collaboration support
