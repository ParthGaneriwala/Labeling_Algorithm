# Hybrid Labeling Algorithm - Quick Reference

## Quick Start

```bash
# Install dependencies
pip install opencv-python numpy

# Run the algorithm
python3 hybrid_labeling.py
```

## Algorithm Summary

**Input:** Image sequence I = {I₁, I₂, ..., Iₙ}  
**Output:** Lane annotations + segmentation masks

**Key Formula:** Pₖ = Pᵢ + (k - i) · γ(δ)

Where:
- Pₖ: Lane points at frame k
- Pᵢ: Initial lane points at frame i
- γ(δ): Directional translation function
- δ: Direction (left, right, up, down, straight)

## Interactive Controls

| Key | Action |
|-----|--------|
| Left Click | Select point |
| k | Save lane, start new |
| ← → ↑ ↓ | Set direction |
| r | Redo frame |
| Enter | Finish |

## Default Parameters

- **Step Size (σ)**: 60 frames
- **Adjustment Factor**: 1.5 pixels/frame
- **Default Direction**: straight

## Directional Translation γ(δ)

```
γ('left')     = (-1.5, 0)
γ('right')    = (1.5, 0)
γ('up')       = (0, -1.5)
γ('down')     = (0, 1.5)
γ('straight') = (0, 0)
```

## Configuration

Edit in `hybrid_labeling.py` main() function:

```python
labeler = HybridLabeler(
    step_size=60,           # σ: frames between manual annotations
    adjustment_factor=1.5   # scaling for γ(δ)
)
```

## Output Files

1. **Annotations**: `*.lines.txt`
   - Format: `x1 y1 x2 y2 ... xn yn`
   - One line per lane

2. **Segmentation Masks**: `*.png`
   - Binary images
   - Lane pixels > 0, background = 0

3. **Image List**: `segmentation_list.txt`
   - Format: `<image_path> <mask_path>`

## Comparison with Other Methods

| Method | Manual Effort | Consistency |
|--------|---------------|-------------|
| CDLEM | High | Low |
| ALINA | Medium | Medium |
| **Hybrid** | **Low (1.67%)** | **High** |

## Advantages

✓ 98.33% reduction in manual effort (σ=60)  
✓ Mathematical consistency  
✓ CLRerNet compatible  
✓ Scalable to large datasets  
✓ No ROI definition required  

## Parameter Selection Guide

### Step Size (σ)

- **30-40**: Dynamic urban scenes
- **60** (default): Balanced, most scenarios
- **80-100**: Stable highway footage

### Adjustment Factor

- **1.0-1.5**: Slow-moving or stationary camera
- **1.5** (default): Normal driving speeds
- **2.0-3.0**: High-speed or rapid scene changes

## Usage Examples

### Basic Usage
```python
from hybrid_labeling import HybridLabeler

labeler = HybridLabeler()
labeler.process_image_sequence('input/', 'output/')
```

### Full Pipeline
```python
labeler = HybridLabeler(step_size=60, adjustment_factor=1.5)

# Phase 1: Annotate
labeler.process_image_sequence('images/', 'annotations/')

# Phase 2: Generate masks
labeler.generate_segmentation_masks(
    base_dir='dataset/',
    annotations_dir='annotations/',
    output_dir='masks/'
)
```

### Custom Parameters
```python
# Highway footage
highway_labeler = HybridLabeler(step_size=100, adjustment_factor=1.0)

# Urban footage
urban_labeler = HybridLabeler(step_size=30, adjustment_factor=2.0)
```

## Troubleshooting

**Problem**: Annotations drift over frames  
**Solution**: Reduce step_size or adjustment_factor

**Problem**: Points outside image  
**Solution**: Automatic clipping is built-in

**Problem**: Slow processing  
**Solution**: Increase step_size

## Files

- `hybrid_labeling.py` - Main implementation
- `HYBRID_LABELING_DOCS.md` - Full documentation
- `example_usage.py` - Usage examples
- `README.md` - Repository overview

## References

Based on research addressing CLRerNet lane detection architecture requirements.

For detailed information, see `HYBRID_LABELING_DOCS.md`.
