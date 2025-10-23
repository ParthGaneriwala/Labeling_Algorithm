#!/usr/bin/env python3
"""
Example usage script for the Hybrid Labeling Algorithm

This script demonstrates how to use the HybridLabeler class for
annotating lane markings in a video sequence.
"""

import os
import sys
from hybrid_labeling import HybridLabeler


def example_basic_usage():
    """
    Example 1: Basic usage with default parameters
    """
    print("=" * 70)
    print("Example 1: Basic Usage")
    print("=" * 70)
    
    # Configure paths
    input_dir = 'datasets/assisttaxi2/valid/images2'
    output_dir = 'datasets/assisttaxi2/valid/processed_images'
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Note: Input directory '{input_dir}' does not exist.")
        print("Please update the paths in this script to match your dataset location.")
        return False
    
    # Initialize labeler with default parameters
    labeler = HybridLabeler(
        step_size=60,           # Annotate every 60th frame
        adjustment_factor=1.5   # Movement scaling
    )
    
    # Process the image sequence
    print(f"\nProcessing images from: {input_dir}")
    print(f"Saving annotations to: {output_dir}")
    labeler.process_image_sequence(input_dir, output_dir)
    
    print("\n✓ Annotation phase completed!")
    return True


def example_custom_parameters():
    """
    Example 2: Custom parameters for different scenarios
    """
    print("\n" + "=" * 70)
    print("Example 2: Custom Parameters")
    print("=" * 70)
    
    # For highway footage (stable scenes)
    labeler_highway = HybridLabeler(
        step_size=100,          # Less frequent manual annotation
        adjustment_factor=1.0   # Slower lane movement
    )
    
    # For urban footage (dynamic scenes)
    labeler_urban = HybridLabeler(
        step_size=30,           # More frequent manual annotation
        adjustment_factor=2.0   # Faster lane movement
    )
    
    print("\nConfiguration for highway footage:")
    print(f"  Step size: {labeler_highway.step_size}")
    print(f"  Adjustment factor: {labeler_highway.adjustment_factor}")
    
    print("\nConfiguration for urban footage:")
    print(f"  Step size: {labeler_urban.step_size}")
    print(f"  Adjustment factor: {labeler_urban.adjustment_factor}")
    
    # You can then use these labelers with:
    # labeler_highway.process_image_sequence(highway_input_dir, highway_output_dir)
    # labeler_urban.process_image_sequence(urban_input_dir, urban_output_dir)


def example_full_pipeline():
    """
    Example 3: Complete pipeline including mask generation
    """
    print("\n" + "=" * 70)
    print("Example 3: Full Pipeline (Annotation + Mask Generation)")
    print("=" * 70)
    
    # Configure paths
    base_dir = 'datasets/assisttaxi2/valid'
    input_dir = os.path.join(base_dir, 'images2')
    annotations_dir = os.path.join(base_dir, 'processed_images')
    masks_dir = os.path.join(base_dir, 'segmentation_masks')
    
    if not os.path.exists(input_dir):
        print(f"Note: Input directory '{input_dir}' does not exist.")
        print("This is a demonstration of the complete pipeline.")
        print("Update paths to match your dataset location.")
        return
    
    # Initialize labeler
    labeler = HybridLabeler(step_size=60, adjustment_factor=1.5)
    
    # Phase 1: Annotation
    print("\n--- Phase 1: Interactive Labeling ---")
    labeler.process_image_sequence(input_dir, annotations_dir)
    
    # Phase 2: Mask Generation
    print("\n--- Phase 2: Segmentation Mask Generation ---")
    labeler.generate_segmentation_masks(
        base_dir=base_dir,
        annotations_dir=annotations_dir,
        output_dir=masks_dir
    )
    
    print("\n✓ Complete pipeline executed successfully!")
    print(f"  Annotations: {annotations_dir}")
    print(f"  Masks: {masks_dir}")


def example_with_training_list():
    """
    Example 4: Generate masks from a training list file
    """
    print("\n" + "=" * 70)
    print("Example 4: Mask Generation with Training List")
    print("=" * 70)
    
    base_dir = 'datasets/assisttaxi2/train'
    train_txt = os.path.join(base_dir, 'train.txt')
    annotations_dir = os.path.join(base_dir, 'processed_images')
    output_dir = os.path.join(base_dir, 'laneseg_label_w16')
    
    if not os.path.exists(train_txt):
        print(f"Note: Training list '{train_txt}' does not exist.")
        print("This example shows how to use a training list file.")
        return
    
    labeler = HybridLabeler()
    
    print(f"Generating masks for images listed in: {train_txt}")
    labeler.generate_segmentation_masks(
        base_dir=base_dir,
        annotations_dir=annotations_dir,
        output_dir=output_dir,
        image_list_file=train_txt
    )
    
    print("\n✓ Masks generated for training set!")


def demonstrate_gamma_function():
    """
    Example 5: Understanding the directional translation function
    """
    print("\n" + "=" * 70)
    print("Example 5: Directional Translation Function γ(δ)")
    print("=" * 70)
    
    labeler = HybridLabeler(adjustment_factor=1.5)
    
    print("\nDirectional translation vectors:")
    directions = ['left', 'right', 'up', 'down', 'straight']
    for direction in directions:
        dx, dy = labeler.gamma(direction)
        print(f"  γ('{direction}') = ({dx:+.1f}, {dy:+.1f})")
    
    print("\nPoint adjustment example:")
    initial_point = (100, 200)
    print(f"  Initial point: {initial_point}")
    
    for frame_offset in range(1, 4):
        for direction in ['right', 'down', 'straight']:
            adjusted = labeler.adjust_points([initial_point], frame_offset, 0, direction)
            print(f"  Frame +{frame_offset}, direction '{direction}': {adjusted[0]}")


def main():
    """
    Main function to run all examples
    """
    print("\n" + "=" * 70)
    print("HYBRID LABELING ALGORITHM - EXAMPLE USAGE")
    print("=" * 70)
    
    try:
        # Run examples
        example_basic_usage()
        example_custom_parameters()
        example_full_pipeline()
        example_with_training_list()
        demonstrate_gamma_function()
        
        print("\n" + "=" * 70)
        print("All examples completed!")
        print("=" * 70)
        print("\nTo use the hybrid labeling algorithm:")
        print("  1. Update the paths in hybrid_labeling.py main() function")
        print("  2. Run: python3 hybrid_labeling.py")
        print("\nFor more information, see HYBRID_LABELING_DOCS.md")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
