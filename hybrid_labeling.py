"""
Hybrid Interactive and Automated Labeling Method

This module implements Algorithm 1: Hybrid Interactive and Automated Labeling Method
for lane marking annotation compatible with CLRerNet architecture.

The hybrid approach combines:
1. Manual point annotations on selected frames
2. Automated propagation across sequential frames using directional translation
3. Generation of segmentation masks from annotated points

This method addresses limitations in CDLEM and ALINA by reducing manual effort
while maintaining annotation consistency.
"""

import os
import cv2
import numpy as np
import re


class HybridLabeler:
    """
    Implements the hybrid labeling algorithm for lane marking annotation.
    
    The algorithm processes image sequences by:
    - Manually annotating key frames at regular intervals (step size σ)
    - Automatically propagating annotations to intermediate frames
    - Generating binary segmentation masks for all frames
    """
    
    def __init__(self, step_size=60, adjustment_factor=1.5):
        """
        Initialize the hybrid labeler.
        
        Args:
            step_size (int): Number of frames between manual annotations (σ in Algorithm 1)
            adjustment_factor (float): Scaling factor for directional translation
        """
        self.step_size = step_size
        self.adjustment_factor = adjustment_factor
        self.current_points = []
        self.lanes_points = []
        self.lane_directions = []
        
    def gamma(self, direction):
        """
        Directional translation function γ(δ) from Algorithm 1.
        
        Maps lane direction to a translation vector for point adjustment.
        
        Args:
            direction (str): Lane direction ('left', 'right', 'up', 'down', 'straight')
            
        Returns:
            tuple: (dx, dy) translation vector
        """
        direction_map = {
            'left': (-self.adjustment_factor, 0),
            'right': (self.adjustment_factor, 0),
            'up': (0, -self.adjustment_factor),
            'down': (0, self.adjustment_factor),
            'straight': (0, 0)
        }
        return direction_map.get(direction, (0, 0))
    
    def adjust_points(self, points, k, i, direction):
        """
        Adjust points using directional translation function (Line 8 in Algorithm 1).
        
        Implements: P_k = P_i + (k - i) · γ(δ)
        
        Args:
            points (list): Initial lane points P_i
            k (int): Current frame index
            i (int): Initial frame index
            direction (str): Lane direction δ
            
        Returns:
            list: Adjusted points P_k
        """
        dx, dy = self.gamma(direction)
        frame_offset = k - i
        adjusted = [(x + frame_offset * dx, y + frame_offset * dy) for x, y in points]
        return adjusted
    
    def mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback for manual point selection.
        
        Allows user to click points on the image for lane marking.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))
            cv2.circle(param, (x, y), 4, (0, 0, 255), -1)
            cv2.imshow("Image", param)
    
    def manual_annotation(self, image, output_file):
        """
        Manual selection of lane points P_i on image I_i (Lines 3-5 in Algorithm 1).
        
        User selects points and defines lane direction δ.
        
        Args:
            image (numpy.ndarray): Image to annotate
            output_file (str): Path to save annotations
            
        Returns:
            None
        """
        self.current_points = []
        direction = 'straight'
        img_copy = image.copy()
        
        cv2.imshow("Image", img_copy)
        cv2.setMouseCallback("Image", self.mouse_callback, img_copy)
        
        print("Instructions:")
        print("  - Left click: Select lane points")
        print("  - 'k': Save current lane and start new one")
        print("  - Arrow keys: Set direction (Left/Right/Up/Down, default: Straight)")
        print("  - 'r': Redo current frame")
        print("  - Enter: Finish and proceed")
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('k'):  # Save current lane
                self._save_points(output_file)
                self.lanes_points.append(self.current_points.copy())
                self.lane_directions.append(direction)
                self.current_points = []
                direction = 'straight'
                
            elif key == 13:  # Enter - finish
                if self.current_points:
                    self._save_points(output_file)
                    self.lanes_points.append(self.current_points.copy())
                    self.lane_directions.append(direction)
                break
                
            elif key == ord('r'):  # Redo
                print("Redoing current frame...")
                self.current_points = []
                img_copy = image.copy()
                cv2.imshow("Image", img_copy)
                cv2.setMouseCallback("Image", self.mouse_callback, img_copy)
                
            elif key == 81:  # Left arrow
                direction = 'left'
                print(f"Direction set to: {direction}")
                
            elif key == 82:  # Up arrow
                direction = 'up'
                print(f"Direction set to: {direction}")
                
            elif key == 83:  # Right arrow
                direction = 'right'
                print(f"Direction set to: {direction}")
                
            elif key == 84:  # Down arrow
                direction = 'down'
                print(f"Direction set to: {direction}")
        
        cv2.destroyAllWindows()
    
    def _save_points(self, file_path):
        """
        Save points to annotation file (Line 5 in Algorithm 1).
        
        Args:
            file_path (str): Path to annotation file
        """
        with open(file_path, 'a') as f:
            f.write(' '.join(f'{x} {y}' for x, y in self.current_points) + '\n')
    
    def propagate_annotations(self, image, i, k, output_file):
        """
        Propagate adjusted points to frame I_k (Lines 7-10 in Algorithm 1).
        
        Args:
            image (numpy.ndarray): Image to annotate
            i (int): Initial frame index
            k (int): Current frame index
            output_file (str): Path to save annotations
        """
        img_copy = image.copy()
        
        # Process each lane
        for lane_idx, initial_points in enumerate(self.lanes_points):
            direction = self.lane_directions[lane_idx]
            # Adjust points using directional translation (Line 8)
            adjusted_points = self.adjust_points(initial_points, k, i, direction)
            
            # Visualize adjusted points
            for x, y in adjusted_points:
                cv2.circle(img_copy, (int(x), int(y)), 4, (0, 255, 0), -1)
            
            # Save annotations (Line 10)
            with open(output_file, 'a') as f:
                f.write(' '.join(f'{x:.2f} {y:.2f}' for x, y in adjusted_points) + '\n')
        
        # Display automated frame briefly
        cv2.imshow("Automated Frame", img_copy)
        cv2.waitKey(50)
    
    def process_image_sequence(self, input_dir, output_dir):
        """
        Main algorithm loop (Lines 2-13 in Algorithm 1).
        
        Process image sequence I = {I_1, I_2, ..., I_N} with hybrid labeling.
        
        Args:
            input_dir (str): Directory containing input images
            output_dir (str): Directory for output annotations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get sorted image files
        files = sorted(
            [f for f in os.listdir(input_dir) if f.endswith('.jpg')],
            key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf')
        )
        
        N = len(files)
        i = 0  # Frame index (Line 1)
        
        # Main loop (Line 2)
        while i < N:
            image_path = os.path.join(input_dir, files[i])
            image = cv2.imread(image_path)
            output_file = os.path.join(output_dir, files[i].replace('.jpg', '.lines.txt'))
            
            print(f"Processing frame {i+1}/{N} - {files[i]}")
            
            # Manual annotation (Lines 3-5)
            self.lanes_points = []
            self.lane_directions = []
            self.manual_annotation(image, output_file)
            
            # Automated propagation for subsequent frames (Lines 6-10)
            for j in range(1, self.step_size):
                k = i + j
                if k >= N:
                    break
                
                next_image_path = os.path.join(input_dir, files[k])
                next_image = cv2.imread(next_image_path)
                next_output_file = os.path.join(output_dir, files[k].replace('.jpg', '.lines.txt'))
                
                self.propagate_annotations(next_image, i, k, next_output_file)
            
            cv2.destroyAllWindows()
            i += self.step_size  # Line 13
    
    def generate_segmentation_masks(self, base_dir, annotations_dir, output_dir, image_list_file=None):
        """
        Generate segmentation masks M_i from lane points (Lines 14-19 in Algorithm 1).
        
        Implements mask generation: M_i(x,y) = 1 if (x,y) in lane segment, 0 otherwise
        
        Args:
            base_dir (str): Base directory containing images
            annotations_dir (str): Directory with .lines.txt annotation files
            output_dir (str): Directory to save segmentation masks
            image_list_file (str): Optional file listing images to process
        """
        mask_dir = os.path.join(output_dir, 'images')
        os.makedirs(mask_dir, exist_ok=True)
        
        # Get list of images to process
        if image_list_file and os.path.exists(image_list_file):
            with open(image_list_file, 'r') as f:
                image_paths = [line.strip() for line in f.readlines()]
        else:
            # Process all images in annotations directory
            image_paths = [f.replace('.lines.txt', '.jpg') 
                          for f in os.listdir(annotations_dir) 
                          if f.endswith('.lines.txt')]
        
        output_list_file = os.path.join(output_dir, 'segmentation_list.txt')
        
        with open(output_list_file, 'w') as out_list:
            for img_path in image_paths:
                if image_list_file:
                    full_image_path = os.path.join(base_dir, img_path)
                else:
                    full_image_path = os.path.join(base_dir, img_path)
                
                annotation_file = os.path.join(annotations_dir, 
                                              os.path.basename(img_path).replace('.jpg', '.lines.txt'))
                
                # Load image
                image = cv2.imread(full_image_path)
                if image is None:
                    print(f"Warning: Could not load image {full_image_path}")
                    continue
                
                # Create blank mask (Line 16)
                mask = np.zeros_like(image)
                
                # Read lane points and generate mask
                if os.path.exists(annotation_file):
                    with open(annotation_file, 'r') as f:
                        for lane_idx, line in enumerate(f):
                            points = line.strip().split()
                            if len(points) < 4:
                                continue
                            
                            # Extract points
                            point_coords = []
                            for i in range(0, len(points), 2):
                                x = int(round(float(points[i])))
                                y = int(round(float(points[i+1])))
                                point_coords.append((x, y))
                            
                            # Draw lane segments (Line 17)
                            for j in range(len(point_coords) - 1):
                                pt1, pt2 = point_coords[j], point_coords[j+1]
                                # Ensure points are within bounds
                                if (0 <= pt1[0] < image.shape[1] and 0 <= pt1[1] < image.shape[0] and
                                    0 <= pt2[0] < image.shape[1] and 0 <= pt2[1] < image.shape[0]):
                                    cv2.line(mask, pt1, pt2, [(lane_idx+1)] * 4, 3, lineType=cv2.LINE_8)
                
                # Save mask (Line 19)
                mask_filename = os.path.join(mask_dir, 
                                            os.path.basename(full_image_path).replace('.jpg', '.png'))
                cv2.imwrite(mask_filename, mask)
                
                # Write to output list
                if image_list_file:
                    out_list.write(f"{img_path} {os.path.relpath(mask_filename, base_dir)}\n")
                else:
                    out_list.write(f"{os.path.basename(img_path)} {os.path.basename(mask_filename)}\n")
        
        print(f"Segmentation masks saved to {mask_dir}")
        print(f"Output list saved to {output_list_file}")


def main():
    """
    Example usage of the hybrid labeling algorithm.
    """
    # Configuration
    input_dir = 'datasets/assisttaxi2/valid/images2'
    output_dir = 'datasets/assisttaxi2/valid/processed_images'
    step_size = 60  # σ in Algorithm 1
    adjustment_factor = 1.5  # Scaling for γ(δ)
    
    # Initialize labeler
    labeler = HybridLabeler(step_size=step_size, adjustment_factor=adjustment_factor)
    
    # Phase 1: Process image sequence with hybrid labeling (Lines 1-13)
    print("=" * 60)
    print("Phase 1: Hybrid Interactive and Automated Labeling")
    print("=" * 60)
    labeler.process_image_sequence(input_dir, output_dir)
    
    # Phase 2: Generate segmentation masks (Lines 14-19)
    print("\n" + "=" * 60)
    print("Phase 2: Segmentation Mask Generation")
    print("=" * 60)
    mask_output_dir = 'datasets/assisttaxi2/valid/segmentation_masks'
    labeler.generate_segmentation_masks(
        base_dir='datasets/assisttaxi2/valid',
        annotations_dir=output_dir,
        output_dir=mask_output_dir
    )
    
    print("\n" + "=" * 60)
    print("Hybrid labeling completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
