#!/usr/bin/env python3
"""
Validate YOLO annotations for potential issues
"""

import os
from pathlib import Path
import numpy as np


def validate_annotations(dataset_dir="/Users/daehyunyoo/Desktop/Military Object Detection/data/yolo_dataset"):
    """Validate YOLO format annotations for common issues"""
    
    dataset_path = Path(dataset_dir)
    issues = []
    stats = {
        'total_files': 0,
        'total_objects': 0,
        'invalid_coords': 0,
        'invalid_classes': 0,
        'empty_files': 0,
        'class_range': [float('inf'), -float('inf')]
    }
    
    for split in ['train', 'val']:
        labels_dir = dataset_path / split / 'labels'
        if not labels_dir.exists():
            continue
            
        label_files = list(labels_dir.glob('*.txt'))
        print(f"Checking {len(label_files)} files in {split} split...")
        
        for label_file in label_files:
            stats['total_files'] += 1
            
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                if not lines:
                    stats['empty_files'] += 1
                    continue
                
                for line_num, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        parts = line.split()
                        if len(parts) != 5:
                            issues.append(f"{label_file}:{line_num+1} - Wrong number of values: {len(parts)}")
                            continue
                        
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:])
                        
                        stats['total_objects'] += 1
                        
                        # Update class range
                        stats['class_range'][0] = min(stats['class_range'][0], class_id)
                        stats['class_range'][1] = max(stats['class_range'][1], class_id)
                        
                        # Validate class ID
                        if class_id < 0 or class_id > 95:  # We have 96 classes (0-95)
                            issues.append(f"{label_file}:{line_num+1} - Invalid class ID: {class_id}")
                            stats['invalid_classes'] += 1
                        
                        # Validate coordinates (should be between 0 and 1)
                        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                            issues.append(f"{label_file}:{line_num+1} - Invalid coordinates: {x_center}, {y_center}, {width}, {height}")
                            stats['invalid_coords'] += 1
                        
                        # Check for extremely small or large boxes
                        if width <= 0 or height <= 0:
                            issues.append(f"{label_file}:{line_num+1} - Zero or negative box size: {width}x{height}")
                            stats['invalid_coords'] += 1
                        
                        if width > 1 or height > 1:
                            issues.append(f"{label_file}:{line_num+1} - Box size larger than image: {width}x{height}")
                            stats['invalid_coords'] += 1
                    
                    except ValueError as e:
                        issues.append(f"{label_file}:{line_num+1} - Parse error: {e}")
            
            except Exception as e:
                issues.append(f"{label_file} - File error: {e}")
    
    # Print validation results
    print("\n" + "="*60)
    print("ANNOTATION VALIDATION RESULTS")
    print("="*60)
    
    print(f"Total files checked: {stats['total_files']}")
    print(f"Total objects: {stats['total_objects']}")
    print(f"Empty files: {stats['empty_files']}")
    print(f"Class range: {stats['class_range'][0]} to {stats['class_range'][1]}")
    
    print(f"\nIssues found:")
    print(f"  Invalid coordinates: {stats['invalid_coords']}")
    print(f"  Invalid classes: {stats['invalid_classes']}")
    print(f"  Total issues: {len(issues)}")
    
    if issues:
        print(f"\nFirst 20 issues:")
        for issue in issues[:20]:
            print(f"  {issue}")
        
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more issues")
    else:
        print("\n✅ No issues found!")
    
    return stats, issues


if __name__ == "__main__":
    validate_annotations()