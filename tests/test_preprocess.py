import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.preprocessing.preprocess import find_endoscopy_bbox, crop_image_remove_black_borders, process_dataset

def test_find_endoscopy_bbox():
    # Create a dummy image (black background with a white square in the middle)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = 255
    
    bbox = find_endoscopy_bbox(img)
    assert bbox is not None
    x, y, w, h = bbox
    assert x == 20
    assert y == 20
    assert w == 60
    assert h == 60

def test_find_endoscopy_bbox_no_content():
    # Completely black image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = find_endoscopy_bbox(img)
    assert bbox == (0, 0, 100, 100)

def test_crop_image_remove_black_borders():
    # Mock image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = 255
    
    cropped, bbox = crop_image_remove_black_borders(img)
    assert bbox == (20, 20, 60, 60)
    assert cropped.shape == (60, 60, 3)

@patch('src.preprocessing.preprocess.cv2.imread')
@patch('src.preprocessing.preprocess.cv2.imwrite')
@patch('src.preprocessing.preprocess.iter_images')
def test_process_dataset(mock_iter, mock_imwrite, mock_imread):
    from pathlib import Path
    mock_iter.return_value = [Path("dummy_in_dir/img1.jpg"), Path("dummy_in_dir/img2.jpg")]
    
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = 255
    mock_imread.return_value = img
    
    results = process_dataset(Path("dummy_in_dir"), Path("dummy_out_dir"))
    
    assert len(results) == 2
    assert mock_imwrite.call_count == 2
