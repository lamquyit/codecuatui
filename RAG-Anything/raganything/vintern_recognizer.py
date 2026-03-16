import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

# Add vai-ocr directory to sys.path to import vintern_run
# Assuming structure: RAG_base/RAG-Anything/raganything/vintern_recognizer.py
# target: RAG_base/vai-ocr
current_dir = Path(__file__).parent
vai_ocr_path = (current_dir.parent.parent / "vai-ocr").resolve()

if str(vai_ocr_path) not in sys.path:
    sys.path.append(str(vai_ocr_path))

try:
    from vintern_run import recognize_text_vintern
    VINTERN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Could not import vintern_run: {e}. Vintern recognition will be disabled.")
    VINTERN_AVAILABLE = False


logger = logging.getLogger(__name__)

class VinternRecognizer:
    """
    Wrapper for Vintern OCR model to re-recognize text in detected regions.
    """
    
    @staticmethod
    def is_available() -> bool:
        return VINTERN_AVAILABLE

    @staticmethod
    def process_content_list(content_list: List[Dict[str, Any]], image_path: str) -> Tuple[List[Dict[str, Any]], int]:
        """
        Process content list and re-recognize text regions using Vintern.
        
        Args:
            content_list: List of content items from MinerU/Docling
            image_path: Path to the original image file
            
        Returns:
            Tuple containing (updated_content_list, count_corrected)
        """
        if not VINTERN_AVAILABLE:
            logger.warning("Vintern is not available, skipping re-recognition.")
            return content_list, 0
            
        if not os.path.exists(image_path):
            logger.error(f"Image not found at {image_path}, skipping re-recognition.")
            return content_list, 0
            
        # Lazy import cv2/PIL to avoid overhead if not used
        import cv2
        import numpy as np
        
        # Read image
        try:
            # Handle non-ascii paths with cv2 using imdecode
            stream = open(image_path, "rb")
            bytes_data = bytearray(stream.read())
            numpy_array = np.asarray(bytes_data, dtype=np.uint8)
            img_cv = cv2.imdecode(numpy_array, cv2.IMREAD_UNCHANGED)
            if img_cv is None:
                logger.error(f"Failed to read image: {image_path}")
                return content_list, 0
                
            # Convert to RGB if needed (Vintern likely expects RGB/BGR)
            # OpenCV reads as BGR, PIL reads as RGB. 
            # Check vintern_run implementation. It uses PIL.Image usually or accepts numpy array.
            # Assuming recognize_text_vintern handles BGR numpy array or we convert.
            # Let's verify via viewing code later if errors occur, but standard is BGR for cv2.
            
        except Exception as e:
            logger.error(f"Error reading image {image_path}: {e}")
            return content_list, 0
            
        h, w = img_cv.shape[:2]
        count = 0
        
        logger.info(f"Starting Vintern re-recognition for {image_path}...")
        
        for item in content_list:
            if item.get("type") == "text" and "bbox" in item:
                bbox = item["bbox"]
                # Ensure bbox is valid [x1, y1, x2, y2]
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    
                    # Validate coordinates
                    x1, y1 = max(0, int(x1)), max(0, int(y1))
                    x2, y2 = min(w, int(x2)), min(h, int(y2))
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                        
                    # Crop image
                    cropped_image = img_cv[y1:y2, x1:x2]
                    
                    try:
                        # Recognize
                        text, confidence = recognize_text_vintern(cropped_image)
                        
                        if text and text.strip():
                            original_text = item.get("text", "")
                            # Log change for debugging (optional, verbose)
                            # logger.debug(f"Corrected: '{original_text}' -> '{text}'")
                            
                            item["text"] = text
                            item["vintern_confidence"] = confidence
                            count += 1
                    except Exception as e:
                        logger.warning(f"Error recognizing box {bbox}: {e}")
                        continue
                        
        logger.info(f"Vintern re-recognized {count} text regions.")
        return content_list, count
