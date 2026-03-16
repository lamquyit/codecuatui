"""
Vintern Client for vai-ocr Integration.
Connects to the vai-ocr FastAPI server running Vintern OCR.
"""

import requests
import base64
import logging
import tempfile
import os
from typing import Optional, Union
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)


class VinternClient:
    """
    Client for interacting with the vai-ocr Vintern API server.
    
    The vai-ocr server uses PaddleOCR for detection + Vintern for recognition.
    Default endpoint: http://localhost:8003
    """
    
    def __init__(self, api_url: str = "http://localhost:8003"):
        """
        Initialize Vintern client.
        
        Args:
            api_url: Base URL of the vai-ocr server (default: http://localhost:8003)
        """
        self.api_url = api_url.rstrip("/")
        self.ocr_endpoint = f"{self.api_url}/ocr-file/"
        
    def _save_image_to_temp(self, image_input) -> str:
        """
        Save PIL Image or image path to a temporary file.
        Returns the path to the temp file.
        """
        if isinstance(image_input, str) and os.path.exists(image_input):
            return image_input  # Already a file path
            
        if isinstance(image_input, Image.Image):
            # Save PIL Image to temp file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image_input.save(tmp, format="PNG")
                return tmp.name
                
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def classify(self, image_input) -> str:
        """
        Classify if image contains handwriting or printed text.
        
        Note: vai-ocr doesn't have a dedicated classify endpoint.
        We use OCR and check if the result looks like handwriting based on confidence.
        For now, we return "YES" to always process with Vintern since that's the goal.
        
        Returns:
            'YES' if handwriting, 'NO' if printed, 'ERROR' on failure
        """
        # Simplified: Always return YES to use Vintern for OCR
        # In production, you could implement heuristics here
        return "YES"

    def ocr(self, image_input) -> str:
        """
        Perform OCR on an image using vai-ocr Vintern server.
        
        Args:
            image_input: File path (str) or PIL Image
            
        Returns:
            Extracted text from the image
        """
        temp_file_created = False
        
        try:
            # Get file path
            if isinstance(image_input, Image.Image):
                file_path = self._save_image_to_temp(image_input)
                temp_file_created = True
            elif isinstance(image_input, str):
                file_path = image_input
            else:
                raise ValueError(f"Unsupported image input: {type(image_input)}")
            
            # Prepare multipart form data
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f, "image/png")}
                data = {"language": "vi"}  # Vietnamese language
                
                logger.info(f"Calling vai-ocr API at {self.ocr_endpoint}")
                response = requests.post(
                    self.ocr_endpoint,
                    files=files,
                    data=data,
                    timeout=60  # Vintern can be slow
                )
                response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            # Extract text from results
            # vai-ocr returns: {"results": [[{"box": [...], "text": "...", "conf": 0.99}, ...]]}
            all_text = []
            
            if "results" in result:
                for page_results in result["results"]:
                    if isinstance(page_results, list):
                        for item in page_results:
                            if isinstance(item, dict) and "text" in item:
                                all_text.append(item["text"])
            
            # Also check raw_results if present
            if "raw_results" in result:
                for page_results in result["raw_results"]:
                    if isinstance(page_results, list):
                        for item in page_results:
                            if isinstance(item, dict) and "text" in item:
                                if item["text"] not in all_text:
                                    all_text.append(item["text"])
            
            extracted_text = "\n".join(all_text)
            logger.info(f"Extracted {len(all_text)} text regions from image")
            
            return extracted_text
            
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to vai-ocr server at {self.api_url}. Is it running?")
            return ""
        except Exception as e:
            logger.error(f"Error calling vai-ocr OCR: {e}")
            return ""
        finally:
            # Cleanup temp file if created
            if temp_file_created and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except:
                    pass

    def ocr_with_details(self, image_input) -> dict:
        """
        Perform OCR and return full details including bounding boxes.
        
        Returns:
            Dictionary with 'text', 'boxes', and 'confidence' info
        """
        temp_file_created = False
        
        try:
            if isinstance(image_input, Image.Image):
                file_path = self._save_image_to_temp(image_input)
                temp_file_created = True
            else:
                file_path = image_input
            
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f, "image/png")}
                data = {"language": "vi"}
                
                response = requests.post(
                    self.ocr_endpoint,
                    files=files,
                    data=data,
                    timeout=60
                )
                response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error calling vai-ocr: {e}")
            return {"error": str(e), "results": []}
        finally:
            if temp_file_created and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except:
                    pass
