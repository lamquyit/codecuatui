import logging
import numpy as np
from typing import Optional
from PIL import Image
from pathlib import Path
from raganything.vintern_client import VinternClient
from raganything.config import RAGAnythingConfig

logger = logging.getLogger(__name__)


class HandwritingRouter:
    """
    Decides if a text block is handwriting or printed text.
    Uses a hybrid approach: Heuristic filter + Vintern Verification.
    """

    def __init__(self, config: RAGAnythingConfig):
        self.config = config
        self.client = None

        # Initialize client if handwriting processing is enabled
        if getattr(config, "enable_handwriting_processing", False):
            # TODO: Get API URL from config or env
            self.client = VinternClient()

    def is_handwriting(self, image_crop: Image.Image) -> bool:
        """
        Check if the image crop is handwriting.
        Returns True if handwriting, False if printed text.
        """
        if not self.client:
            return False

        # Step 1: Heuristic Pre-filter (Fast)
        # Skip blocks that are obviously printed text
        if self._is_obviously_printed(image_crop):
            logger.debug("Heuristic: Detected as printed text, skipping Vintern call")
            return False

        # Step 2: Vintern Verification (Slower but accurate)
        try:
            response = self.client.classify(image_crop)
            logger.info(f"Vintern Classification: {response}")

            # New format: client now returns YES/NO/ERROR
            if response == "YES":
                return True
            else:
                return False

        except Exception as e:
            logger.warning(f"Handwriting check failed, assuming printed text: {e}")
            return False

    def _is_obviously_printed(self, img: Image.Image) -> bool:
        """
        Heuristic check to quickly identify obviously printed text.
        Uses image statistics to detect clean, uniform backgrounds typical of printed docs.
        
        Returns True if the image is likely printed text (skip Vintern).
        Returns False if uncertain (needs Vintern verification).
        """
        try:
            # Convert to grayscale numpy array
            gray = img.convert("L")
            arr = np.array(gray)

            # Calculate statistics
            mean_val = np.mean(arr)
            std_val = np.std(arr)

            # Heuristic 1: Very clean background (high mean, low std)
            # Typical printed doc: white bg (mean > 230), uniform (std < 30)
            if mean_val > 220 and std_val < 30:
                # Could still be handwriting on white paper, so don't filter too aggressively
                # Only skip if extremely uniform (likely computer-generated)
                if std_val < 15:
                    return True

            # Heuristic 2: Check for horizontal line regularity (typical of printed text)
            # This is a simplified check - real implementation could use Hough transform
            row_means = np.mean(arr, axis=1)
            row_std = np.std(row_means)

            # Very regular rows (low variation) suggest printed text
            if row_std < 5:
                return True

            return False

        except Exception as e:
            logger.debug(f"Heuristic check failed: {e}")
            return False  # If heuristic fails, proceed to Vintern
