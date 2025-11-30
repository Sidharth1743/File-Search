"""
Enhanced OCR Engine for Historical Medical Documents
Handles image preprocessing and text extraction from PDFs and images
"""

import os
import google.generativeai as genai
import PIL.Image
from PIL import ImageEnhance, ImageFilter
import fitz  # PyMuPDF
import cv2
import numpy as np
from typing import Optional, Dict, Any


class OCREngine:
    """
    Enhanced OCR system optimized for old medical documents with advanced preprocessing.
    """
    
    def __init__(self, api_key: Optional[str] = None, use_advanced_model: bool = True):
        """
        Initialize OCR engine with Gemini API.
        
        Args:
            api_key: Google API key (if None, reads from GOOGLE_API_KEY env var)
            use_advanced_model: Use gemini-2.0-flash-exp for better accuracy
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables or provided")
        
        genai.configure(api_key=self.api_key)
        self.model = self._configure_model(use_advanced_model)
        print(f"✅ OCR Engine initialized with model: {self.model.model_name}")
    
    def _configure_model(self, use_advanced_model: bool):
        """Configure Gemini model with optimal settings."""
        generation_config = {
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        model_name = 'gemini-2.0-flash-exp' if use_advanced_model else 'gemini-2.0-flash'
        
        return genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
    
    @staticmethod
    def preprocess_image(image: PIL.Image, enhancement_level: str = "medium") -> PIL.Image:
        """
        Apply advanced preprocessing to improve OCR accuracy.
        
        Args:
            image: PIL Image object
            enhancement_level: "light", "medium", or "aggressive"
        
        Returns:
            Enhanced PIL Image
        """
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        if enhancement_level == "aggressive":
            denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            kernel = np.ones((2,2), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            enhanced = PIL.Image.fromarray(cleaned)
            
        elif enhancement_level == "medium":
            denoised = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced_cv = clahe.apply(denoised)
            blurred = cv2.GaussianBlur(enhanced_cv, (0,0), 3)
            sharpened = cv2.addWeighted(enhanced_cv, 1.5, blurred, -0.5, 0)
            enhanced = PIL.Image.fromarray(sharpened)
            
        else:  # "light"
            enhanced = PIL.Image.fromarray(gray)
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.5)
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.3)
        
        if enhancement_level != "aggressive":
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(1.1)
        
        return enhanced
    
    @staticmethod
    def detect_and_deskew(image: PIL.Image) -> PIL.Image:
        """Detect and correct skew in scanned documents."""
        img_array = np.array(image.convert('L'))
        edges = cv2.Canny(img_array, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is not None and len(lines) > 0:
            angles = []
            for rho, theta in lines[:20, 0]:
                angle = np.degrees(theta) - 90
                if abs(angle) < 45:
                    angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                if abs(median_angle) > 0.5:
                    print(f"  🔄 Detected skew: {median_angle:.2f}°, correcting...")
                    return image.rotate(median_angle, expand=True, fillcolor='white')
        
        return image
    
    def _get_medical_prompt(self) -> str:
        """Get specialized prompt for medical document OCR."""
        return """You are an expert OCR system specialized in extracting text from historical medical and anatomical documents, specifically spine-related research materials.

**Document Context**: This is a historical medical/scientific document about spinal anatomy, pathology, or treatment. It may contain:
- Old German,Thai,Tamil, Latin, French,English or any language medical terminology
- Anatomical nomenclature (vertebrae, discs, ligaments, nerves)
- Clinical observations and case studies
- Measurement data and statistics
- Author names, institutional affiliations, publication dates
- References to other medical literature

**OCR Instructions**:
1. **Preserve Medical Terminology**: Extract medical and anatomical terms EXACTLY as written
2. **Document Structure**: Maintain original structure with proper markdown
3. **Handle Old Typography**: Recognize old typefaces, ligatures, faded text
4. **Special Elements**: Extract figure/table captions, preserve numerical data
5. **Quality Markers**: Note unclear text, illegible sections, confidence levels

**Output Format**: Pure markdown, no code blocks, start immediately with extracted content.

Begin extraction:"""
    
    def _get_standard_prompt(self) -> str:
        """Get standard OCR prompt."""
        return """You are an expert OCR system specialized in extracting text from scanned academic papers. Extract all visible text while preserving document structure using markdown formatting. Do not wrap output in code blocks. Begin immediately with extracted text."""
    
    def extract_text_from_image(
        self,
        image: PIL.Image,
        use_preprocessing: bool = True,
        enhancement_level: str = "medium",
        medical_context: bool = True,
        save_debug_images: bool = False,
        page_num: Optional[int] = None
    ) -> str:
        """
        Extract text from a single image using OCR.
        
        Args:
            image: PIL Image object
            use_preprocessing: Apply image enhancement
            enhancement_level: "light", "medium", or "aggressive"
            medical_context: Use medical-specific prompting
            save_debug_images: Save preprocessed images for debugging
            page_num: Page number for debug filename
        
        Returns:
            Extracted text as string
        """
        try:
            original_image = image.copy()
            
            if use_preprocessing:
                print("  🔧 Preprocessing image...")
                image = self.detect_and_deskew(image)
                image = self.preprocess_image(image, enhancement_level)
                
                if save_debug_images and page_num is not None:
                    debug_dir = "debug_images"
                    os.makedirs(debug_dir, exist_ok=True)
                    image.save(f"{debug_dir}/page_{page_num}_preprocessed.png")
                    print(f"  💾 Debug image saved: {debug_dir}/page_{page_num}_preprocessed.png")
            
            prompt = self._get_medical_prompt() if medical_context else self._get_standard_prompt()
            
            response = self.model.generate_content([prompt, image], stream=True)
            response.resolve()
            
            extracted_text = response.text if response.text else ""
            
            if not extracted_text.strip() and use_preprocessing:
                print("  ⚠️  Empty result with preprocessing, retrying with original image...")
                retry_response = self.model.generate_content([prompt, original_image], stream=True)
                retry_response.resolve()
                extracted_text = retry_response.text if retry_response.text else ""
            
            if not extracted_text.strip():
                print("  ❌ WARNING: No text extracted from this image!")
                return "[No text could be extracted from this page]"
            
            return extracted_text
            
        except Exception as e:
            print(f"  ❌ ERROR during OCR API call: {e}")
            import traceback
            traceback.print_exc()
            return f"[Error during image processing: {e}]"
    
    def process_pdf(
        self,
        pdf_path: str,
        use_preprocessing: bool = True,
        enhancement_level: str = "medium",
        high_dpi: bool = True,
        medical_context: bool = True,
        save_debug_images: bool = False,
        try_native_text: bool = True
    ) -> str:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            use_preprocessing: Apply image enhancement
            enhancement_level: "light", "medium", or "aggressive"
            high_dpi: Use 300 DPI for better quality
            medical_context: Use medical-specific OCR prompting
            save_debug_images: Save preprocessed images
            try_native_text: Try extracting native PDF text first
        
        Returns:
            Extracted text from all pages
        """
        all_text = ""
        try:
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            print(f"📄 Processing PDF with {total_pages} pages...")
            print(f"   Settings: DPI={'300' if high_dpi else '200'}, Enhancement={enhancement_level}, Preprocessing={use_preprocessing}")
            
            for page_num in range(total_pages):
                print(f"\n  📖 Page {page_num + 1}/{total_pages}")
                page = doc.load_page(page_num)
                
                native_text = ""
                if try_native_text:
                    native_text = page.get_text().strip()
                    if native_text and len(native_text) > 100:
                        print(f"  ✅ Extracted native PDF text (~{len(native_text)} characters)")
                        all_text += f"\n\n{'='*60}\n### Page {page_num + 1}\n{'='*60}\n\n{native_text}"
                        continue
                
                print(f"  🔍 No native text found, using OCR...")
                
                dpi = 300 if high_dpi else 200
                pix = page.get_pixmap(dpi=dpi)
                img = PIL.Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                if save_debug_images:
                    debug_dir = "debug_images"
                    os.makedirs(debug_dir, exist_ok=True)
                    img.save(f"{debug_dir}/page_{page_num + 1}_original.png")
                
                page_text = self.extract_text_from_image(
                    img,
                    use_preprocessing=use_preprocessing,
                    enhancement_level=enhancement_level,
                    medical_context=medical_context,
                    save_debug_images=save_debug_images,
                    page_num=page_num + 1
                )
                
                all_text += f"\n\n{'='*60}\n### Page {page_num + 1}\n{'='*60}\n\n{page_text}"
                
                print(f"  ✅ Extracted ~{len(page_text)} characters")
                
                import time
                time.sleep(1)
            
            doc.close()
            print(f"\n✅ PDF processing complete! Total characters: {len(all_text)}")
            return all_text
        except Exception as e:
            print(f"ERROR during PDF processing: {e}")
            import traceback
            traceback.print_exc()
            return f"An error occurred during PDF processing: {e}"
    
    def process_image(
        self,
        image_path: str,
        use_preprocessing: bool = True,
        enhancement_level: str = "medium",
        medical_context: bool = True,
        save_debug_images: bool = False
    ) -> str:
        """
        Extract text from image file.
        
        Args:
            image_path: Path to image file
            use_preprocessing: Apply image enhancement
            enhancement_level: "light", "medium", or "aggressive"
            medical_context: Use medical-specific prompting
            save_debug_images: Save preprocessed images
        
        Returns:
            Extracted text
        """
        try:
            print(f"🖼️  Processing image: {image_path}")
            img = PIL.Image.open(image_path)
            
            if save_debug_images:
                debug_dir = "debug_images"
                os.makedirs(debug_dir, exist_ok=True)
                img.save(f"{debug_dir}/original_image.png")
            
            text = self.extract_text_from_image(
                img,
                use_preprocessing=use_preprocessing,
                enhancement_level=enhancement_level,
                medical_context=medical_context,
                save_debug_images=save_debug_images,
                page_num=0
            )
            print("✅ Image processing complete!")
            return text
        except Exception as e:
            print(f"ERROR during image processing: {e}")
            import traceback
            traceback.print_exc()
            return f"An error occurred: {e}"
    
    def process_file(self, file_path: str, **kwargs) -> str:
        """
        Automatically detect file type and extract text.

        Args:
            file_path: Path to PDF or image file
            **kwargs: Additional arguments passed to process_pdf or process_image

        Returns:
            Extracted text
        """
        if not os.path.exists(file_path):
            return f"❌ Error: File not found at '{file_path}'"

        _, file_extension = os.path.splitext(file_path.lower())

        if file_extension == '.pdf':
            return self.process_pdf(file_path, **kwargs)
        elif file_extension in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif']:
            # Filter out PDF-only parameters before passing to process_image
            image_kwargs = {
                k: v for k, v in kwargs.items()
                if k in ['use_preprocessing', 'enhancement_level', 'medical_context', 'save_debug_images']
            }
            return self.process_image(file_path, **image_kwargs)
        else:
            return f"Unsupported file type for OCR: '{file_extension}'."


# Convenience function for backward compatibility
def create_ocr_engine(api_key: Optional[str] = None, use_advanced_model: bool = True) -> OCREngine:
    """Create and return an OCR engine instance."""
    return OCREngine(api_key=api_key, use_advanced_model=use_advanced_model)