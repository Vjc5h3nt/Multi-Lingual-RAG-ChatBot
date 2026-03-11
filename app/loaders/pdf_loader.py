import unicodedata
from typing import List, Optional
from pypdf import PdfReader
from app.loaders.base_loader import BaseLoader
from app.models import Document


DEFAULT_OCR_LANGUAGES = [
    "eng",
    "hin",
    "tel",
    "tam",
    "rus",
    "ukr",
    "por",
    "spa",
    "fra",
    "deu",
    "ita",
]


class PDFLoader(BaseLoader):
    def __init__(self, force_ocr: bool = False, ocr_languages: Optional[List[str]] = None):
        self.force_ocr = force_ocr
        self.ocr_languages = ocr_languages or DEFAULT_OCR_LANGUAGES

    def load(self, file_path: str) -> List[Document]:
        reader = PdfReader(file_path)
        documents: List[Document] = []

        empty_pages = 0
        low_quality_pages = 0

        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
            except Exception:
                text = None

            if text and text.strip():
                if self._looks_like_garbage_text(text):
                    low_quality_pages += 1
                else:
                    documents.append(
                        Document(
                            content=text,
                            metadata={
                                "source": file_path,
                                "page": page_num + 1,
                                "ocr": False
                            }
                        )
                    )
                    continue
            else:
                empty_pages += 1

        should_run_ocr = self.force_ocr
        if len(reader.pages) > 0 and not should_run_ocr:
            weak_page_count = empty_pages + low_quality_pages
            should_run_ocr = weak_page_count >= max(1, int(0.3 * len(reader.pages)))

        if should_run_ocr:
            reason = "force_ocr=True" if self.force_ocr else "weak extracted text detected"
            print(
                f"[OCR] {reason} for: {file_path}. Running OCR with languages: "
                f"{'+'.join(self.ocr_languages)}"
            )
            documents = self._ocr_pdf(file_path)

        return documents

    def _looks_like_garbage_text(self, text: str) -> bool:
        cleaned = " ".join(text.split())
        if len(cleaned) < 40:
            return True

        meaningful_chars = 0
        garbage_chars = 0

        for char in cleaned:
            if char.isspace():
                continue

            category = unicodedata.category(char)
            if char.isalnum() or category.startswith("L") or category.startswith("N"):
                meaningful_chars += 1
            elif category.startswith("P"):
                meaningful_chars += 1
            elif category.startswith("S") or category.startswith("C"):
                garbage_chars += 1

        total_chars = meaningful_chars + garbage_chars
        if total_chars == 0:
            return True

        garbage_ratio = garbage_chars / total_chars
        return garbage_ratio > 0.3

    def _ocr_pdf(self, file_path: str) -> List[Document]:
        from pdf2image import convert_from_path
        import pytesseract
        import platform

        ocr_language_string = "+".join(self.ocr_languages)

        # Platform-specific paths
        if platform.system() == "Windows":
            # Explicit paths for Windows installation
            pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract\tesseract.exe"
            poppler_path = r"D:\poppler-25.12.0\Library\bin"
            pages = convert_from_path(
                file_path,
                dpi=300,
                poppler_path=poppler_path
            )
        else:
            # macOS/Linux: use system-installed tesseract and poppler
            # (installed via brew install tesseract poppler)
            pages = convert_from_path(
                file_path,
                dpi=300
            )

        docs: List[Document] = []

        for i, img in enumerate(pages):
            text = pytesseract.image_to_string(img, lang=ocr_language_string)
            if text and text.strip():
                docs.append(
                    Document(
                        content=text,
                        metadata={
                            "source": file_path,
                            "page": i + 1,
                            "ocr": True,
                            "ocr_languages": self.ocr_languages,
                        }
                    )
                )

        return docs
