from typing import List
from pypdf import PdfReader
from app.loaders.base_loader import BaseLoader
from app.models import Document

class PDFLoader(BaseLoader):

    def load(self, file_path: str) -> List[Document]:
        reader = PdfReader(file_path)
        documents: List[Document] = []

        empty_pages = 0

        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
            except Exception:
                text = None

            if text and text.strip():
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
            else:
                empty_pages += 1

        # If many pages have no selectable text → run OCR
        if len(reader.pages) > 0 and empty_pages >= max(1, int(0.3 * len(reader.pages))):
            print(f"[OCR] Detected scanned PDF: {file_path}. Running OCR...")
            documents.extend(self._ocr_pdf(file_path))

        return documents

    def _ocr_pdf(self, file_path: str) -> List[Document]:
        from pdf2image import convert_from_path
        import pytesseract
        import platform

        # 🔧 Platform-specific paths
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
            text = pytesseract.image_to_string(img, lang="eng+tel+hin")
            if text and text.strip():
                docs.append(
                    Document(
                        content=text,
                        metadata={
                            "source": file_path,
                            "page": i + 1,
                            "ocr": True
                        }
                    )
                )

        return docs
