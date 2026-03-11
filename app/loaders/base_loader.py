from abc import ABC, abstractmethod
from typing import List
from app.models import Document

class BaseLoader(ABC):

    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """
        Load a file and return a list of Document objects.
        """
        pass
