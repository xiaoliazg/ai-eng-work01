from typing import List

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
from paddleocr import PaddleOCR


class MyImageLoader(BaseReader):
    def load_data(self, file_path: str, **kwargs) -> List[Document]:
        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )

        result = ocr.predict(file_path)
        extracted_text = result[0]["rec_texts"]
        text_str = "\n".join(extracted_text)
        print(text_str)
        doc = Document(
            text=text_str,
            metadata={"file_path": file_path}
        )
        return [doc]


loader = MyImageLoader()
documents = loader.load_data("./data/066F16B9-3D72-4C7C-A19D-E14B0732EC72_1_105_c.jpeg")

print(documents)
