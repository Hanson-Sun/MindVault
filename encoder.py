from typing import Union, List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
    NLTKTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from nltk.tokenize import sent_tokenize
import os
import json
from tqdm import tqdm

class Encoder:
    def __init__(self, model):
        self.model = model

    # this returns a list of documents
    def load_PDF(self, path : str):
        loader = PyPDFLoader(path)
        pages = loader.load_and_split()
        return pages

    def load_text(self, path : str):
        loader = TextLoader(path)
        data = loader.load()
        return data

    def load_markdown(self, path : str):
        loader = UnstructuredMarkdownLoader(path);
        data = loader.load_and_split();
        return data

    # returns a single document
    def load_code(self, path : str):
        load = TextLoader(path)
        data = load.load()
        return data

    # figure out how to load images and store the proper metadata
    def load_image(self, path : str):
        pass

    # this processes a list of documents
    def chunkify_PDF(self, data, chunk_sentences = 5):
        for i in range(len(data)):
            sentences = sent_tokenize(data[i].page_content)
            data[i].page_content = [' '.join(sentences[n:n+chunk_sentences]) for n in range(0, len(sentences) + chunk_sentences, chunk_sentences)]
        return data

    def chunkify_markdown(self, data, chunk_sentences = 5):
        data = data[0]
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2")
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        large_split = markdown_splitter.split_text(data.page_content)
        for i in range(len(large_split)):
            sentences = sent_tokenize(large_split[i].page_content)
            large_split[i].page_content = [' '.join(sentences[n:n+chunk_sentences]) for n in range(0, len(sentences) + chunk_sentences, chunk_sentences)]
            large_split[i].metadata = {**large_split[i].metadata, **data.metadata}

        return large_split

    def chunkify_code(self, data, language_type:Language, chunk_sentences = 5):
        data = data[0]
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language_type, chunk_size= chunk_sentences * 100, chunk_overlap=0
        )
        large_split = splitter.split_text(data.page_content)
        document = Document(page_content="", metadata={**data.metadata})

        document.page_content = large_split
        return [document]

    def get_programming_language(self, file_extension : str) -> Union[Language, bool]:
        switcher = {
            ".h" : Language.CPP,
            ".cpp": Language.CPP,
            ".go": Language.GO,
            ".java": Language.JAVA,
            ".kt": Language.KOTLIN,
            ".js": Language.JS,
            ".ts": Language.TS,
            ".php": Language.PHP,
            ".proto": Language.PROTO,
            ".py": Language.PYTHON,
            ".rst": Language.RST,
            ".rb": Language.RUBY,
            ".rs": Language.RUST,
            ".scala": Language.SCALA,
            ".swift": Language.SWIFT,
            ".md": Language.MARKDOWN,
            ".tex": Language.LATEX,
            ".html": Language.HTML,
            ".sol": Language.SOL,
            ".cs": Language.CSHARP,
            ".cbl": Language.COBOL,
            ".c": Language.CPP,
            ".lua": Language.LUA,
            ".pl": Language.PERL,
        }
        return switcher.get(file_extension, False)

    def tokenize(self, content : str):
        return sent_tokenize(content)
    
    def encode_chunks(self, sentences : list):
        if sentences and all(isinstance(sentence, str) and sentence.strip() for sentence in sentences):
            return self.model.encode(sentences)
        else:
            print("Warning: Attempted to encode empty or invalid sentences.")
            return []
    
    def process_file(self, file_path) -> Tuple[List, List, List, bool]:
        encoded = []
        pages = []
        chunks = []
        if (file_path.endswith(".pdf")):
            try:
                pages = self.load_PDF(file_path)
            except Exception as e:
                print(f"ERROR: {e}")
                return ([], [], [], False)
            chunks = self.chunkify_PDF(pages)
            for i in tqdm(range(len(chunks)), desc="Encoding chunks"):
                encoded.append(self.encode_chunks(chunks[i].page_content))
        elif (file_path.endswith(".md")):
            try:
                pages = self.load_markdown(file_path)
            except Exception as e:
                print(f"ERROR: {e}")
                return ([], [], [], False)
            chunks = self.chunkify_markdown(pages)
            for i in tqdm(range(len(chunks)), desc="Encoding chunks"):
                encoded.append(self.encode_chunks(chunks[i].page_content))
        elif (language := self.get_programming_language(os.path.splitext(file_path)[1])):
            try:
                pages = self.load_code(file_path)
            except Exception as e:
                print(f"ERROR: {e}")
                return ([], [], [], False)
            chunks = self.chunkify_code(pages, language)
            for i in tqdm(range(len(chunks)), desc="Encoding chunks"):
                encoded.append(self.encode_chunks(chunks[i].page_content))
        else:
            tqdm.write("Unsupported file type")
            return ([], [], [], False)
        
        return (encoded, pages, chunks, True)