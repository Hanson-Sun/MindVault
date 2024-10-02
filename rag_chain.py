import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import LLM_MODEL_NAME, LLM_INSTRUCT_FORMAT
from database import VectorDatabase

class LLMModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model, self.tokenizer = self.init_model(model_name)
        self.inference_model = self.inferenced_model(self.model, self.tokenizer)
    
    def init_model(self, model_name):
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return model, tokenizer

    def quantize_model(self, model):
        return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    def inferenced_model(self, model, quantize=True):
        model = self.model
        if quantize:
            model = self.quantize_model(model)
        model.eval()
        return model

    def generate(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", return_attention_mask=False)
        outputs = self.inference_model.generate(**inputs, max_length=200)
        text = self.tokenizer.batch_decode(outputs)[0]
        return text
    
    def generate(self, instruct, system, prompt, format):
        text = format.format(instruct=instruct, system=system, prompt=prompt)
        return self.generate(text)
    
class SearchAgent:
    def __init__(self, model, database : VectorDatabase):
        self.model = model
        self.database = database
        self.results = []
    
    def refine_query_vector(self, query : str):
        # okay so use the llm, ask it to refine a query
        generated_results = self.model.generate("", "", "", LLM_INSTRUCT_FORMAT)
        pass

    def refine_query_fulltext(self, query : str):
        # okay use the llm to refine a query for full text search
        generated_results = self.model.generate("", "", "", LLM_INSTRUCT_FORMAT)
        pass

    def hybrid_search(self, query : str, n : int):
        # combine both results
        vector_query = self.refine_query_vector(query)
        fulltext_query = self.refine_query_fulltext(query)

        vss_results = VectorDatabase.vss_search(vector_query, n)
        fts_results = VectorDatabase.fulltext_search(fulltext_query, n)

        self.results = VectorDatabase.cross_encoder_reranking(vss_results + fts_results, n)

        return self.results

class RAGChain:
    def __init__(self, model_name=LLM_MODEL_NAME):
        self.context_chain = []
        self.pipeline = []
        self.model = LLMModel(model_name)

    def add_to_pipeline(self, function):
        self.pipeline.append(function)

    def add(self, text):
        self.context_chain.append(text)

    def process_chain(self):
        return " ".join(self.chain)

    def generate(self, text, instruct_format=LLM_INSTRUCT_FORMAT):
        # text = self.process_chain() + text

        for function in self.pipeline:
            text = function(text)
    
        self.add(text)

        self.model.generate(text, self.process_chain(), "", instruct_format)
        return text



