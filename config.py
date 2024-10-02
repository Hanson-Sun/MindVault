from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer
from torch import cuda
import torch

# Check if CUDA is available and set the default device accordingly
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.set_device(0)  # Sets the default CUDA device to the first GPU (index 0)
else:
    device = torch.device("cpu")

INDEX_IGNORE_NAME = ".indexignore"

DATABASE_NAME = "MindVault.db"

LOCAL_DIRECTORY_PATH = "./" 

SERVER_PORT = 5000

LLM_MODEL_NAME = "microsoft/phi-2"

# can use formatted_string = template.format(**values)
LLM_INSTRUCT_FORMAT = "Instruct: {instruct}\nSystem: {system}\nResponse: {response}"

# okay i should abstract this into a function so its easy to change
SENTENCE_ENCODING_MODEL = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
if cuda.is_available():
    SENTENCE_ENCODING_MODEL = SENTENCE_ENCODING_MODEL.to(device)

CROSS_ENCODER_MODEL = CrossEncoder('cross-encoder/msmarco-MiniLM-L12-en-de-v1')

CROSS_ENCODER_TOKENIZER = AutoTokenizer.from_pretrained('cross-encoder/msmarco-MiniLM-L12-en-de-v1') 



