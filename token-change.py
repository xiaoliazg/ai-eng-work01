import os
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from dotenv import load_dotenv

load_dotenv()

# 增加调试日志
# import logging
# import sys
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger("llama_index").addHandler(logging.StreamHandler(stream=sys.stdout))

Settings.llm = OpenAILike(
    model="qwen3.6-flash",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASH_SCOPE_API_KEY"),
    is_chat_model=True
)

Settings.embed_model = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
    embed_batch_size=6,
    embed_input_length=8192
)

documents = SimpleDirectoryReader("data").load_data()

sentence_splitter = SentenceSplitter(
    chunk_size=256,
    chunk_overlap=50
)
nodes = sentence_splitter.get_nodes_from_documents(documents)
for node in nodes:
    print(node.text)
    print(node.metadata)





