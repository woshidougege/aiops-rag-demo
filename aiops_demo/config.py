"""
配置文件 - LangChain 框架配置
使用 OpenAI 兼容的 API 接口
"""

# LLM服务配置
LLM_CONFIG = {
    "api_base": "http://192.168.20.67:3000/v1",
    "api_key": "sk-JmkUHwNVXkX6VxuXC819A7De47F24d8a87FaC93e0f03F1A0",
    "model": "Qwen-32B",
    "temperature": 0.7,
    "max_tokens": 2000
}

# 向量模型配置（使用SiliconFlow API）
EMBEDDING_CONFIG = {
    "api_url": "https://api.siliconflow.cn/v1/embeddings",
    "api_key": "sk-hbosbxnyalzntoorrqrbjsohpqakykysjxfebkfdupcyjizi",
    "model": "BAAI/bge-m3",
    "dimension": 1024  # BGE-M3的向量维度
}

# Milvus配置
MILVUS_CONFIG = {
    "host": "192.168.1.65",
    "port": "19530",
    "collection_name": "aiops_knowledge_v1"
}
