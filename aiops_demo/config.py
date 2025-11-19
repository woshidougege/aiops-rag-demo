"""
配置文件 - AIOps RAG 完整配置
包含：文本大模型 + 向量模型 + 精排模型 + Milvus 向量数据库
"""

# ===================================================================
# 1. 文本大模型配置（Qwen-32B）
# ===================================================================
LLM_CONFIG = {
    "provider": "proxy/openai",
    "api_base": "http://192.168.20.67:3000/v1",
    "api_key": "sk-JmkUHwNVXkX6VxuXC819A7De47F24d8a87FaC93e0f03F1A0",
    "model": "Qwen-32B",
    "temperature": 0.7,
    "max_tokens": 2000
}

# ===================================================================
# 2. 向量模型配置（BGE-M3）
# ===================================================================
EMBEDDING_CONFIG = {
    "provider": "proxy/openai",
    "api_url": "https://api.siliconflow.cn/v1/embeddings",
    "api_key": "sk-hbosbxnyalzntoorrqrbjsohpqakykysjxfebkfdupcyjizi",
    "model": "BAAI/bge-m3",
    "dimension": 1024  # BGE-M3的向量维度
}

# ===================================================================
# 3. 精排模型配置（BGE-Reranker-v2-M3）
# ===================================================================
RERANKER_CONFIG = {
    "enabled": True,  # 是否启用精排
    "provider": "proxy/siliconflow",
    "api_url": "https://api.siliconflow.cn/v1/rerank",
    "api_key": "sk-hbosbxnyalzntoorrqrbjsohpqakykysjxfebkfdupcyjizi",
    "model": "BAAI/bge-reranker-v2-m3",
    "top_n": 3  # 精排后返回的文档数量
}

# ===================================================================
# 4. Milvus 向量数据库配置
# ===================================================================
MILVUS_CONFIG = {
    "type": "Milvus",
    "uri": "192.168.1.65",
    "host": "192.168.1.65",
    "port": "19530",
    "collection_name": "aiops_knowledge_v1"
}
