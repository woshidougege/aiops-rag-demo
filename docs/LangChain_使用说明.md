# AIOps RAG Demo - LangChain 版本使用说明

## 🎯 改造概述

本项目已从原生 API 调用改造为基于 **LangChain 框架**的实现，提升了代码的可维护性和扩展性。

## 🔄 主要变化

### 1. 框架升级
- **原实现**：手动调用 API，自己实现向量化和相似度计算
- **新实现**：使用 LangChain 统一接口，标准化 RAG 流程

### 2. 核心组件

#### LLM 调用
```python
# 原方式：手动 requests
response = requests.post(api_url, json={...})

# LangChain 方式
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(base_url=..., api_key=..., model=...)
response = llm.invoke("your prompt")
```

#### Embeddings
```python
# 原方式：手动调用 API
response = requests.post(embedding_url, json={...})
vector = response.json()['data'][0]['embedding']

# LangChain 方式
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(base_url=..., api_key=..., model=...)
vector = embeddings.embed_query("text")
```

#### 向量存储
```python
# 原方式：手动计算余弦相似度
def cosine_similarity(vec1, vec2):
    # ... 手动实现

# LangChain 方式
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(documents, embeddings)
results = vectorstore.similarity_search("query", k=3)
```

#### RAG Chain
```python
# 原方式：手动组装流程
cases = search(query)
prompt = build_prompt(query, cases)
response = call_llm(prompt)

# LangChain 方式：声明式 Chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

diagnosis_chain = (
    prompt_template 
    | llm 
    | StrOutputParser()
)
result = diagnosis_chain.invoke({"error_log": log})
```

## 📦 新增依赖

```txt
# LangChain 核心
langchain==0.1.0
langchain-community==0.0.10
langchain-core==0.1.10

# LangChain 集成
langchain-openai==0.0.2
langchain-milvus==0.0.2

# 向量存储
faiss-cpu==1.7.4

# 文档处理
pypdf==3.17.4
python-docx==1.1.0
```

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置文件
`config.py` 保持不变，LangChain 兼容 OpenAI 格式的 API：
```python
LLM_CONFIG = {
    "api_base": "http://192.168.20.67:3000/v1",
    "api_key": "sk-xxx",
    "model": "Qwen-32B",
    "temperature": 0.7,
    "max_tokens": 2000
}

EMBEDDING_CONFIG = {
    "api_url": "https://api.siliconflow.cn/v1/embeddings",
    "api_key": "sk-xxx",
    "model": "BAAI/bge-m3",
    "dimension": 1024
}
```

### 3. 启动服务
```bash
cd aiops_demo
python app_simple.py
```

访问：
- Web 界面：http://localhost:8888
- API 文档：http://localhost:8888/docs

## 🔍 核心功能

### 1. 向量检索
```python
# 自动向量化 + 相似度搜索
results = vectorstore.similarity_search_with_score(query, k=3)

# 返回格式
[
    (Document(page_content="...", metadata={...}), score),
    ...
]
```

### 2. 提示模板
```python
diagnosis_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位资深的 AIOps 运维专家..."),
    ("user", "请分析以下故障：{error_log}\n相似案例：{retrieved_cases}")
])
```

### 3. 链式调用
```python
# LCEL (LangChain Expression Language)
chain = prompt | llm | output_parser
result = chain.invoke({"error_log": "..."})
```

## 📊 性能对比

| 指标 | 原实现 | LangChain 版 |
|------|--------|-------------|
| 代码行数 | 241行 | 241行（更清晰） |
| 向量检索 | 手动实现 | FAISS 优化 |
| 可扩展性 | 低 | 高 |
| 可维护性 | 中 | 高 |
| 调试难度 | 高 | 低（标准接口） |

## 🌟 优势

### 1. 标准化
- 统一的 LLM 调用接口
- 标准的向量存储操作
- 规范的 Document 数据结构

### 2. 可扩展
- 轻松切换向量库（FAISS → Milvus → Pinecone）
- 支持多种 LLM（OpenAI、Azure、本地模型）
- 灵活的 Chain 组合

### 3. 社区支持
- 丰富的文档和示例
- 活跃的社区
- 持续更新的集成库

### 4. 降级方案
- 向量存储失败 → 简单关键词匹配
- LLM 调用失败 → 返回最相似案例
- 健壮的错误处理

## 🔧 扩展示例

### 使用 Milvus 持久化存储
```python
from langchain_community.vectorstores import Milvus

vectorstore = Milvus(
    embedding_function=embeddings,
    connection_args={
        "host": "192.168.1.65",
        "port": "19530"
    },
    collection_name="aiops_knowledge_v1"
)
```

### 添加 Reranker
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)
```

### 使用 Agent
```python
from langchain.agents import create_openai_functions_agent
from langchain.tools import Tool

tools = [
    Tool(
        name="RAG搜索",
        func=rag_engine.search,
        description="搜索历史故障案例"
    )
]

agent = create_openai_functions_agent(llm, tools, prompt)
```

## 📚 参考资料

- [LangChain 官方文档](https://python.langchain.com/)
- [LangChain RAG 教程](https://python.langchain.com/docs/use_cases/question_answering/)
- [FAISS 向量库](https://github.com/facebookresearch/faiss)
- [Milvus 向量数据库](https://milvus.io/)

## 🐛 常见问题

### Q1: 向量存储初始化失败？
**A:** 检查 `data/knowledge_base.json` 是否存在，系统会自动降级到简单匹配。

### Q2: LLM 调用超时？
**A:** 增加 `max_tokens` 或调整 `temperature`，系统有降级方案返回历史案例。

### Q3: 如何切换到 Milvus？
**A:** 将 `FAISS.from_documents` 替换为 `Milvus.from_documents`，配置连接参数即可。

## 📝 TODO

- [ ] 添加混合检索（BM25 + 向量）
- [ ] 集成 Reranker 二次排序
- [ ] 支持流式输出
- [ ] 添加 Agent 工具调用
- [ ] 完善单元测试

---

**更新时间**：2024-11-19  
**版本**：v2.0 (LangChain)
