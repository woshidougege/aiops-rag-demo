"""
AIOps RAG Demo - LangChain 框架版本
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
import os
import json

# LangChain 核心组件
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import LLM_CONFIG, EMBEDDING_CONFIG, MILVUS_CONFIG, RERANKER_CONFIG


# ===================================================================
# 初始化
# ===================================================================
app = FastAPI(
    title="AIOps RAG Demo",
    description="智能运维故障诊断系统",
    version="1.0.0"
)

print("🚀 启动 AIOps RAG Demo...")


# ===================================================================
# Reranker 精排器
# ===================================================================
class SiliconFlowReranker:
    """SiliconFlow API 的 Reranker"""
    
    def __init__(self):
        self.api_url = RERANKER_CONFIG['api_url']
        self.api_key = RERANKER_CONFIG['api_key']
        self.model = RERANKER_CONFIG['model']
        self.top_n = RERANKER_CONFIG['top_n']
    
    def rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        """对检索结果进行精排"""
        if not documents:
            return documents
        
        try:
            import requests
            # 准备文档文本
            texts = [doc.get('content', '') or f"{doc.get('error_type', '')} {doc.get('root_cause', '')}" 
                    for doc in documents]
            
            # 调用 Reranker API
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "query": query,
                    "documents": texts,
                    "top_n": self.top_n
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                # 根据 Reranker 返回的索引重新排序
                reranked = []
                for item in result.get('results', [])[:self.top_n]:
                    idx = item['index']
                    doc = documents[idx].copy()
                    doc['rerank_score'] = item['relevance_score']
                    reranked.append(doc)
                print(f"  ✓ Reranker 精排完成，返回 Top-{len(reranked)}")
                return reranked
            else:
                print(f"  ⚠ Reranker 调用失败: {response.status_code}")
                return documents[:self.top_n]
        
        except Exception as e:
            print(f"  ⚠ Reranker 异常: {e}")
            return documents[:self.top_n]


# ===================================================================
# RAG引擎（LangChain 完整版：LLM + Embedding + Reranker + Milvus）
# ===================================================================
class LangChainRAGEngine:
    """基于 LangChain 的 RAG 引擎"""
    
    def __init__(self):
        print("🚀 初始化 LangChain RAG 引擎...")
        
        # 初始化 LLM
        self.llm = ChatOpenAI(
            base_url=LLM_CONFIG['api_base'],
            api_key=LLM_CONFIG['api_key'],
            model=LLM_CONFIG['model'],
            temperature=LLM_CONFIG['temperature'],
            max_tokens=LLM_CONFIG['max_tokens']
        )
        print(f"✓ LLM 已初始化: {LLM_CONFIG['model']}")
        
        # 初始化 Embeddings
        self.embeddings = OpenAIEmbeddings(
            base_url=EMBEDDING_CONFIG['api_url'].replace('/embeddings', ''),
            api_key=EMBEDDING_CONFIG['api_key'],
            model=EMBEDDING_CONFIG['model']
        )
        print(f"✓ Embedding 已初始化: {EMBEDDING_CONFIG['model']}")
        
        # 初始化 Reranker（如果启用）
        if RERANKER_CONFIG.get('enabled', False):
            self.reranker = SiliconFlowReranker()
            print(f"✓ Reranker 已初始化: {RERANKER_CONFIG['model']}")
        else:
            self.reranker = None
            print("⚠ Reranker 未启用")
        
        # 加载知识库并初始化向量存储
        self.vectorstore = None
        self.knowledge_base = []
        self.load_knowledge()
        
        # 创建诊断提示模板
        self.diagnosis_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一位资深的 AIOps 运维专家，擅长分析系统故障并提供解决方案。"),
            ("user", """请分析以下故障并以JSON格式输出诊断结果。

【当前故障】
{error_log}

【历史相似案例】
{retrieved_cases}

【输出要求】
请严格按照以下JSON格式输出（不要有其他文字）：
{{
  "diagnosis": "故障诊断（一句话概括）",
  "root_cause": "根本原因分析（深入技术细节）",
  "solution": "解决方案（分步骤，可执行）",
  "confidence": 0.85
}}""")
        ])
        
        # 创建诊断链
        self.diagnosis_chain = (
            self.diagnosis_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        print("✓ RAG 引擎初始化完成\n")
    
    def load_knowledge(self):
        """加载知识库到向量存储"""
        kb_path = "data/knowledge_base.json"
        if os.path.exists(kb_path):
            with open(kb_path, 'r', encoding='utf-8') as f:
                self.knowledge_base = json.load(f)
            print(f"✓ 加载了 {len(self.knowledge_base)} 条知识案例")
            
            # 转换为 LangChain Documents
            documents = []
            for case in self.knowledge_base:
                content = f"""错误类型: {case['error_type']}
日志内容: {case.get('log_content', '')}
根本原因: {case['root_cause']}
解决方案: {case['solution']}"""
                
                metadata = {
                    "error_type": case['error_type'],
                    "root_cause": case['root_cause'],
                    "solution": case['solution'],
                    "severity": case.get('severity', 'medium')
                }
                documents.append(Document(page_content=content, metadata=metadata))
            
            # 优先使用 Milvus，失败则降级到 FAISS
            try:
                from langchain_milvus import Milvus
                from pymilvus import connections
                
                print(f"🔄 尝试连接 Milvus: {MILVUS_CONFIG['host']}:{MILVUS_CONFIG['port']}...")
                
                # 先建立连接
                connections.connect(
                    alias="default",
                    host=MILVUS_CONFIG['host'],
                    port=int(MILVUS_CONFIG['port']),
                    timeout=10
                )
                print("  ✓ Milvus 连接成功")
                
                # 创建向量存储
                self.vectorstore = Milvus.from_documents(
                    documents,
                    self.embeddings,
                    collection_name=MILVUS_CONFIG['collection_name'],
                    connection_args={"alias": "default"},
                    drop_old=True  # 重新创建collection
                )
                print(f"✓ 向量存储已创建（使用 Milvus: {MILVUS_CONFIG['host']}:{MILVUS_CONFIG['port']}）")
            except Exception as e:
                print(f"⚠ Milvus 连接失败: {e}")
                print("🔄 降级使用 FAISS...")
                try:
                    self.vectorstore = FAISS.from_documents(documents, self.embeddings)
                    print("✓ 向量存储已创建（使用 FAISS 本地存储）")
                except Exception as e2:
                    print(f"⚠ FAISS 创建失败: {e2}")
                    self.vectorstore = None
        else:
            print(f"⚠ 知识库文件不存在: {kb_path}")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """检索相似案例"""
        print(f"🔍 检索: {query[:50]}...")
        
        if not self.vectorstore:
            print("⚠ 向量存储未初始化，使用简单匹配")
            return self.simple_search(query, top_k)
        
        try:
            # 使用 LangChain 的相似度搜索
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)
            
            results = []
            for doc, score in docs_with_scores:
                # 转换分数为相似度 (FAISS 返回的是距离，越小越相似)
                similarity = 1.0 / (1.0 + score)
                results.append({
                    "error_type": doc.metadata.get('error_type', '未知'),
                    "root_cause": doc.metadata.get('root_cause', ''),
                    "solution": doc.metadata.get('solution', ''),
                    "severity": doc.metadata.get('severity', 'medium'),
                    "similarity": round(similarity, 3),
                    "content": doc.page_content
                })
            
            print(f"✓ 向量检索找到 {len(results)} 个相似案例")
            
            # 使用 Reranker 精排（如果启用）
            if self.reranker and RERANKER_CONFIG.get('enabled', False):
                print("🔄 使用 Reranker 进行精排...")
                results = self.reranker.rerank(query, results)
            
            return results
            
        except Exception as e:
            print(f"⚠ 向量检索失败: {e}")
            return self.simple_search(query, top_k)
    
    def simple_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """简单的关键词匹配搜索（降级方案）"""
        results = []
        query_lower = query.lower()
        
        for case in self.knowledge_base:
            case_text = f"{case['error_type']} {case.get('log_content', '')} {case['root_cause']}".lower()
            # 简单的关键词匹配
            common_words = set(query_lower.split()) & set(case_text.split())
            similarity = len(common_words) / max(len(query_lower.split()), 1)
            
            if similarity > 0:
                results.append({
                    **case,
                    "similarity": round(similarity, 3)
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def diagnose(self, error_log: str) -> Dict:
        """完整诊断流程"""
        print("🔮 开始诊断...")
        
        # 1. RAG 检索相似案例
        cases = self.search(error_log, top_k=3)
        
        # 2. 格式化检索结果
        if cases:
            history_text = ""
            for i, case in enumerate(cases, 1):
                history_text += f"\n案例{i}（相似度: {case.get('similarity', 0):.2f}）:\n"
                history_text += f"  错误类型: {case['error_type']}\n"
                history_text += f"  根本原因: {case['root_cause']}\n"
                history_text += f"  解决方案: {case['solution']}\n"
        else:
            history_text = "无相似历史案例"
        
        # 3. 使用 LangChain 链进行诊断
        try:
            print("🤖 调用 LLM 生成诊断...")
            response = self.diagnosis_chain.invoke({
                "error_log": error_log,
                "retrieved_cases": history_text
            })
            
            print(f"📝 [DEBUG] LLM 原始响应:\n{response[:500]}...")  # 只打印前500字符
            
            # 4. 解析 JSON 结果
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)
            
            print(f"📝 [DEBUG] 解析后的结果: {result}")
            print(f"📝 [DEBUG] confidence 字段: {result.get('confidence')} (类型: {type(result.get('confidence'))})")
            
            result['retrieved_cases'] = cases
            print("✓ 诊断完成")
            return result
            
        except Exception as e:
            print(f"⚠ LLM 调用失败: {e}")
            # 降级方案：使用第一个相似案例
            if cases:
                return {
                    "diagnosis": f"基于历史案例的诊断：{cases[0]['error_type']}",
                    "root_cause": cases[0]['root_cause'],
                    "solution": cases[0]['solution'],
                    "confidence": cases[0].get('similarity', 0.5),
                    "retrieved_cases": cases
                }
            else:
                return {
                    "diagnosis": "无法自动诊断",
                    "root_cause": "未找到相似案例，建议人工排查",
                    "solution": "请提供更多日志信息或联系运维团队",
                    "confidence": 0.0,
                    "retrieved_cases": []
                }


# 初始化RAG引擎
rag_engine = LangChainRAGEngine()


# ===================================================================
# 数据模型
# ===================================================================
class DiagnosisRequest(BaseModel):
    error_log: str
    top_k: Optional[int] = 3


class DiagnosisResponse(BaseModel):
    success: bool
    diagnosis: str
    root_cause: str
    solution: str
    confidence: float
    retrieved_cases: list


# ===================================================================
# API路由
# ===================================================================
@app.get("/", response_class=HTMLResponse)
async def index():
    """主页"""
    html_path = "templates/index.html"
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>AIOps RAG Demo</h1><p><a href='/docs'>API文档</a></p>")


@app.get("/health")
async def health():
    return {"status": "healthy", "knowledge_base": len(rag_engine.knowledge_base)}


@app.post("/api/diagnose", response_model=DiagnosisResponse)
async def diagnose(request: DiagnosisRequest):
    """诊断API"""
    try:
        result = rag_engine.diagnose(request.error_log)
        
        print(f"📝 [DEBUG] RAG引擎返回的结果: {result}")
        print(f"📝 [DEBUG] retrieved_cases 数量: {len(result.get('retrieved_cases', []))}")
        
        # 安全地转换 confidence
        raw_confidence = result.get('confidence', 0.5)
        try:
            confidence_value = float(raw_confidence)
        except (ValueError, TypeError) as e:
            print(f"⚠ [DEBUG] confidence 转换失败: {raw_confidence}, 错误: {e}")
            confidence_value = 0.5
        
        print(f"📝 [DEBUG] confidence 最终值: {confidence_value}")
        
        # 安全地处理 retrieved_cases
        safe_cases = []
        for i, c in enumerate(result.get('retrieved_cases', [])):
            print(f"📝 [DEBUG] 处理案例 {i}: {c}")
            safe_cases.append({
                "error_type": c.get('error_type', '未知'),
                "similarity": float(c.get('similarity', 0)),
                "root_cause": c.get('root_cause', ''),
                "solution": c.get('solution', '')
            })
        
        # 处理 solution 字段：可能是 list 或 str
        raw_solution = result.get('solution', '')
        if isinstance(raw_solution, list):
            solution_str = '\n'.join(str(s) for s in raw_solution)
        else:
            solution_str = str(raw_solution)
        
        response = DiagnosisResponse(
            success=True,
            diagnosis=result.get('diagnosis', ''),
            root_cause=result.get('root_cause', ''),
            solution=solution_str,
            confidence=confidence_value,
            retrieved_cases=safe_cases
        )
        
        print("✅ [DEBUG] DiagnosisResponse 构造成功")
        return response
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ [DEBUG] API 异常:\n{error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


# ===================================================================
# 启动
# ===================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🌐 访问地址: http://localhost:8888")
    print("📚 API文档: http://localhost:8888/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8888, log_level="info")
