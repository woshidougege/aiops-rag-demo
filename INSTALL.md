# AIOps RAG Demo - 安装和测试指南

## 📦 环境要求

- **Python**: 3.9+
- **系统**: Windows / Linux / macOS
- **内存**: 至少 2GB 可用
- **网络**: 需要访问 LLM 和 Embedding API

---

## 🚀 快速安装

### 1. 克隆项目

```bash
git clone https://github.com/YOUR_USERNAME/aiops-rag-demo.git
cd aiops-rag-demo
```

### 2. 创建虚拟环境（推荐）

#### Windows
```bash
python -m venv venv
.\venv\Scripts\activate
```

#### Linux/macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

**依赖包列表**：
```txt
# 核心框架
fastapi==0.104.1
uvicorn[standard]==0.24.0

# LangChain 生态
langchain==0.1.0
langchain-community==0.0.10
langchain-core==0.1.10
langchain-openai==0.0.2
langchain-milvus==0.0.2

# 向量存储
faiss-cpu==1.7.4
pymilvus==2.3.4

# 其他依赖
pydantic==2.5.0
requests==2.31.0
tiktoken==0.5.2
numpy==1.24.3
```

### 4. 配置服务

编辑 `aiops_demo/config.py`：

```python
"""
配置文件 - LangChain 框架配置
使用 OpenAI 兼容的 API 接口
"""

# LLM服务配置
LLM_CONFIG = {
    "api_base": "http://192.168.20.67:3000/v1",  # 你的 LLM API 地址
    "api_key": "sk-xxx",  # 你的 API Key
    "model": "Qwen-32B",
    "temperature": 0.7,
    "max_tokens": 2000
}

# 向量模型配置（使用SiliconFlow API）
EMBEDDING_CONFIG = {
    "api_url": "https://api.siliconflow.cn/v1/embeddings",
    "api_key": "sk-xxx",  # 你的 API Key
    "model": "BAAI/bge-m3",
    "dimension": 1024
}

# Milvus配置（可选）
MILVUS_CONFIG = {
    "host": "192.168.1.65",
    "port": "19530",
    "collection_name": "aiops_knowledge_v1"
}
```

### 5. 启动服务

```bash
cd aiops_demo
python app_simple.py
```

输出示例：
```
🚀 启动 AIOps RAG Demo...
🚀 初始化 LangChain RAG 引擎...
✓ LLM 已初始化: Qwen-32B
✓ Embedding 已初始化: BAAI/bge-m3
✓ 加载了 10 条知识案例
✓ 向量存储已创建（使用 FAISS）
✓ RAG 引擎初始化完成

============================================================
🌐 访问地址: http://localhost:8888
📚 API文档: http://localhost:8888/docs
============================================================
```

---

## 🧪 测试功能

### 1. 健康检查

```bash
curl http://localhost:8888/health
```

**预期响应**：
```json
{
  "status": "healthy",
  "knowledge_base": 10
}
```

### 2. API 测试

#### 测试案例 1: OOM 错误
```bash
curl -X POST "http://localhost:8888/api/diagnose" \
  -H "Content-Type: application/json" \
  -d '{
    "error_log": "java.lang.OutOfMemoryError: Java heap space at com.example.Service.process(Service.java:42)",
    "top_k": 3
  }'
```

#### 测试案例 2: 数据库连接池
```bash
curl -X POST "http://localhost:8888/api/diagnose" \
  -H "Content-Type: application/json" \
  -d '{
    "error_log": "Could not get JDBC Connection; nested exception is org.apache.commons.dbcp.SQLNestedException: Cannot get a connection, pool error Timeout waiting for idle object",
    "top_k": 3
  }'
```

#### 测试案例 3: Kubernetes Pod 崩溃
```bash
curl -X POST "http://localhost:8888/api/diagnose" \
  -H "Content-Type: application/json" \
  -d '{
    "error_log": "Back-off restarting failed container order-service in pod order-service-7d9f8b5c4-xk2z9",
    "top_k": 3
  }'
```

### 3. Web 界面测试

1. 浏览器访问：http://localhost:8888
2. 输入错误日志或点击示例按钮
3. 点击"开始诊断"
4. 查看诊断结果

---

## 🔧 测试脚本

使用内置的测试脚本验证服务连接：

```bash
cd aiops_demo
python test_services.py
```

**输出示例**：
```
============================================================
测试服务连接
============================================================

1. 测试LLM服务...
   ✓ LLM服务正常 (Qwen-32B)

2. 测试向量服务...
   ✓ 向量服务正常 (BAAI/bge-m3)
   向量维度: 1024

3. 测试Milvus...
   ⚠ Milvus未配置（使用FAISS本地存储）

============================================================
✓ 所有服务测试通过
============================================================
```

---

## 🐍 Python 测试代码

创建测试脚本 `test_api.py`：

```python
import requests
import json

# 测试API
def test_diagnose():
    url = "http://localhost:8888/api/diagnose"
    
    test_cases = [
        {
            "name": "OOM Error",
            "error_log": "java.lang.OutOfMemoryError: Java heap space"
        },
        {
            "name": "Database Connection",
            "error_log": "Could not get JDBC Connection; pool error Timeout"
        },
        {
            "name": "Redis Connection",
            "error_log": "redis.exceptions.ConnectionError: Error 111 connecting to localhost:6379. Connection refused."
        }
    ]
    
    for case in test_cases:
        print(f"\n测试案例: {case['name']}")
        print("="*60)
        
        response = requests.post(url, json={
            "error_log": case['error_log'],
            "top_k": 3
        })
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ 诊断成功")
            print(f"  故障: {result['diagnosis']}")
            print(f"  根因: {result['root_cause'][:100]}...")
            print(f"  置信度: {result['confidence']}")
            print(f"  相似案例数: {len(result['retrieved_cases'])}")
        else:
            print(f"✗ 请求失败: {response.status_code}")

if __name__ == "__main__":
    test_diagnose()
```

运行测试：
```bash
python test_api.py
```

---

## 📊 性能测试

### 使用 Apache Bench

```bash
# 安装 ab (Apache Bench)
# Ubuntu/Debian: sudo apt-get install apache2-utils
# macOS: 已自带

# 创建测试数据文件
cat > post_data.json << EOF
{
  "error_log": "java.lang.OutOfMemoryError: Java heap space",
  "top_k": 3
}
EOF

# 性能测试（100请求，10并发）
ab -n 100 -c 10 -p post_data.json -T application/json \
   http://localhost:8888/api/diagnose
```

### 使用 Python Locust

创建 `locustfile.py`：

```python
from locust import HttpUser, task, between

class AIOpsUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def diagnose(self):
        self.client.post("/api/diagnose", json={
            "error_log": "java.lang.OutOfMemoryError: Java heap space",
            "top_k": 3
        })

# 运行: locust -f locustfile.py
# 访问: http://localhost:8089
```

---

## 🐛 常见问题

### Q1: 依赖安装失败？

**问题**：`pip install` 报错

**解决**：
```bash
# 升级 pip
pip install --upgrade pip

# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q2: 向量存储初始化失败？

**问题**：`FAISS.from_documents` 报错

**解决**：
```bash
# 检查 faiss 安装
pip uninstall faiss-cpu
pip install faiss-cpu==1.7.4

# 或使用 GPU 版本（如果有 CUDA）
pip install faiss-gpu
```

### Q3: LLM 调用超时？

**问题**：请求长时间无响应

**解决**：
1. 检查 LLM API 地址是否正确
2. 测试网络连通性：`curl http://your-llm-api/v1/models`
3. 增加超时时间（在 `config.py` 中设置）
4. 检查 API Key 是否有效

### Q4: 知识库文件不存在？

**问题**：启动时提示 `knowledge_base.json` 不存在

**解决**：
```bash
# 确保文件存在
ls aiops_demo/data/knowledge_base.json

# 如果不存在，系统会自动降级到简单匹配模式
# 可以参考示例创建知识库文件
```

### Q5: Web 界面无法访问？

**问题**：浏览器显示"无法访问"

**解决**：
1. 检查服务是否启动：`curl http://localhost:8888/health`
2. 检查端口占用：`netstat -ano | findstr 8888` (Windows) 或 `lsof -i :8888` (Linux/Mac)
3. 尝试使用 127.0.0.1 替代 localhost

---

## 📈 性能基准

### 测试环境
- CPU: Intel i7-10700K
- 内存: 16GB
- Python: 3.9
- 系统: Windows 10

### 性能指标
| 指标 | 数值 |
|------|------|
| **RAG 检索** | ~50ms |
| **LLM 推理** | 2-5秒 |
| **总响应时间** | 3-6秒 |
| **并发能力** | 10+ QPS |
| **内存占用** | ~500MB |

---

## 🎓 下一步

✅ 安装完成后，可以：

1. **阅读文档**
   - [LangChain 使用说明](docs/LangChain_使用说明.md)
   - [改造前后对比](docs/改造前后对比.md)
   - [完整需求文档](docs/需求文档.md)

2. **扩展功能**
   - 添加更多故障案例到知识库
   - 集成 Reranker 提升检索准确率
   - 使用 Milvus 替代 FAISS
   - 添加 Agent 工具调用

3. **生产部署**
   - 使用 Docker 容器化
   - 配置 Nginx 反向代理
   - 添加监控和日志
   - 设置负载均衡

---

**更新时间**: 2024-11-19  
**版本**: 2.0.0 (LangChain)
