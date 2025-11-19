"""
测试所有配置的服务是否可用
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'aiops_demo'))

import requests
from pymilvus import connections
from config import LLM_CONFIG, EMBEDDING_CONFIG, RERANKER_CONFIG, MILVUS_CONFIG

print("=" * 60)
print("🧪 AIOps RAG 服务测试")
print("=" * 60)

# ===================================================================
# 1. 测试文本大模型 (Qwen-32B)
# ===================================================================
print("\n【1/4】测试文本大模型 (Qwen-32B)...")
print(f"  URL: {LLM_CONFIG['api_base']}")
print(f"  Model: {LLM_CONFIG['model']}")

try:
    response = requests.post(
        f"{LLM_CONFIG['api_base']}/chat/completions",
        headers={
            "Authorization": f"Bearer {LLM_CONFIG['api_key']}",
            "Content-Type": "application/json"
        },
        json={
            "model": LLM_CONFIG['model'],
            "messages": [{"role": "user", "content": "你好，请回复'测试成功'"}],
            "max_tokens": 20
        },
        timeout=10
    )
    
    if response.status_code == 200:
        result = response.json()
        reply = result['choices'][0]['message']['content']
        print(f"  ✅ 成功！响应: {reply}")
    else:
        print(f"  ❌ 失败！状态码: {response.status_code}")
        print(f"  响应: {response.text}")
except Exception as e:
    print(f"  ❌ 异常: {e}")

# ===================================================================
# 2. 测试向量模型 (BGE-M3)
# ===================================================================
print("\n【2/4】测试向量模型 (BGE-M3)...")
print(f"  URL: {EMBEDDING_CONFIG['api_url']}")
print(f"  Model: {EMBEDDING_CONFIG['model']}")

try:
    response = requests.post(
        EMBEDDING_CONFIG['api_url'],
        headers={
            "Authorization": f"Bearer {EMBEDDING_CONFIG['api_key']}",
            "Content-Type": "application/json"
        },
        json={
            "model": EMBEDDING_CONFIG['model'],
            "input": "测试文本"
        },
        timeout=10
    )
    
    if response.status_code == 200:
        result = response.json()
        embedding = result['data'][0]['embedding']
        print(f"  ✅ 成功！向量维度: {len(embedding)}")
    else:
        print(f"  ❌ 失败！状态码: {response.status_code}")
        print(f"  响应: {response.text}")
except Exception as e:
    print(f"  ❌ 异常: {e}")

# ===================================================================
# 3. 测试精排模型 (BGE-Reranker-v2-M3)
# ===================================================================
print("\n【3/4】测试精排模型 (BGE-Reranker-v2-M3)...")
print(f"  URL: {RERANKER_CONFIG['api_url']}")
print(f"  Model: {RERANKER_CONFIG['model']}")

try:
    response = requests.post(
        RERANKER_CONFIG['api_url'],
        headers={
            "Authorization": f"Bearer {RERANKER_CONFIG['api_key']}",
            "Content-Type": "application/json"
        },
        json={
            "model": RERANKER_CONFIG['model'],
            "query": "数据库连接失败",
            "documents": [
                "MySQL数据库连接超时",
                "Redis缓存连接异常",
                "网络延迟导致连接失败"
            ],
            "top_n": 2
        },
        timeout=10
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"  ✅ 成功！精排结果数量: {len(result.get('results', []))}")
        for i, item in enumerate(result.get('results', []), 1):
            print(f"    排名{i}: 文档{item['index']}, 相关度{item['relevance_score']:.4f}")
    else:
        print(f"  ❌ 失败！状态码: {response.status_code}")
        print(f"  响应: {response.text}")
except Exception as e:
    print(f"  ❌ 异常: {e}")

# ===================================================================
# 4. 测试 Milvus 向量数据库
# ===================================================================
print("\n【4/4】测试 Milvus 向量数据库...")
print(f"  Host: {MILVUS_CONFIG['host']}")
print(f"  Port: {MILVUS_CONFIG['port']}")

try:
    connections.connect(
        alias="test",
        host=MILVUS_CONFIG['host'],
        port=MILVUS_CONFIG['port'],
        timeout=5
    )
    print(f"  ✅ 成功！已连接到 Milvus")
    connections.disconnect("test")
except Exception as e:
    print(f"  ❌ 失败: {e}")
    print(f"  提示: 请确保 Milvus 服务器正在运行且网络可达")

# ===================================================================
# 测试总结
# ===================================================================
print("\n" + "=" * 60)
print("✅ 测试完成！")
print("=" * 60)
print("\n💡 提示:")
print("  - 如果 Milvus 不可用，系统会自动降级到 FAISS")
print("  - FAISS 是内存向量存储，功能完全够用")
print("  - 其他 3 个服务（LLM、Embedding、Reranker）都必须可用")
