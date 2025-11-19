"""
测试所有服务连接
"""
import requests
from config import LLM_CONFIG, EMBEDDING_CONFIG, MILVUS_CONFIG

print("="*60)
print("测试服务连接")
print("="*60)

# 1. 测试LLM服务
print("\n1. 测试LLM服务...")
try:
    response = requests.post(
        f"{LLM_CONFIG['api_base']}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_CONFIG['api_key']}"},
        json={
            "model": LLM_CONFIG['model'],
            "messages": [{"role": "user", "content": "你好"}],
            "max_tokens": 10
        },
        timeout=10
    )
    if response.status_code == 200:
        print(f"   ✓ LLM服务正常 ({LLM_CONFIG['model']})")
        print(f"   响应: {response.json()['choices'][0]['message']['content']}")
    else:
        print(f"   ✗ LLM服务异常: {response.status_code}")
except Exception as e:
    print(f"   ✗ LLM连接失败: {e}")

# 2. 测试向量服务
print("\n2. 测试向量服务...")
try:
    response = requests.post(
        EMBEDDING_CONFIG['api_url'],
        headers={"Authorization": f"Bearer {EMBEDDING_CONFIG['api_key']}"},
        json={
            "model": EMBEDDING_CONFIG['model'],
            "input": "测试文本"
        },
        timeout=10
    )
    if response.status_code == 200:
        embedding = response.json()['data'][0]['embedding']
        print(f"   ✓ 向量服务正常 ({EMBEDDING_CONFIG['model']})")
        print(f"   向量维度: {len(embedding)}")
    else:
        print(f"   ✗ 向量服务异常: {response.status_code}")
except Exception as e:
    print(f"   ✗ 向量服务失败: {e}")

# 3. 测试Milvus
print("\n3. 测试Milvus...")
try:
    from pymilvus import connections
    connections.connect(
        alias="test",
        host=MILVUS_CONFIG['host'],
        port=MILVUS_CONFIG['port'],
        timeout=5
    )
    print(f"   ✓ Milvus连接正常 ({MILVUS_CONFIG['host']}:{MILVUS_CONFIG['port']})")
    connections.disconnect("test")
except Exception as e:
    print(f"   ✗ Milvus连接失败: {e}")

print("\n" + "="*60)
print("测试完成！")
print("="*60)
