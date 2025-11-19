"""
Milvus 数据查看工具
类似于 MySQL 的命令行客户端，可以查看和检索 Milvus 数据
"""

import sys
import os
from pymilvus import connections, Collection, utility
from langchain_openai import OpenAIEmbeddings
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import EMBEDDING_CONFIG, MILVUS_CONFIG


class MilvusViewer:
    """Milvus 数据查看器"""
    
    def __init__(self):
        """连接 Milvus"""
        try:
            connections.connect(
                alias="default",
                host=MILVUS_CONFIG['host'],
                port=int(MILVUS_CONFIG['port']),
                timeout=10
            )
            print(f"✓ 已连接到 Milvus: {MILVUS_CONFIG['host']}:{MILVUS_CONFIG['port']}")
            
            # 初始化 Embeddings（用于向量搜索）
            self.embeddings = OpenAIEmbeddings(
                base_url=EMBEDDING_CONFIG['api_url'].replace('/embeddings', ''),
                api_key=EMBEDDING_CONFIG['api_key'],
                model=EMBEDDING_CONFIG['model']
            )
            
        except Exception as e:
            print(f"✗ 连接失败: {e}")
            raise
    
    def list_collections(self):
        """列出所有 Collections"""
        print("\n" + "="*60)
        print("📚 所有 Collections")
        print("="*60)
        
        collections = utility.list_collections()
        
        if not collections:
            print("（无 Collection）")
            return
        
        for i, name in enumerate(collections, 1):
            try:
                col = Collection(name)
                count = col.num_entities
                print(f"{i}. {name} ({count:,} 条记录)")
            except Exception as e:
                print(f"{i}. {name} (读取失败: {e})")
    
    def show_collection_info(self, collection_name: str):
        """显示 Collection 详细信息"""
        print("\n" + "="*60)
        print(f"📋 Collection 信息: {collection_name}")
        print("="*60)
        
        try:
            collection = Collection(collection_name)
            collection.load()
            
            print(f"记录数量: {collection.num_entities:,}")
            print(f"Schema: {collection.schema}")
            print(f"\n字段列表:")
            for field in collection.schema.fields:
                print(f"  - {field.name} ({field.dtype})")
            
        except Exception as e:
            print(f"✗ 获取信息失败: {e}")
    
    def query_all(self, collection_name: str, limit: int = 10):
        """查询所有数据（分页）"""
        print("\n" + "="*60)
        print(f"📄 查询数据: {collection_name} (前 {limit} 条)")
        print("="*60)
        
        try:
            collection = Collection(collection_name)
            collection.load()
            
            # 查询所有数据
            results = collection.query(
                expr="pk >= 0",  # 查询所有（pk 是主键）
                limit=limit,
                output_fields=["*"]
            )
            
            if not results:
                print("（无数据）")
                return
            
            for i, item in enumerate(results, 1):
                print(f"\n--- 记录 {i} ---")
                for key, value in item.items():
                    if key == 'vector':
                        print(f"{key}: [向量数据，维度={len(value)}]")
                    else:
                        # 截断长文本
                        if isinstance(value, str) and len(value) > 100:
                            print(f"{key}: {value[:100]}...")
                        else:
                            print(f"{key}: {value}")
            
            print(f"\n共 {len(results)} 条记录")
            
        except Exception as e:
            print(f"✗ 查询失败: {e}")
    
    def search(self, collection_name: str, query: str, top_k: int = 3):
        """语义搜索"""
        print("\n" + "="*60)
        print(f"🔍 语义搜索: {query}")
        print("="*60)
        
        try:
            collection = Collection(collection_name)
            collection.load()
            
            # 生成查询向量
            query_vector = self.embeddings.embed_query(query)
            
            # 搜索
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["text", "source_file", "error_type", "root_cause", "solution"]
            )
            
            if not results or len(results[0]) == 0:
                print("（无结果）")
                return
            
            for i, hit in enumerate(results[0], 1):
                print(f"\n--- Top {i} (距离: {hit.distance:.4f}) ---")
                entity = hit.entity
                
                # 显示字段
                for field_name in ["text", "source_file", "error_type", "root_cause", "solution"]:
                    if hasattr(entity, field_name):
                        value = getattr(entity, field_name)
                        if isinstance(value, str) and len(value) > 200:
                            print(f"{field_name}: {value[:200]}...")
                        else:
                            print(f"{field_name}: {value}")
            
        except Exception as e:
            print(f"✗ 搜索失败: {e}")
    
    def delete_collection(self, collection_name: str):
        """删除 Collection"""
        confirm = input(f"\n⚠️  确认删除 Collection '{collection_name}'? (yes/no): ")
        if confirm.lower() == 'yes':
            try:
                utility.drop_collection(collection_name)
                print(f"✓ Collection '{collection_name}' 已删除")
            except Exception as e:
                print(f"✗ 删除失败: {e}")
        else:
            print("已取消")


def main():
    """命令行交互界面"""
    viewer = MilvusViewer()
    
    print("\n" + "="*60)
    print("🔧 Milvus 数据查看工具")
    print("="*60)
    
    while True:
        print("\n可用命令:")
        print("  1. list         - 列出所有 Collections")
        print("  2. info <名称>  - 查看 Collection 信息")
        print("  3. query <名称> [limit] - 查询数据")
        print("  4. search <名称> <关键词> [top_k] - 语义搜索")
        print("  5. delete <名称> - 删除 Collection")
        print("  6. quit         - 退出")
        
        cmd = input("\n> ").strip().split()
        
        if not cmd:
            continue
        
        action = cmd[0].lower()
        
        try:
            if action == 'list':
                viewer.list_collections()
            
            elif action == 'info':
                if len(cmd) < 2:
                    print("用法: info <collection_name>")
                else:
                    viewer.show_collection_info(cmd[1])
            
            elif action == 'query':
                if len(cmd) < 2:
                    print("用法: query <collection_name> [limit]")
                else:
                    limit = int(cmd[2]) if len(cmd) > 2 else 10
                    viewer.query_all(cmd[1], limit)
            
            elif action == 'search':
                if len(cmd) < 3:
                    print("用法: search <collection_name> <query> [top_k]")
                else:
                    query = ' '.join(cmd[2:-1]) if len(cmd) > 3 else cmd[2]
                    top_k = int(cmd[-1]) if len(cmd) > 3 and cmd[-1].isdigit() else 3
                    viewer.search(cmd[1], query, top_k)
            
            elif action == 'delete':
                if len(cmd) < 2:
                    print("用法: delete <collection_name>")
                else:
                    viewer.delete_collection(cmd[1])
            
            elif action in ['quit', 'exit', 'q']:
                print("再见！")
                break
            
            else:
                print(f"未知命令: {action}")
        
        except Exception as e:
            print(f"❌ 执行失败: {e}")


if __name__ == "__main__":
    main()
