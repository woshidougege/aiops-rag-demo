"""
文档导入示例 - 快速开始
演示如何将企业文档导入到 Milvus 向量数据库
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from aiops_demo.tools.document_importer import DocumentImporter


def example_1_import_json():
    """示例 1: 导入 JSON 格式的知识库"""
    print("\n" + "="*60)
    print("示例 1: 导入 JSON 知识库")
    print("="*60)
    
    importer = DocumentImporter()
    
    # 导入现有的 JSON 知识库
    json_file = "aiops_demo/data/knowledge_base.json"
    
    if os.path.exists(json_file):
        importer.import_from_json(
            json_file,
            collection_name="aiops_knowledge_v1",
            drop_old=True  # 删除旧数据重新导入
        )
    else:
        print(f"⚠ 文件不存在: {json_file}")


def example_2_import_single_file():
    """示例 2: 导入单个文档文件"""
    print("\n" + "="*60)
    print("示例 2: 导入单个文档")
    print("="*60)
    
    importer = DocumentImporter()
    
    # 假设你有一个运维手册
    file_path = "docs/LangChain_使用说明.md"
    
    if os.path.exists(file_path):
        # 1. 加载并切分文档
        chunks = importer.load_document(file_path)
        
        # 2. 导入到 Milvus
        importer.import_to_milvus(
            chunks,
            collection_name="langchain_docs",
            drop_old=True
        )
    else:
        print(f"⚠ 文件不存在: {file_path}")


def example_3_import_directory():
    """示例 3: 批量导入整个目录"""
    print("\n" + "="*60)
    print("示例 3: 批量导入文档目录")
    print("="*60)
    
    importer = DocumentImporter()
    
    # 导入 docs 目录下的所有文档
    doc_dir = "docs"
    
    if os.path.exists(doc_dir):
        # 1. 递归加载目录下所有支持的文档
        chunks = importer.load_directory(doc_dir, recursive=True)
        
        # 2. 导入到 Milvus
        importer.import_to_milvus(
            chunks,
            collection_name="project_docs",
            drop_old=True
        )
    else:
        print(f"⚠ 目录不存在: {doc_dir}")


def example_4_custom_metadata():
    """示例 4: 自定义元数据导入"""
    print("\n" + "="*60)
    print("示例 4: 自定义元数据")
    print("="*60)
    
    from langchain_core.documents import Document
    
    importer = DocumentImporter()
    
    # 手动创建文档（可以添加自定义元数据）
    documents = [
        Document(
            page_content="MySQL 数据库连接失败通常是由于网络问题或配置错误导致的。",
            metadata={
                "category": "数据库",
                "severity": "high",
                "tags": ["MySQL", "连接", "故障"],
                "department": "运维部"
            }
        ),
        Document(
            page_content="Redis 内存占用过高需要检查是否有大key存在。",
            metadata={
                "category": "缓存",
                "severity": "medium",
                "tags": ["Redis", "内存", "性能"],
                "department": "运维部"
            }
        ),
    ]
    
    # 导入
    importer.import_to_milvus(
        documents,
        collection_name="custom_knowledge",
        drop_old=True
    )


def example_5_incremental_import():
    """示例 5: 增量导入（不删除旧数据）"""
    print("\n" + "="*60)
    print("示例 5: 增量导入新文档")
    print("="*60)
    
    importer = DocumentImporter()
    
    # 假设有新的文档要添加
    new_docs_dir = "docs"
    
    if os.path.exists(new_docs_dir):
        chunks = importer.load_directory(new_docs_dir)
        
        # 注意：drop_old=False，不删除已有数据
        importer.import_to_milvus(
            chunks,
            collection_name="aiops_knowledge_v1",
            drop_old=False  # 增量导入
        )


def main():
    """主函数：选择要运行的示例"""
    print("\n" + "="*60)
    print("📚 文档导入示例")
    print("="*60)
    print("\n选择要运行的示例:")
    print("  1 - 导入 JSON 知识库")
    print("  2 - 导入单个文档文件")
    print("  3 - 批量导入文档目录")
    print("  4 - 自定义元数据导入")
    print("  5 - 增量导入")
    print("  0 - 运行所有示例")
    
    choice = input("\n请选择 (0-5): ").strip()
    
    try:
        if choice == '1':
            example_1_import_json()
        elif choice == '2':
            example_2_import_single_file()
        elif choice == '3':
            example_3_import_directory()
        elif choice == '4':
            example_4_custom_metadata()
        elif choice == '5':
            example_5_incremental_import()
        elif choice == '0':
            example_1_import_json()
            example_2_import_single_file()
            example_3_import_directory()
            example_4_custom_metadata()
        else:
            print("无效选择")
    
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
