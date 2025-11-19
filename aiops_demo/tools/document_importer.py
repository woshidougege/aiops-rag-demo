"""
企业文档导入向量数据库工具
支持多种文档格式：PDF, Word, Markdown, TXT 等
"""

import os
import json
from typing import List, Dict
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import connections
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import EMBEDDING_CONFIG, MILVUS_CONFIG


class DocumentImporter:
    """企业文档批量导入工具"""
    
    def __init__(self):
        """初始化"""
        # 初始化 Embeddings
        self.embeddings = OpenAIEmbeddings(
            base_url=EMBEDDING_CONFIG['api_url'].replace('/embeddings', ''),
            api_key=EMBEDDING_CONFIG['api_key'],
            model=EMBEDDING_CONFIG['model']
        )
        
        # 文本分割器（重要：文档要切分成合适大小的块）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,        # 每块500字符
            chunk_overlap=50,      # 块之间重叠50字符，保证上下文连贯
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        
        # 连接 Milvus
        self._connect_milvus()
        
        # 支持的文件类型
        self.supported_formats = {
            '.pdf': self._load_pdf,
            '.txt': self._load_txt,
            '.md': self._load_markdown,
            '.doc': self._load_word,
            '.docx': self._load_word,
        }
    
    def _connect_milvus(self):
        """连接 Milvus"""
        try:
            connections.connect(
                alias="default",
                host=MILVUS_CONFIG['host'],
                port=int(MILVUS_CONFIG['port']),
                timeout=10
            )
            print(f"✓ Milvus 连接成功: {MILVUS_CONFIG['host']}:{MILVUS_CONFIG['port']}")
        except Exception as e:
            print(f"✗ Milvus 连接失败: {e}")
            raise
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        """加载 PDF 文件"""
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    def _load_txt(self, file_path: str) -> List[Document]:
        """加载 TXT 文件"""
        loader = TextLoader(file_path, encoding='utf-8')
        return loader.load()
    
    def _load_markdown(self, file_path: str) -> List[Document]:
        """加载 Markdown 文件"""
        loader = UnstructuredMarkdownLoader(file_path)
        return loader.load()
    
    def _load_word(self, file_path: str) -> List[Document]:
        """加载 Word 文件"""
        loader = UnstructuredWordDocumentLoader(file_path)
        return loader.load()
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        加载单个文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            切分后的文档列表
        """
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in self.supported_formats:
            print(f"⚠ 不支持的文件格式: {file_ext}")
            return []
        
        try:
            # 1. 加载文档
            print(f"📄 加载文档: {file_path}")
            loader_func = self.supported_formats[file_ext]
            documents = loader_func(file_path)
            
            # 2. 添加元数据
            for doc in documents:
                doc.metadata.update({
                    'source_file': os.path.basename(file_path),
                    'file_path': file_path,
                    'file_type': file_ext,
                })
            
            # 3. 切分文档（重要！）
            print(f"✂ 切分文档...")
            chunks = self.text_splitter.split_documents(documents)
            print(f"✓ 切分为 {len(chunks)} 个块")
            
            return chunks
            
        except Exception as e:
            print(f"✗ 加载失败: {e}")
            return []
    
    def load_directory(self, directory: str, recursive: bool = True) -> List[Document]:
        """
        批量加载目录下的所有文档
        
        Args:
            directory: 目录路径
            recursive: 是否递归子目录
            
        Returns:
            所有文档的切分块
        """
        all_chunks = []
        
        if recursive:
            file_paths = Path(directory).rglob('*')
        else:
            file_paths = Path(directory).glob('*')
        
        for file_path in file_paths:
            if file_path.is_file():
                chunks = self.load_document(str(file_path))
                all_chunks.extend(chunks)
        
        print(f"\n📊 总计加载: {len(all_chunks)} 个文档块")
        return all_chunks
    
    def import_to_milvus(
        self, 
        documents: List[Document], 
        collection_name: str = None,
        drop_old: bool = False
    ):
        """
        导入文档到 Milvus
        
        Args:
            documents: 文档列表
            collection_name: Collection 名称
            drop_old: 是否删除旧数据
        """
        if not documents:
            print("⚠ 没有文档需要导入")
            return None
        
        collection_name = collection_name or MILVUS_CONFIG['collection_name']
        
        try:
            print(f"\n🚀 开始导入到 Milvus Collection: {collection_name}")
            print(f"   文档数量: {len(documents)}")
            print(f"   删除旧数据: {drop_old}")
            
            # 创建向量存储并导入
            vectorstore = Milvus.from_documents(
                documents,
                self.embeddings,
                collection_name=collection_name,
                connection_args={"alias": "default"},
                drop_old=drop_old
            )
            
            print(f"✓ 导入完成！Collection: {collection_name}")
            return vectorstore
            
        except Exception as e:
            print(f"✗ 导入失败: {e}")
            raise
    
    def import_from_json(
        self, 
        json_file: str, 
        collection_name: str = None,
        drop_old: bool = False
    ):
        """
        从 JSON 格式的知识库导入
        JSON 格式示例:
        [
            {
                "error_type": "数据库连接失败",
                "log_content": "...",
                "root_cause": "...",
                "solution": "...",
                "severity": "high"
            }
        ]
        
        Args:
            json_file: JSON 文件路径
            collection_name: Collection 名称
            drop_old: 是否删除旧数据
        """
        try:
            print(f"📄 加载 JSON 知识库: {json_file}")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 转换为 Document 对象
            documents = []
            for item in data:
                # 构建文档内容
                content = f"""错误类型: {item.get('error_type', '')}
日志内容: {item.get('log_content', '')}
根本原因: {item.get('root_cause', '')}
解决方案: {item.get('solution', '')}"""
                
                # 添加元数据
                metadata = {
                    "error_type": item.get('error_type', ''),
                    "root_cause": item.get('root_cause', ''),
                    "solution": item.get('solution', ''),
                    "severity": item.get('severity', 'medium')
                }
                
                documents.append(Document(page_content=content, metadata=metadata))
            
            print(f"✓ 加载了 {len(documents)} 条记录")
            
            # 导入到 Milvus
            return self.import_to_milvus(documents, collection_name, drop_old)
            
        except Exception as e:
            print(f"✗ JSON 导入失败: {e}")
            raise


def main():
    """命令行使用示例"""
    import argparse
    
    parser = argparse.ArgumentParser(description='企业文档导入工具')
    parser.add_argument('--file', help='单个文件路径')
    parser.add_argument('--dir', help='目录路径（批量导入）')
    parser.add_argument('--json', help='JSON 知识库文件路径')
    parser.add_argument('--collection', default=MILVUS_CONFIG['collection_name'], 
                       help='Milvus Collection 名称')
    parser.add_argument('--drop-old', action='store_true', 
                       help='删除旧数据重新导入')
    
    args = parser.parse_args()
    
    importer = DocumentImporter()
    
    try:
        if args.json:
            # JSON 格式导入
            importer.import_from_json(
                args.json, 
                collection_name=args.collection,
                drop_old=args.drop_old
            )
        
        elif args.file:
            # 单文件导入
            chunks = importer.load_document(args.file)
            importer.import_to_milvus(
                chunks, 
                collection_name=args.collection,
                drop_old=args.drop_old
            )
        
        elif args.dir:
            # 目录批量导入
            chunks = importer.load_directory(args.dir, recursive=True)
            importer.import_to_milvus(
                chunks, 
                collection_name=args.collection,
                drop_old=args.drop_old
            )
        
        else:
            print("请指定 --file, --dir 或 --json 参数")
            parser.print_help()
    
    except Exception as e:
        print(f"\n❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
