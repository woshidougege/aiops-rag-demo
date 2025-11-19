"""
快速测试 AIOps Agent
"""
import sys
sys.path.insert(0, 'aiops_demo')

from tools.aiops_agent import AIOpsAgent

# 初始化 Agent
print("🚀 初始化 AIOps Agent...")
agent = AIOpsAgent()

# 测试案例1：MySQL 连接失败
print("\n" + "="*60)
print("测试案例：MySQL 连接失败")
print("="*60)

error_log = """
2024-11-19 17:00:00 ERROR: 
pymysql.err.OperationalError: (2003, "Can't connect to MySQL server on '192.168.30.18' (10061)")
Connection refused when trying to connect to database
"""

result = agent.diagnose_with_tools(error_log)

print("\n📊 诊断结果:")
print(f"  诊断: {result.get('diagnosis')}")
print(f"  根本原因: {result.get('root_cause')}")
print(f"  解决方案: {result.get('solution')}")
print(f"  使用的工具: {result.get('tool_calls')}")
print(f"  置信度: {result.get('confidence')}")
