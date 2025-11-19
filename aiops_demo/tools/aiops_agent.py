"""
AIOps 智能诊断 Agent
使用 LangGraph + Tool Calling 自动执行诊断操作
"""
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from typing import Dict, List
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tools.ssh_tool import SSHTool
from config import LLM_CONFIG


class AIOpsAgent:
    """AIOps 自动化诊断 Agent"""
    
    def __init__(self):
        """初始化 Agent（使用 LangGraph）"""
        self.ssh_tool = SSHTool()
        
        # 初始化 LLM with Tool Calling
        self.llm = ChatOpenAI(
            base_url=LLM_CONFIG['api_base'],
            api_key=LLM_CONFIG['api_key'],
            model=LLM_CONFIG['model'],
            temperature=0.3
        )
        
        # 定义工具
        self.tools = self._create_tools()
        
        # 系统提示词
        self.system_message = """你是一位资深的 AIOps 运维专家，精通 Linux 系统管理和故障诊断。

你有一个强大的工具：execute_ssh_command
- 可以在服务器 192.168.30.18 上执行任意 Shell 命令
- **你需要自己根据错误类型生成合适的诊断命令**

诊断流程：
1. 分析错误日志，提取关键信息（服务名称、端口、错误类型）
2. 根据服务类型，生成诊断命令：
   - MySQL: systemctl status mysql
   - Redis: systemctl status redis  
   - 端口检查: ss -tuln | grep <port>
   - 进程检查: ps aux | grep <service>
   - 日志查看: tail -50 /var/log/<service>/*.log
3. 执行命令获取实际状态
4. 基于结果给出：诊断、根本原因、解决方案

注意：最终必须返回 JSON 格式：{"diagnosis": "...", "root_cause": "...", "solution": "...", "confidence": 0.9}"""
        
        # 使用 LangGraph 创建 ReAct Agent（最简单的方式）
        self.agent_executor = create_react_agent(
            model=self.llm,
            tools=self.tools
        )
        
        print("✓ LangGraph Agent 已创建")
    
    def _create_tools(self) -> List:
        """创建工具列表"""
        
        ssh_tool_instance = self.ssh_tool
        
        @tool
        def execute_ssh_command(command: str) -> str:
            """在服务器 192.168.30.18 (root@192.168.30.18) 上执行 Shell 命令。
            
            使用场景：
            - 检查服务状态：systemctl status <service>
            - 查看端口监听：ss -tuln | grep <port>
            - 查看进程：ps aux | grep <process>
            - 查看日志：tail -50 /var/log/<service>/*.log
            - 检查磁盘：df -h
            - 检查内存：free -h
            
            Args:
                command: 要执行的 Shell 命令（如：systemctl status mysql）
                
            Returns:
                命令的输出结果
            """
            try:
                result = ssh_tool_instance.execute_command(command)
                
                if result['success']:
                    output = result['stdout'].strip()
                    return f"✓ 命令执行成功:\n{output}" if output else "✓ 命令执行成功（无输出）"
                else:
                    error = result['stderr'].strip()
                    return f"✗ 命令执行失败 (退出码: {result['exit_code']}):\n{error}"
                    
            except Exception as e:
                return f"✗ 执行异常: {str(e)}"
        
        return [execute_ssh_command]
    
    def diagnose_with_tools(self, error_log: str) -> Dict:
        """
        使用 Tool Calling 进行智能诊断
        
        Args:
            error_log: 错误日志
            
        Returns:
            {
                "diagnosis": "诊断结果",
                "root_cause": "根本原因",
                "solution": "解决方案",
                "confidence": 0.9,
                "retrieved_cases": []
            }
        """
        try:
            print("🤖 LangGraph Agent 开始分析...")
            
            # LangGraph Agent 执行（系统消息 + 用户输入）
            result = self.agent_executor.invoke({
                "messages": [
                    ("system", self.system_message),
                    ("user", f"""分析以下故障日志并进行诊断：

{error_log}

请：
1. 提取关键信息（服务名称、端口）
2. 使用 execute_ssh_command 工具执行诊断命令
3. 基于命令结果给出 JSON 格式诊断""")
                ]
            })
            
            # 解析最后的消息
            messages = result.get('messages', [])
            if messages:
                final_message = messages[-1].content
                print(f"📝 Agent 输出: {final_message[:200]}...")
                
                # 尝试解析 JSON
                import json
                import re
                json_match = re.search(r'\{[^}]+\}', final_message, re.DOTALL)
                if json_match:
                    diagnosis_data = json.loads(json_match.group())
                    diagnosis_data['retrieved_cases'] = []
                    return diagnosis_data
            
            # 降级：返回原始输出
            return {
                "diagnosis": "Agent 诊断完成",
                "root_cause": final_message if messages else "无输出",
                "solution": "参见诊断内容",
                "confidence": 0.8,
                "retrieved_cases": []
            }
            
        except Exception as e:
            print(f"⚠ Agent 执行失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "diagnosis": "Agent 诊断失败",
                "root_cause": str(e),
                "solution": "请使用 Chat 模式",
                "confidence": 0.0,
                "retrieved_cases": []
            }
