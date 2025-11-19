"""
AIOps 智能诊断 Agent
使用 LangChain Tool Calling 自动执行诊断操作
"""
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Dict, List
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tools.ssh_tool import SSHTool
from config import LLM_CONFIG


class AIOpsAgent:
    """AIOps 自动化诊断 Agent"""
    
    def __init__(self):
        """初始化 Agent（免密登录到 192.168.30.18）"""
        self.ssh_tool = SSHTool()
        
        # 初始化 LLM
        self.llm = ChatOpenAI(
            base_url=LLM_CONFIG['api_base'],
            api_key=LLM_CONFIG['api_key'],
            model=LLM_CONFIG['model'],
            temperature=0.3  # 降低温度，让 Agent 更稳定
        )
        
        # 定义工具
        self.tools = self._create_tools()
        
        # 创建 Agent Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一位资深的 AIOps 运维专家，精通 Linux 系统管理和故障诊断。

你有一个强大的工具：execute_ssh_command
- 可以在远程服务器上执行任意 Shell 命令
- **你需要自己根据错误类型生成合适的诊断命令**

诊断流程：
1. 分析错误日志，提取关键信息：
   - IP 地址
   - 端口号
   - 服务名称（MySQL、Redis、Nginx、Kafka 等）
   - 错误类型

2. 根据服务类型，自己生成诊断命令，例如：
   - MySQL: systemctl status mysql 或 systemctl status mysqld
   - Redis: systemctl status redis
   - PostgreSQL: systemctl status postgresql
   - Nginx: systemctl status nginx
   - Kafka: systemctl status kafka
   - Docker: docker ps, docker logs <container>
   - 端口检查: ss -tuln | grep <port> 或 netstat -tuln | grep <port>
   - 进程检查: ps aux | grep <service>
   - 日志查看: tail -100 /var/log/<service>/*.log

3. 执行命令获取实际状态

4. 基于返回结果，给出：
   - 诊断结论
   - 根本原因
   - 详细的解决方案

注意：
- 命令要具体，不要泛泛而谈
- 如果第一个命令失败，尝试其他常见的变体
- 综合多个命令的结果进行判断
- 最终输出必须包含：diagnosis, root_cause, solution"""),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # 创建 Agent
        self.agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )
    
    def _create_tools(self) -> List[Tool]:
        """创建工具列表"""
        
        def execute_ssh_command(command: str) -> str:
            """
            在服务器 192.168.30.18 上执行 Shell 命令
            
            Args:
                command: 要执行的 Shell 命令（如：systemctl status mysql）
                
            Returns:
                命令的输出结果
            """
            try:
                result = self.ssh_tool.execute_command(command)
                
                if result['success']:
                    output = result['stdout'].strip()
                    return f"✓ 命令执行成功:\n{output}" if output else "✓ 命令执行成功（无输出）"
                else:
                    error = result['stderr'].strip()
                    return f"✗ 命令执行失败 (退出码: {result['exit_code']}):\n{error}"
                    
            except Exception as e:
                return f"✗ 执行异常: {str(e)}"
        
        return [
            Tool(
                name="execute_ssh_command",
                func=execute_ssh_command,
                description="""在服务器 192.168.30.18 (root@192.168.30.18) 上执行 Shell 命令。
                
使用场景：
- 检查服务状态：systemctl status <service>
- 查看端口监听：ss -tuln | grep <port>
- 查看进程：ps aux | grep <process>
- 查看日志：tail -50 /var/log/<service>/*.log
- 检查磁盘：df -h
- 检查内存：free -h

输入：直接输入要执行的 Shell 命令（不需要 IP，默认连接到 192.168.30.18）
示例：systemctl status mysql"""
            )
        ]
    
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
                "tool_calls": ["使用的工具列表"],
                "confidence": 0.9
            }
        """
        try:
            print("🤖 Agent 开始分析...")
            
            # Agent 执行
            result = self.agent_executor.invoke({
                "input": f"""分析以下故障日志并进行诊断：

{error_log}

请：
1. 提取关键信息（IP、端口、服务）
2. 如果可以，使用工具进行实际检查
3. 给出诊断、根本原因和解决方案"""
            })
            
            # 解析 Agent 输出
            output = result.get('output', '')
            
            return {
                "diagnosis": "Agent 自动诊断",
                "root_cause": output,
                "solution": "参见 Agent 分析结果",
                "tool_calls": [step.tool for step in result.get('intermediate_steps', [])],
                "confidence": 0.9
            }
            
        except Exception as e:
            print(f"⚠ Agent 执行失败: {e}")
            return {
                "diagnosis": "Agent 诊断失败",
                "root_cause": str(e),
                "solution": "请使用传统 RAG 方式诊断",
                "tool_calls": [],
                "confidence": 0.0
            }
