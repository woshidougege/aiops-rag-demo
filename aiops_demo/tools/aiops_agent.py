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
        self.execution_log = []  # 存储执行日志
        
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
        execution_log = self.execution_log  # 引用共享日志
        
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
                # 记录命令到日志
                execution_log.append({"type": "command", "content": command})
                
                result = ssh_tool_instance.execute_command(command)
                
                if result['success']:
                    output = result['stdout'].strip()
                    # 记录结果到日志
                    execution_log.append({"type": "result", "content": output[:500], "success": True})
                    return f"✓ 命令执行成功:\n{output}" if output else "✓ 命令执行成功（无输出）"
                else:
                    error = result['stderr'].strip()
                    # 记录错误到日志
                    execution_log.append({"type": "result", "content": error[:500], "success": False, "exit_code": result['exit_code']})
                    return f"✗ 命令执行失败 (退出码: {result['exit_code']}):\n{error}"
                    
            except Exception as e:
                execution_log.append({"type": "error", "content": str(e)})
                return f"✗ 执行异常: {str(e)}"
        
        return [execute_ssh_command]
    
    async def diagnose_with_tools_stream(self, error_log: str):
        """
        流式执行诊断（实时返回每一步）- 简化版，直接监控 SSH 工具
        """
        try:
            print("[Agent Stream] 开始流式诊断")
            # 清空之前的日志
            self.execution_log.clear()
            
            yield {"type": "thinking", "content": "🔍 Agent 正在分析错误日志..."}
            
            # 由于 LangGraph 的 astream 事件格式复杂，这里简化：
            # 直接执行并收集步骤日志
            import asyncio
            
            # 在后台执行 Agent
            result_container = []
            last_log_index = 0  # 记录已发送的日志位置
            
            async def run_agent():
                try:
                    # invoke 是同步的，使用 to_thread 避免阻塞事件循环
                    result = await asyncio.to_thread(
                        self.agent_executor.invoke,
                        {
                            "messages": [
                                ("system", self.system_message),
                                ("user", f"分析故障：\n{error_log}\n\n请使用工具诊断")
                            ]
                        }
                    )
                    result_container.append(result)
                except Exception as e:
                    print(f"[Agent] 执行失败: {e}")
                    result_container.append({"error": str(e)})
            
            # 启动 Agent 任务
            agent_task = asyncio.create_task(run_agent())
            
            # 实时轮询执行日志（真正的流式输出）
            while not agent_task.done():
                # 检查是否有新的执行日志
                if len(self.execution_log) > last_log_index:
                    for i in range(last_log_index, len(self.execution_log)):
                        log_entry = self.execution_log[i]
                        
                        if log_entry["type"] == "command":
                            # 发送命令
                            yield {
                                "type": "tool_call",
                                "args": {"command": log_entry["content"]}
                            }
                        elif log_entry["type"] == "result":
                            # 发送结果
                            yield {
                                "type": "tool_result",
                                "content": log_entry["content"],
                                "success": log_entry.get("success", True)
                            }
                        elif log_entry["type"] == "error":
                            yield {
                                "type": "thinking",
                                "content": f"❌ 错误: {log_entry['content']}"
                            }
                    
                    last_log_index = len(self.execution_log)
                
                await asyncio.sleep(0.2)  # 每200ms检查一次
            
            # 等待完成
            await agent_task
            
            # 发送剩余的日志
            if len(self.execution_log) > last_log_index:
                for i in range(last_log_index, len(self.execution_log)):
                    log_entry = self.execution_log[i]
                    if log_entry["type"] == "command":
                        yield {"type": "tool_call", "args": {"command": log_entry["content"]}}
                    elif log_entry["type"] == "result":
                        yield {"type": "tool_result", "content": log_entry["content"], "success": log_entry.get("success", True)}
            
            if result_container:
                result = result_container[0]
                if "error" in result:
                    yield {"type": "error", "message": result["error"]}
                else:
                    # 提取最终消息并解析
                    messages = result.get("messages", [])
                    final_content = messages[-1].content if messages else "诊断完成"
                    
                    # 尝试从 LLM 输出中提取结构化信息
                    import re
                    import json
                    
                    diagnosis = "基于命令执行结果的诊断"
                    root_cause = "参见执行日志中的命令输出"
                    solution = "参见 LLM 分析"
                    
                    # 尝试解析 JSON
                    json_match = re.search(r'\{[^{}]*"diagnosis"[^{}]*\}', final_content, re.DOTALL)
                    if json_match:
                        try:
                            parsed = json.loads(json_match.group())
                            diagnosis = parsed.get("diagnosis", diagnosis)
                            root_cause = parsed.get("root_cause", root_cause)
                            solution = parsed.get("solution", solution)
                        except:
                            pass
                    else:
                        # 没有 JSON，使用原文
                        diagnosis = final_content[:300]
                    
                    yield {"type": "thinking", "content": "✓ 诊断完成"}
                    yield {
                        "type": "final_result",
                        "data": {
                            "diagnosis": diagnosis,
                            "root_cause": root_cause,
                            "solution": solution,
                            "confidence": 0.85,
                            "retrieved_cases": []
                        }
                    }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield {"type": "error", "message": str(e)}
    
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
