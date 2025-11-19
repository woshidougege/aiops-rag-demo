"""
SSH 远程执行工具 - 简化版
直接免密登录到 192.168.30.18
"""
import paramiko
from typing import Dict

class SSHTool:
    """SSH 远程命令执行工具（免密登录）"""
    
    # 固定配置
    DEFAULT_HOST = "192.168.30.18"
    DEFAULT_USER = "root"
    DEFAULT_PORT = 22
    
    def __init__(self):
        pass
    
    def execute_command(self, command: str, host: str = None) -> Dict:
        """
        在远程服务器上执行命令（使用免密登录）
        
        Args:
            command: 要执行的 Shell 命令
            host: 服务器 IP（默认 192.168.30.18）
            
        Returns:
            {
                "success": bool,
                "stdout": str,
                "stderr": str,
                "exit_code": int,
                "command": str
            }
        """
        # 使用默认主机
        if not host:
            host = self.DEFAULT_HOST
        
        try:
            print(f"  🔌 SSH 连接到 {host}...")
            
            # 创建 SSH 客户端
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # 免密登录（使用系统默认的 SSH 密钥）
            ssh.connect(
                hostname=host,
                port=self.DEFAULT_PORT,
                username=self.DEFAULT_USER,
                look_for_keys=True,  # 自动查找 ~/.ssh/id_rsa 等密钥
                timeout=10
            )
            
            print(f"  ✓ 已连接，执行命令: {command}")
            
            # 执行命令
            stdin, stdout, stderr = ssh.exec_command(command, timeout=30)
            
            # 获取结果
            exit_code = stdout.channel.recv_exit_status()
            stdout_text = stdout.read().decode('utf-8', errors='ignore')
            stderr_text = stderr.read().decode('utf-8', errors='ignore')
            
            ssh.close()
            
            if exit_code == 0:
                print(f"  ✓ 命令执行成功")
            else:
                print(f"  ⚠ 命令执行失败 (退出码: {exit_code})")
            
            return {
                "success": exit_code == 0,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "exit_code": exit_code,
                "command": command
            }
            
        except Exception as e:
            print(f"  ❌ SSH 执行失败: {e}")
            return {
                "success": False,
                "stdout": "",
                "stderr": f"SSH 连接或执行失败: {str(e)}",
                "exit_code": -1,
                "command": command
            }
