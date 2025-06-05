import time
import json
from concurrent.futures import ThreadPoolExecutor
from envs.utils.async_mcp_manager import AsyncMCPManager

def main():
    print("开始测试负载均衡功能...")
    
    # 配置MCP服务器
    config = {
        "mcpServers": {
            "test-server": {
                "url": "http://10.46.7.131:8080/sse",
            }
        }
    }
    
    # 初始化管理器
    print("\n1. 初始化AsyncMCPManager...")
    manager = AsyncMCPManager(num_instances=3)
    tools = manager.initConfig(config)
    
    if not tools:
        print("错误：没有找到可用的工具")
        return
        
    # 获取第一个工具用于测试
    test_tool = tools[0]
    print(f"获取到工具: {test_tool.name}")
    print(f"实例数量: {len(test_tool.instances)}")
    
    # 记录每个实例的调用次数
    call_counts = [0] * len(test_tool.instances)
    
    def make_request(i):
        """发送测试请求"""
        try:
            params = {
                "message": f"测试请求 {i}",
                "timestamp": time.time()
            }
            
            # 记录请求开始时间
            start_time = time.time()
            
            # 发送请求
            response = test_tool.call(params)
            
            # 计算响应时间
            elapsed = time.time() - start_time
            
            # 记录被调用的实例
            with test_tool._lock:
                for idx, (active, _) in enumerate(test_tool._instance_states):
                    if active > 0:
                        call_counts[idx] += 1
                        print(f"\n请求 {i:2d} 由实例 {idx} 处理，耗时: {elapsed:.3f}秒")
                        print(f"当前负载分布: {[state[0] for state in test_tool._instance_states]}")
                        break
            
            return response
            
        except Exception as e:
            print(f"\n请求 {i} 失败: {e}")
            return None
    
    print("\n2. 开始并发测试...")
    print("发送30个并发请求...")
    
    # 使用线程池并发发送请求
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request, i) for i in range(30)]
        
        # 等待所有请求完成
        for future in futures:
            future.result()
    
    # 统计结果
    print("\n3. 测试结果统计:")
    print("\n负载分布情况:")
    total_calls = sum(call_counts)
    for i, count in enumerate(call_counts):
        percentage = (count / total_calls * 100) if total_calls > 0 else 0
        print(f"实例 {i}: {count:2d} 次调用 ({percentage:5.1f}%)")
        
    # 检查负载均衡效果
    max_calls = max(call_counts)
    min_calls = min(call_counts)
    imbalance = (max_calls - min_calls) / (total_calls / len(call_counts)) if total_calls > 0 else 0
    
    print(f"\n负载不均衡度: {imbalance:.2f}")
    print("(0表示完全均衡，数值越大表示越不均衡)")
    
    # 检查是否所有实例都被使用
    unused = sum(1 for count in call_counts if count == 0)
    if unused > 0:
        print(f"\n警告：有 {unused} 个实例未被使用")
    else:
        print("\n所有实例都参与了负载均衡")

if __name__ == "__main__":
    main() 