import time
import asyncio
import aiohttp
from openai import OpenAI
from easydict import EasyDict
from .base_generator import BaseGenerator, register_generator
from verl.utils.vllm_request import vllm_generate


def register_api_method(name):
    """装饰器用于将方法注册到字典中"""
    def decorator(func):
        # 在装饰器中不直接访问类，而是在类中处理
        func._api_method_name = name
        return func
    return decorator


def register_async_api_method(name):
    """装饰器用于将异步方法注册到字典中"""
    def decorator(func):
        # 在装饰器中不直接访问类，而是在类中处理
        func._async_api_method_name = name
        return func
    return decorator


@register_generator('api')
class APIGenerator(BaseGenerator):
    # 创建一个函数映射字典
    api_methods = {}
    async_api_methods = {}

    def __init__(self, config: EasyDict):
        super().__init__(config)
        self.api_method = config.api_method

        # 在初始化时注册所有带有 _api_method_name 属性的方法
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, '_api_method_name'):
                self.api_methods[attr._api_method_name] = attr
            if callable(attr) and hasattr(attr, '_async_api_method_name'):
                self.async_api_methods[attr._async_api_method_name] = attr

        if self.api_method not in self.api_methods:
            raise ValueError(f"Unsupported API method: {self.api_method}. The API method should be in [{', '.join(self.api_methods.keys())}]")

        if hasattr(config, 'port'):
            self.port = config.port
        else:
            self.port = 9000
        
        if hasattr(config, 'model_name'):
            self.model_name = config.model_name
        else:
            self.model_name = None
        
        self.selected_method = self.api_methods[self.api_method]
        # 如果存在异步版本，也选择对应的异步方法
        if self.api_method in self.async_api_methods:
            self.selected_async_method = self.async_api_methods[self.api_method]
        else:
            # 如果没有实现特定的异步方法，提供一个基于同步方法的异步包装
            self.selected_async_method = None
    
    @register_api_method('qwq')
    def get_response_qwq(self, message_list, temperature=0.7):
        client = OpenAI(api_key='dummy_key', base_url='http://10.35.146.75:8419/v1')
        model_name = 'deepseek-r1-friday'
        response = client.chat.completions.create(
            model=model_name,
            messages=message_list,
            temperature=temperature, 
            stream=True
        )

        response_str = ""
        for chunk in response:
            for choice in chunk.choices:
                response_str += choice.delta.content
        
        return response_str

    @register_async_api_method('qwq')
    async def get_response_qwq_async(self, message_list, temperature=0.7):
        # 使用线程池运行同步版本
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.get_response_qwq, message_list, temperature
        )

    @register_api_method('deepseek-r1')
    def get_response_r1(self, message_list, temperature=0.7):
        client = OpenAI(api_key='dummy_key', base_url='http://deepseek.sankuai.com/v1')
        model_name = 'DeepSeek-R1-BF16'
        response = client.chat.completions.create(
            model=model_name,
            messages=message_list,
            temperature=temperature, 
            stream=True
        )

        response_str = ""
        for chunk in response:
            for choice in chunk.choices:
                response_str += choice.delta.content
        
        return response_str

    @register_async_api_method('deepseek-r1')
    async def get_response_r1_async(self, message_list, temperature=0.7):
        # 使用线程池运行同步版本
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.get_response_r1, message_list, temperature
        )
    
    @register_api_method('local')
    def get_response_local(self, message_list, temperature=0.7):
        stream = False
        response = ""
        
        # 尝试调用API的最大次数
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                print(f"尝试调用API服务 (尝试 {attempt+1}/{max_attempts}), 端口: {self.port}")
                generation_stream = vllm_generate(
                    f'http://0.0.0.0:{self.port}', message_list, model=self.model_name, 
                    temperature=temperature, stream=stream
                )
                
                if stream:
                    if generation_stream is not None:
                        for chunk in generation_stream:
                            response += chunk
                else:
                    response = generation_stream
                
                # 检查响应是否为空
                if response is None or response == "":
                    print(f"警告: API返回了空响应 (尝试 {attempt+1}/{max_attempts})")
                    if attempt < max_attempts - 1:
                        print("等待1秒后重试...")
                        time.sleep(1)
                        continue
                    else:
                        print("所有尝试都失败，返回默认响应")
                        return "API服务暂时不可用，请稍后再试。"
                else:
                    # 正常获取到响应，跳出循环
                    break
                
            except Exception as e:
                print(f'API调用错误: {e} (尝试 {attempt+1}/{max_attempts})')
                if attempt < max_attempts - 1:
                    print("等待1秒后重试...")
                    time.sleep(1)
                else:
                    print("所有尝试都失败，返回默认响应")
                    return "API服务暂时不可用，请稍后再试。"
        
        return response

    @register_async_api_method('local')
    async def get_response_local_async(self, message_list, temperature=0.7):
        """异步版本的本地VLLM API调用"""
        stream = False
        response = ""
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                print(f"尝试调用API服务 (尝试 {attempt+1}/{max_attempts}), 端口: {self.port}")
                generation_stream = await vllm_generate(
                    f'http://0.0.0.0:{self.port}', 
                    message_list, 
                    model=self.model_name, 
                    temperature=temperature, 
                    stream=stream,
                    is_async=True
                )
                
                if stream:
                    if generation_stream is not None:
                        for chunk in generation_stream:
                            response += chunk
                else:
                    response = generation_stream
                
                # 检查响应是否为空
                if response is None or response == "":
                    print(f"警告: API返回了空响应 (尝试 {attempt+1}/{max_attempts})")
                    if attempt < max_attempts - 1:
                        print("等待1秒后重试...")
                        await asyncio.sleep(1)
                        continue
                    else:
                        print("所有尝试都失败，返回默认响应")
                        return "API服务暂时不可用，请稍后再试。"
                else:
                    # 正常获取到响应，跳出循环
                    break
                
            except Exception as e:
                print(f'API调用错误: {e} (尝试 {attempt+1}/{max_attempts})')
                if attempt < max_attempts - 1:
                    print("等待1秒后重试...")
                    await asyncio.sleep(1)
                else:
                    print("所有尝试都失败，返回默认响应")
                    return "API服务暂时不可用，请稍后再试。"
        
        return response

    def generate(self, input_data, temperature=0.7):
        """同步生成方法"""
        # 检查输入的类型。如果是字符串，转换为 message_list 格式
        if isinstance(input_data, str):
            message_list = [{'role': 'system', 'content': input_data}]
        elif isinstance(input_data, list):
            message_list = input_data
        else:
            raise ValueError("Input must be either a list or a string.")

        # 调用选择的方法
        return self.selected_method(message_list, temperature)

    async def generate_async(self, input_data, temperature=0.7):
        """异步生成方法"""
        # 检查输入的类型。如果是字符串，转换为 message_list 格式
        if isinstance(input_data, str):
            message_list = [{'role': 'system', 'content': input_data}]
        elif isinstance(input_data, list):
            message_list = input_data
        else:
            raise ValueError("Input must be either a list or a string.")

        # 如果有异步方法，调用对应的异步方法
        if self.selected_async_method:
            return await self.selected_async_method(message_list, temperature)
        else:
            # 没有对应的异步方法，使用线程池运行同步方法
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self.selected_method, message_list, temperature
            )
