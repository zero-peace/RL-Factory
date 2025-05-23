import time
import json
import aiohttp
import asyncio
import requests
from typing import Any


# 7B 模型 It took 0.744026 seconds to fall asleep.
# 7B 模型 It took 0.779162 seconds to wake up.

async def vllm_reward_async(url='http://0.0.0.0:8000', mode='sleep'):
    print('Changing mode')
    headers = {"User-Agent": "Test Client"}
    async with aiohttp.ClientSession() as session:
        if mode == 'sleep':
            async with session.post(f'{url}/sleep', headers=headers) as response:
                assert response.status == 200
        elif mode == 'wake_up':
            async with session.post(f'{url}/wake_up', headers=headers) as response:
                assert response.status == 200
        else:
            raise ValueError(f'Invalid mode: {mode}')


def vllm_reward(url='http://0.0.0.0:8000', mode='sleep'):
    print('Changing mode')
    headers = {"User-Agent": "Test Client"}
    if mode == 'sleep':
        response = requests.post(f'{url}/sleep', headers=headers)
    elif mode == 'wake_up':
        response = requests.post(f'{url}/wake_up', headers=headers)
    else:
        raise ValueError(f'Invalid mode: {mode}')

    assert response.status_code == 200


def vllm_generate(
    url, input_data,
    model: str = 'gpt-3.5-turbo',
    temperature: float = 0.7,
    max_tokens: int = 1024,
    stream: bool = False,
    **kwargs: Any
):
    """
    Send chat completion request to vLLM server using /v1/chat/completions endpoint
    
    Args:
        url: Server URL
        prompt: Input prompt/message
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        stream: Whether to stream the response
        **kwargs: Additional parameters for the API
        
    Returns:
        - If stream=False: Dictionary containing the API response or None if request fails
        - If stream=True: Generator yielding response chunks or None if request fails
    """
    headers = {
        "User-Agent": "Test Client",
        "Content-Type": "application/json"
    }
    
    if type(input_data) == str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": input_data}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs
        }
    elif type(input_data) == list:
        for temp_data in input_data:
            assert 'content' in temp_data.keys()
        payload = {
            "model": model,
            "messages": input_data,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs
        }
    else:
        raise ValueError('Unknown type of input data: {}'.format(type(input_data)))
    
    try:
        print(f"VLLM请求开始: URL={url}")
        start_time = time.time()
        
        # 设置请求超时，防止永久阻塞
        timeout = kwargs.get('timeout', 30)  # 默认30秒超时
        
        response = requests.post(
            f'{url}/v1/chat/completions',
            headers=headers,
            json=payload,
            stream=stream,  # 启用流式请求
            timeout=timeout  # 设置超时
        )
        
        request_time = time.time() - start_time
        print(f"VLLM请求耗时: {request_time:.2f}秒")
        
        if response.status_code != 200:
            error_msg = f"VLLM请求错误 {response.status_code}: {response.text}"
            print(error_msg)
            return None
            
        if stream:
            def generate_stream():
                stream_start = time.time()
                chunks_count = 0
                chars_count = 0
                try:
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith('data:'):
                                data = decoded_line[5:].strip()
                                if data == '[DONE]':
                                    break
                                try:
                                    chunk = json.loads(data)
                                    if 'choices' in chunk and chunk['choices']:
                                        content = chunk['choices'][0].get('delta', {}).get('content', '')
                                        if content:
                                            chunks_count += 1
                                            chars_count += len(content)
                                            yield content
                                except json.JSONDecodeError:
                                    print(f"JSON解析错误: {decoded_line}")
                                    continue
                finally:
                    stream_total = time.time() - stream_start
                    print(f"VLLM流处理完成: 收到{chunks_count}个块，共{chars_count}个字符，耗时{stream_total:.2f}秒")
                    response.close()
            
            return generate_stream()
        else:
            result = response.json()
            try:
                content = result['choices'][0]['message']['content']
                content_length = len(content) if content else 0
                print(f"VLLM响应成功: {content_length}个字符")
                return content
            except KeyError as e:
                print(f"VLLM响应解析错误: {e}, 响应体: {result}")
                return None
                
    except requests.exceptions.Timeout:
        print(f"VLLM请求超时")
        return None
    except requests.exceptions.ConnectionError:
        print(f"VLLM连接错误: 无法连接到 {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"VLLM请求失败: {e}")
        return None
    except Exception as e:
        print(f"VLLM未知错误: {e}")
        return None



# vllm_reward(url='http://0.0.0.0:8001', mode='sleep')
# vllm_generate(url='http://0.0.0.0:8001', prompt='hi')