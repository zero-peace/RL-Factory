import re
import time
import asyncio
import aiohttp
from easydict import EasyDict
from generator import get_generator

key_list = [
    '1734834926589104200',
    '1796110759362199605',
    '1794984524074676265',
    '1792464990918938657',
    '1789931624457855004',
    '1789903374117359714',
    '1787822145012584505',
    '1785146944130732056',
    '1785146918596018242',
    '1785146889328074779',
    '1785146861184548942',
    '1785146832369369093',
    '1783462889744248849',
    '1783462795028553741'
]

async def web_search(queries):
    url = 'https://aigc.sankuai.com/v1/friday/api/search'
    
    async def fetch_search_results(query):
        data = {
            "query": query,
            "api": "bing-search",
            "top_k": 3
        }
        current_key_index = 0
        max_trial, trial_id = 20, 0
        
        while trial_id < max_trial:
            headers = {
                'Authorization': f'Bearer {key_list[current_key_index]}',
                'Content-Type': 'application/json'
            }
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=data) as response:
                        if response.status == 200:
                            return query, response.status, await response.json()
                        else:
                            current_key_index = (current_key_index + 1) % len(key_list)
                            trial_id += 1
                            if current_key_index == 0:
                                await asyncio.sleep(5)
            except Exception as e:
                print(f"搜索请求出错: {e}")
                current_key_index = (current_key_index + 1) % len(key_list)
                trial_id += 1
                if current_key_index == 0:
                    await asyncio.sleep(5)
        
        return query, None, {'error': 'Exceed rate limit'}
    
    responses = {}
    tasks = [fetch_search_results(query) for query in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if isinstance(result, Exception):
            print(f"搜索任务出错: {result}")
            continue
        query, status_code, result_data = result
        responses[query] = (status_code, result_data)
    
    return responses

def extract_questions(solution_str):
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        processed_str = solution_str

    questions_pattern = r'<questions>(.*?)</questions>'
    matches = list(re.finditer(questions_pattern, processed_str, re.DOTALL))
    
    if not matches:
        return None, processed_str
    
    final_questions = matches[-1].group(1).strip()
    return final_questions, processed_str

async def evaluate_single_question(query, address, question, search_results, generator, maximum_search_num=3):
    search_results = search_results[1]['results']
    search_results_str = ''
    for temp_idx, temp_search_result in enumerate(search_results[:maximum_search_num]):
        search_results_str += f"""搜索结果 {temp_idx + 1}
标题：{temp_search_result['title']}
内容：
{temp_search_result['snippet']}
"""

    system_template = """你是智能搜索助手的小帮手，负责评价搜索结果的好坏，这些评价结果将被用于调整搜索问题，并使用搜索得到的相关信息更好地回复用户。

# 问题说明
1. 用户提供了一个搜索query，而另外一个大模型将这一query改写为了一个具体的问题question，希望能够通过这个问题的搜索结果为用户提供更多的信息。
2. 现在我使用这个问题搜索得到了一些网页结果，你需要评判这些网页结果是否为解决用户的query提供了有效的信息。

# 评价流程
1. 意图识别：你需要首先根据用户的查询query识别用户的意图，你需要尽可能全面地揣测用户意图以及query中的相关信息
2. 网页结果分析：你需要分析这些网页提供了什么新的信息
3. 搜索结果评价：你需要根据网页提供的信息和用户的意图，分析这个问题及对应的搜索结果是否为用户提供了新的信息，进而能够帮助用户更好地进行决策
4. 评价输出：你需要按照如下格式输出评价结果
```json
{
    '评价': '好' or '坏',
}
```
Note 1: 你只能输出'好'和'坏'中的一个，如果提供了有效的信息则输出'好'，否则输出'坏'
"""
    user_template = """下面是用户提供的query：
{}
用户所处的地点为：
{}
下面是其他模型改写的question：
{}
下面是网页搜索的结果：
{}
请你严格按照评价流程，逐步输出。最终按照给定的json输出最终结果。"""
    message = [
        {'role': 'system', 'content': system_template},
        {'role': 'system', 'content': user_template.format(query, address, question, search_results_str)}
    ]
    
    success, result = False, 0
    while not success:
        response = await generator.generate_async(input_data=message, temperature=0.0)
        pattern = r'```json\s*(\{.*?\})\s*```'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            extracted_info = match.group(1)
            try:
                eval_result = eval(extracted_info)
                result = 1 if eval_result['评价'] == '好' else 0
                success = True
            except:
                print("extracted_info: ", extracted_info)

    return result

async def evaluate_questions(query, address, questions_list, generator, maximum_search_num=3):
    results = await web_search(queries=questions_list[:1])

    links = []
    for temp_result in results.values():
        try:
            for temp_link in temp_result[1]['results']:
                links.append(temp_link['link'])
        except:
            pass
    
    if len(links) == 0:
        return -1
    else:
        repeated_ratio = 1 - len(set(links)) / len(links)
    
    repeated_score = (-0.25 * maximum_search_num) * repeated_ratio

    total_score = 0
    tasks = []
    for question, search_result in results.items():
        task = evaluate_single_question(
            query, address, question, search_result, generator, maximum_search_num
        )
        tasks.append(task)
    
    scores = await asyncio.gather(*tasks, return_exceptions=True)
    for score in scores:
        if isinstance(score, Exception):
            print(f"评估问题出错: {score}")
            continue
        total_score += score

    return 2.0 * (total_score + repeated_score) / maximum_search_num

def validate_response_structure(processed_str: str) -> bool:
    validation_passed = True

    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'questions_start': ('<questions>', 1),
        'questions_end': ('</questions>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        if count != expected_count:
            validation_passed = False

    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['questions_start'] or
        positions['questions_start'] > positions['questions_end']):
        validation_passed = False

    return validation_passed

def validate_questions_structure(questions_text: str) -> bool:
    pattern = r'```json\s*(\{.*?\})\s*```'
    match = re.match(pattern, questions_text, flags=re.DOTALL)
    if match:
        content = match.group(1)
        try:
            content_dict = eval(content)
            return content_dict['改写结果']
        except:
            return None
    else:
        return None

def compute_score(data_source, solution_str, ground_truth, format_reward=0.5, extra_info=None):
    # 创建一个新的事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # 在同步函数中运行异步代码
        score = loop.run_until_complete(_compute_score_async(
            data_source, solution_str, ground_truth, format_reward, extra_info
        ))
        return score
    finally:
        # 清理事件循环
        loop.close()

async def _compute_score_async(data_source, solution_str, ground_truth, format_reward=0.5, extra_info=None):
    generator = get_generator('api')(
        config=EasyDict({
            'api_method': 'local',
            'port': extra_info['port'],
            'model_name': extra_info['model_name']
        })
    )
    
    query, address = ground_truth['query'], ground_truth['address']
    questions_text, processed_str = extract_questions(solution_str=solution_str)
    format_correct = validate_response_structure(processed_str)
    format_score = format_reward if format_correct else -abs(format_reward)

    answer_score = 0
    if format_correct and questions_text:
        questions_list = validate_questions_structure(questions_text=questions_text)
        if questions_list is None:
            answer_score = -1.5
        else:
            answer_score = await evaluate_questions(query, address, questions_list, generator)
    else:
        answer_score = -2
    
    total_score = format_score + answer_score
    return total_score

def _compute_score(data_source, solution_str, ground_truth, format_reward=0.5, extra_info=None):
    time.sleep(2)
    return 1.0

