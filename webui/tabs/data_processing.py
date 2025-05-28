"""
数据处理模块
"""

import re
import os
import json
import gradio as gr
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

DEFAULT_PREFIX = """Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

def get_project_path(current_project_path):
    """从显示文本中提取项目路径，并返回数据目录路径
    
    Args:
        current_project_path: 当前项目路径显示组件
    
    Returns:
        str: 项目数据目录路径，如果未选择项目则返回空字符串
    """
    path_text = current_project_path

    if path_text == "**项目路径**: -":
        return ""
    
    # 获取项目根路径
    project_path = path_text.replace("**项目路径**: ", "")
    
    # 返回数据目录路径
    data_path = os.path.join(project_path, "data")
    
    # 确保数据目录存在
    os.makedirs(data_path, exist_ok=True)
    #print(f"Debug - Data path: {data_path}")  # 添加打印调试
    
    return data_path
def make_prefix(question: str, template_type: str, custom_prefix: str = None) -> str:
    if template_type == 'base':
        prefix = custom_prefix if custom_prefix else DEFAULT_PREFIX
        return prefix.format(question=question)
    else:
        raise NotImplementedError(f"Template type {template_type} not implemented")

def process_data(raw_data: str, template_type: str, split: str, custom_prefix: str = None) -> tuple[List[Dict[str, Any]], str]:
    # Parse raw data (assuming it's in a format where each line is a JSON object)
    data_list = []
    status_messages = []
    total_lines = len(raw_data.strip().split('\n'))
    processed_lines = 0
    error_lines = 0
    
    for idx, line in enumerate(raw_data.strip().split('\n')):
        try:
            if not line.strip():  # Skip empty lines
                continue
            example = json.loads(line)  # Parse JSON instead of using eval
            question = example['question'].strip()
            if question[-1] != '?':
                question += '?'
            
            prefix = make_prefix(question, template_type, custom_prefix)
            
            data_item = {
                "data_source": "nq",
                "prompt": [{
                    "role": "user",
                    "content": prefix,
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "target": example['golden_answers'],
                    }
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            data_list.append(data_item)
            processed_lines += 1
        except json.JSONDecodeError as e:
            error_lines += 1
            status_messages.append(f"Error in line {idx + 1}: Invalid JSON format - {str(e)}")
        except KeyError as e:
            error_lines += 1
            status_messages.append(f"Error in line {idx + 1}: Missing required field {str(e)}")
        except Exception as e:
            error_lines += 1
            status_messages.append(f"Error in line {idx + 1}: {str(e)}")
    
    status_summary = [
        f"处理完成:",
        f"- 总行数: {total_lines}",
        f"- 成功处理: {processed_lines}",
        f"- 错误行数: {error_lines}",
        f"- 成功率: {(processed_lines/total_lines)*100:.1f}%"
    ]
    
    if status_messages:
        status_summary.append("\n详细错误信息:")
        status_summary.extend(status_messages)
    
    return data_list, "\n".join(status_summary)

def save_to_parquet(data_list: List[Dict[str, Any]], output_path: str, current_project_path: gr.Markdown, split: str = "train") -> str:
    """保存数据到parquet文件
    
    Args:
        data_list: 要保存的数据列表
        output_path: 用户指定的输出路径（可选）
        current_project_path: 当前项目路径组件
        split: 数据集分割类型（train/test）
    
    Returns:
        str: 保存状态信息
    """
    try:
        if output_path:
            print(output_path)
            df = pd.DataFrame(data_list)
            df.to_parquet(output_path)
        else:
            # 获取项目数据目录
            project_data_path = get_project_path(current_project_path)
            if not project_data_path:
                return "错误：未选择项目"
            
            # 创建data目录
            data_dir = os.path.join(project_data_path, "data")
            os.makedirs(data_dir, exist_ok=True)
            
            # 生成文件名：split_YYYYMMDD_HHMMSS.parquet
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{split}_{timestamp}.parquet"
            output_path = os.path.join(data_dir, filename)
            
            print(f"保存到: {output_path}")
            df = pd.DataFrame(data_list)
            df.to_parquet(output_path)
        
        return f"成功保存 {len(data_list)} 条记录到 {output_path}"
    except Exception as e:
        return f"保存文件时出错: {str(e)}"

def process_data_from_file(file_path: str, template_type: str, split: str, custom_prefix: str = None, ground_truth_keys: List[str] = None, question_key: str = 'question') -> tuple[List[Dict[str, Any]], str]:
    """从文件流式处理数据
    
    Args:
        file_path: 输入文件路径
        template_type: 模板类型
        split: 数据集分割
        custom_prefix: 自定义前缀
        ground_truth_keys: 作为ground_truth的字段列表
        question_key: 作为question的字段名
    
    Returns:
        tuple: (处理后的数据列表, 状态信息)
    """
    data_list = []
    status_messages = []
    processed_lines = 0
    error_lines = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    if not line.strip():  # Skip empty lines
                        continue
                    example = json.loads(line)
                    
                    # 使用选定的字段作为question
                    if question_key not in example:
                        raise KeyError(f"Question field '{question_key}' not found")
                    question = example[question_key].strip()
                    if question[-1] != '?':
                        question += '?'
                    
                    prefix = make_prefix(question, template_type, custom_prefix)
                    
                    # 获取ground_truth值
                    ground_truth_values = []
                    if ground_truth_keys:
                        for key in ground_truth_keys:
                            if key in example:
                                ground_truth_values.append(example[key])
                            else:
                                status_messages.append(f"Warning in line {idx + 1}: Key '{key}' not found")
                    else:
                        # 如果没有指定keys，使用默认的golden_answers
                        ground_truth_values = example.get('golden_answers', [])
                    
                    data_item = {
                        "data_source": "nq",
                        "prompt": [{
                            "role": "user",
                            "content": prefix,
                        }],
                        "ability": "fact-reasoning",
                        "reward_model": {
                            "style": "rule",
                            "ground_truth": {
                                "target": ground_truth_values,
                            }
                        },
                        "extra_info": {
                            'split': split,
                            'index': idx,
                        }
                    }
                    data_list.append(data_item)
                    processed_lines += 1
                    
                    # 每处理1000行打印一次进度
                    if processed_lines % 1000 == 0:
                        print(f"已处理 {processed_lines} 行数据")
                        
                except json.JSONDecodeError as e:
                    error_lines += 1
                    status_messages.append(f"Error in line {idx + 1}: Invalid JSON format - {str(e)}")
                except KeyError as e:
                    error_lines += 1
                    status_messages.append(f"Error in line {idx + 1}: Missing required field {str(e)}")
                except Exception as e:
                    error_lines += 1
                    status_messages.append(f"Error in line {idx + 1}: {str(e)}")
    
    except Exception as e:
        return [], f"读取文件时出错: {str(e)}"
    
    status_summary = [
        f"处理完成:",
        f"- 总行数: {processed_lines + error_lines}",
        f"- 成功处理: {processed_lines}",
        f"- 错误行数: {error_lines}",
        f"- 成功率: {(processed_lines/(processed_lines + error_lines))*100:.1f}%"
    ]
    
    if status_messages:
        status_summary.append("\n详细错误信息:")
        status_summary.extend(status_messages)
    
    return data_list, "\n".join(status_summary)

def read_parquet_file(file_path: str) -> pd.DataFrame:
    """读取parquet文件并返回DataFrame
    
    Args:
        file_path: parquet文件路径
    
    Returns:
        pd.DataFrame: 读取的数据
    """
    try:
        # 使用pyarrow引擎读取，支持流式读取大文件
        df = pd.read_parquet(file_path, engine='pyarrow')
        return df.head(5)
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

def get_available_keys(file_path: str) -> List[str]:
    """从输入文件中获取可用的字段名
    
    Args:
        file_path: 输入文件路径
    
    Returns:
        List[str]: 可用的字段名列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 读取前几行来获取字段名
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    return list(data.keys())
    except Exception as e:
        print(f"获取字段名时出错: {str(e)}")
    return []

def create_data_processing_tab(current_project_path):
    """数据处理标签页
    
    该标签页用于管理和处理实验数据，包括：
    - 数据导入导出
    - 数据预处理
    - 数据可视化
    - 数据集管理
    """
    
    with gr.Blocks() as tab:
        gr.Markdown("# 数据处理")
        gr.Markdown("""
        ## 功能说明
        在此标签页中，您可以：
        - 导入和导出实验数据
        - 进行数据预处理和转换
        - 查看数据可视化结果
        - 管理实验数据集
        """)
        
        with gr.Tabs() as tabs:
            with gr.TabItem("数据预处理"):
                gr.Markdown("# 数据处理")
                gr.Markdown("""
                ## 使用说明
                1. 您可以直接输入JSON文件的路径
                2. 每行应该是一个JSON对象，包含问题字段和答案字段
                3. 示例格式:
                ```json
                {"query": "What is the capital of France?", "golden_answers": ["Paris"]}
                {"query": "Who wrote Romeo and Juliet?", "golden_answers": ["William Shakespeare"]}
                ```
                4. 您可以通过勾选"使用自定义前缀"并输入模板来自定义前缀
                5. 在模板中使用 {question} 作为问题的占位符
                6. 您可以从可用字段中选择一个作为问题字段，以及一个或多个作为ground_truth的值
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 输入数据")
                            file_path = gr.Textbox(
                                label="输入文件路径",
                                placeholder="请输入JSON文件的完整路径",
                            )
                            load_keys_btn = gr.Button("加载字段", variant="secondary")
                        
                        with gr.Group():
                            gr.Markdown("### 处理设置")
                            
                            template_type = gr.Dropdown(
                                choices=["base"],
                                value="base",
                                label="模板类型"
                            )
                            question_key = gr.Dropdown(
                                label="问题字段",
                                choices=[],
                                interactive=True,
                                visible=False
                            )
                            available_keys = gr.Dropdown(
                                label="Ground Truth字段",
                                choices=[],
                                multiselect=True,
                                interactive=True,
                                visible=False
                            )
                            use_custom_prefix = gr.Checkbox(
                                label="使用自定义前缀",
                                value=False
                            )
                            custom_prefix = gr.Textbox(
                                label="自定义前缀模板",
                                placeholder="在此输入自定义前缀模板。使用 {question} 作为问题的占位符。",
                                lines=5,
                                value=DEFAULT_PREFIX,
                                visible=False
                            )
                            split = gr.Dropdown(
                                choices=["train", "test"],
                                value="train",
                                label="数据集分割"
                            )
                            
                            output_path = gr.Textbox(
                                label="输出路径",
                                placeholder="输入parquet文件的保存路径，默认保存到项目目录下的data目录中",
                            )
                            process_btn = gr.Button("处理并保存", variant="primary")
                    
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 处理状态")
                            output = gr.Textbox(
                                label="处理结果",
                                lines=20,
                                show_copy_button=True
                            )
            
            with gr.TabItem("数据可视化"):
                gr.Markdown("""
                ## 数据可视化功能
                在此标签页中，您可以：
                - 读取并预览Parquet格式的数据文件
                - 查看数据的表格形式展示
                - 直观地了解数据结构和内容
                
                ### 使用说明
                1. 输入Parquet文件的完整路径
                2. 点击"预览数据"按钮查看数据内容
                3. 在数据预览区域可以查看表格数据
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 读取文件")
                            parquet_file_path = gr.Textbox(
                                label="Parquet文件路径",
                                placeholder="请输入Parquet文件的完整路径",
                            )
                            preview_btn = gr.Button("预览数据", variant="primary")
                    
                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("### 数据预览")
                            preview_output = gr.Dataframe(
                                label="Parquet文件内容",
                                wrap=True,
                            )
                            preview_status = gr.Textbox(
                                label="预览状态",
                                interactive=False,
                                lines=2
                            )
            
            with gr.TabItem("数据集管理"):
                gr.Markdown("### 数据集管理功能开发中...")
        
        def load_available_keys(file_path: str):
            if not file_path:
                return gr.update(choices=[], visible=False), gr.update(choices=[], visible=False), "请输入文件路径"
            if not os.path.exists(file_path):
                return gr.update(choices=[], visible=False), gr.update(choices=[], visible=False), f"文件不存在: {file_path}"
            
            keys = get_available_keys(file_path)
            if not keys:
                return gr.update(choices=[], visible=False), gr.update(choices=[], visible=False), "无法从文件中获取字段名"
            
            return gr.update(choices=keys, visible=True), gr.update(choices=keys, visible=True), f"成功加载字段: {', '.join(keys)}"
        
        def process_and_save_from_path(file_path: str, template_type: str, split: str, output_path: str, use_custom_prefix: bool, custom_prefix: str, current_project_path: gr.Markdown, selected_keys: List[str], question_key: str):
            prefix = custom_prefix if use_custom_prefix else None
            processed_data, status = process_data_from_file(file_path, template_type, split, prefix, selected_keys, question_key)
            save_status = save_to_parquet(processed_data, output_path, current_project_path, split)
            return f"{status}\n\n{save_status}"
        
        def preview_parquet_file(file_path: str):
            try:
                if not file_path:
                    return pd.DataFrame(), "请输入文件路径"
                if not os.path.exists(file_path):
                    return pd.DataFrame(), f"文件不存在: {file_path}"
                df = read_parquet_file(file_path)
                return df, f"成功读取文件: {file_path}\n显示行数: {len(df)}"
            except Exception as e:
                return pd.DataFrame(), f"读取文件时出错: {str(e)}"
        
        # Handle loading available keys
        load_keys_btn.click(
            fn=load_available_keys,
            inputs=[file_path],
            outputs=[question_key, available_keys, output]
        )
        
        # Handle file processing
        process_btn.click(
            fn=process_and_save_from_path,
            inputs=[file_path, template_type, split, output_path, use_custom_prefix, custom_prefix, current_project_path, available_keys, question_key],
            outputs=output
        )
        
        # Handle parquet preview
        preview_btn.click(
            fn=preview_parquet_file,
            inputs=[parquet_file_path],
            outputs=[preview_output, preview_status]
        )
        
        # Handle custom prefix visibility
        use_custom_prefix.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[use_custom_prefix],
            outputs=[custom_prefix]
        )
    
    return tab 