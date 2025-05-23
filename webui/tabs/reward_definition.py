import gradio as gr
import json
import os
from pathlib import Path
from typing import Dict, Any

from components.rewards.graders import GraderRegistry

def create_reward_definition_tab():
    """奖赏定义主标签页"""
    with gr.Blocks() as tab:
        gr.Markdown("# 奖赏定义")
        
        # 创建子标签页
        with gr.Tabs() as subtabs:
            with gr.TabItem("规则定义"):
                rule_components = create_rule_definition_tab()
            
            with gr.TabItem("模型评判"):
                model_components = create_model_evaluation_tab()
            
            with gr.TabItem("验证工具"):
                validation_components = create_validation_tools_tab()
        
        # 导出按钮和结果显示
        with gr.Row():
            export_json = gr.Button("导出配置文件")
            export_python = gr.Button("生成Python文件")
        
        output_json = gr.JSON(label="配置文件预览")
        output_python = gr.Code(language="python", label="生成的Python代码")
        
        # 处理导出事件
        def export_json_handler():
            config = generate_reward_json(
                rule_components,
                model_components,
                validation_components
            )
            # 保存到文件
            os.makedirs("rewards", exist_ok=True)
            json_path = f"rewards/{rule_components['reward_name']}.json"
            with open(json_path, "w") as f:
                json.dump(config, f, indent=2)
            return config
        
        def export_python_handler(config):
            python_code = generate_reward_python(config)
            # 保存到文件
            os.makedirs("rewards", exist_ok=True)
            py_path = f"rewards/{config['name']}.py"
            with open(py_path, "w") as f:
                f.write(python_code)
            return python_code
        
        export_json.click(
            export_json_handler,
            outputs=output_json
        )
        
        export_python.click(
            export_python_handler,
            inputs=output_json,
            outputs=output_python
        )
    
    return tab


def create_rule_definition_tab():
    """规则定义子标签页"""
    with gr.Blocks() as tab:
        gr.Markdown("## 规则定义")
        
        with gr.Row():
            with gr.Column():
                reward_name = gr.Textbox(label="奖赏函数名称", placeholder="输入奖赏函数的名称")
                reward_description = gr.Textbox(label="奖赏函数描述", lines=3, placeholder="描述该奖赏函数的功能和用途")
                
                gr.Markdown("### 基础规则设置")
                with gr.Row():
                    min_reward = gr.Number(label="最小奖赏值", value=-1.0)
                    max_reward = gr.Number(label="最大奖赏值", value=1.0)
                
                reward_type = gr.Radio(
                    choices=["离散", "连续"],
                    label="奖赏类型",
                    value="连续"
                )
                
                custom_rules = gr.TextArea(
                    label="自定义规则",
                    placeholder="使用Python代码定义自定义规则...",
                    lines=10
                )
        
        return {
            "reward_name": reward_name,
            "reward_description": reward_description,
            "min_reward": min_reward,
            "max_reward": max_reward,
            "reward_type": reward_type,
            "custom_rules": custom_rules
        }


def create_model_evaluation_tab():
    """模型评判子标签页"""
    with gr.Blocks() as tab:
        gr.Markdown("## 模型评判")
        
        with gr.Row():
            with gr.Column():
                use_model = gr.Checkbox(label="使用模型进行评判", value=False)
                
                # 获取所有已注册的评分器
                graders = GraderRegistry.list_graders()
                
                with gr.Group() as model_group:
                    grader_type = gr.Dropdown(
                        choices=list(graders.keys()),
                        label="评分器类型",
                        interactive=True
                    )
                    
                    # 显示评分器描述
                    grader_description = gr.Markdown()
                    
                    def update_description(grader_name):
                        if grader_name in graders:
                            return f"**评分器说明**：{graders[grader_name]}"
                        return ""
                    
                    grader_type.change(
                        fn=update_description,
                        inputs=[grader_type],
                        outputs=[grader_description]
                    )
                    
                    model_config = gr.JSON(
                        label="评分器配置", 
                        value={},
                        visible=False
                    )
                    
                    # 测试区域
                    with gr.Row():
                        test_input = gr.Textbox(label="测试输入")
                        test_reference = gr.Textbox(label="参考答案")
                    
                    test_button = gr.Button("测试评分")
                    test_result = gr.Number(label="评分结果", value=0.0)
                    
                    def test_grader(grader_name: str, test_input: str, test_reference: str) -> float:
                        try:
                            grader_class = GraderRegistry.get(grader_name)
                            grader = grader_class()
                            return grader.grade(test_input, test_reference)
                        except Exception as e:
                            print(f"评分测试出错：{e}")
                            return 0.0
                    
                    test_button.click(
                        fn=test_grader,
                        inputs=[grader_type, test_input, test_reference],
                        outputs=[test_result]
                    )
        
        return {
            "use_model": use_model,
            "grader_type": grader_type,
            "model_config": model_config
        }


def create_validation_tools_tab():
    """验证工具子标签页"""
    with gr.Blocks() as tab:
        gr.Markdown("## 验证工具")
        
        with gr.Row():
            with gr.Column():
                test_data = gr.File(label="上传测试数据")
                validation_method = gr.Radio(
                    choices=["单步验证", "回合验证", "完整轨迹验证"],
                    label="验证方法",
                    value="单步验证"
                )
                
                with gr.Row():
                    run_validation = gr.Button("运行验证")
                    export_results = gr.Button("导出结果")
                
                validation_output = gr.TextArea(label="验证结果", interactive=False)
        
        return {
            "test_data": test_data,
            "validation_method": validation_method,
            "run_validation": run_validation,
            "export_results": export_results,
            "validation_output": validation_output
        }


def generate_reward_json(rule_data: Dict[str, Any], model_data: Dict[str, Any], validation_data: Dict[str, Any]) -> Dict[str, Any]:
    """生成奖赏配置JSON"""
    reward_config = {
        "name": rule_data["reward_name"],
        "description": rule_data["reward_description"],
        "type": rule_data["reward_type"],
        "range": {
            "min": rule_data["min_reward"],
            "max": rule_data["max_reward"]
        },
        "custom_rules": rule_data["custom_rules"],
        "grader": {
            "enabled": model_data["use_model"],
            "type": model_data["grader_type"],
            "config": model_data["model_config"]
        },
        "validation": {
            "method": validation_data["validation_method"]
        }
    }
    return reward_config

def generate_reward_python(reward_config: Dict[str, Any]) -> str:
    """根据配置生成Python奖赏函数"""
    grader_import = ""
    grader_init = ""
    grader_code = ""
    
    if reward_config["grader"]["enabled"]:
        grader_import = "from components.rewards.graders import GraderRegistry"
        grader_init = f"""
        # 初始化评分器
        grader_class = GraderRegistry.get("{reward_config["grader"]["type"]}")
        self.grader = grader_class()"""
        grader_code = """
        # 使用评分器计算奖赏
        if hasattr(self, 'grader'):
            reward = self.grader.grade(next_state, info.get('reference')) if info and 'reference' in info else 0.0"""
    
    template = f'''
import numpy as np
{grader_import}

class {reward_config["name"]}:
    """
    {reward_config["description"]}
    """
    def __init__(self):
        self.min_reward = {reward_config["range"]["min"]}
        self.max_reward = {reward_config["range"]["max"]}
        self.reward_type = "{reward_config["type"]}"
        {grader_init}
        
    def calculate_reward(self, state, action, next_state, info=None):
        """计算奖赏值"""
        reward = 0.0
        {grader_code}
        
        # 自定义规则
{reward_config["custom_rules"]}
        
        # 确保奖赏在范围内
        reward = np.clip(reward, self.min_reward, self.max_reward)
        return reward
        
    def reset(self):
        """重置奖赏函数状态"""
        pass
'''
    return template
