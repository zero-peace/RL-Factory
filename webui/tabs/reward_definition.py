import gradio as gr
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from components.rewards.graders import GraderRegistry

# 在文件开头定义类型映射
REQUIREMENT_TYPE_MAP = {
    "count": "数量限制",
    "length": "内容长度",
    "format": "内容格式"
}

RULE_TYPE_MAP = {
    "思考过程": "think",
    "结果标签": "answer",
    "工具标签": "tool_call"
}

# 添加验证器映射
VALIDATOR_TYPE_MAP = {
    "无": None,
    "JSON格式": "json_validator",
    "XML格式": "xml_validator",
    "Python代码": "python_validator",
    "URL格式": "url_validator",
    "邮箱格式": "email_validator"
}

def get_available_requirement_types(requirements: List[Dict], editing_requirement: Dict = None) -> List[str]:
    """获取可用的要求类型列表"""
    all_types = list(REQUIREMENT_TYPE_MAP.values())
    used_types = {REQUIREMENT_TYPE_MAP[r["type"]] for r in requirements}
    
    # 如果是编辑模式，不要排除当前正在编辑的要求的类型
    if editing_requirement:
        used_types.discard(REQUIREMENT_TYPE_MAP[editing_requirement["type"]])
    
    return [t for t in all_types if t not in used_types]

def get_available_rule_types(rules: List[Dict], editing_rule: Dict = None) -> List[str]:
    """获取可用的规则类型列表"""
    all_types = ["思考过程", "结果标签", "工具标签", "自定义标签"]
    used_labels = {r["label"] for r in rules}
    
    # 如果是编辑模式，不要排除当前正在编辑的规则的标签
    if editing_rule:
        used_labels.discard(editing_rule["label"])
    
    # 如果预定义的标签已经使用，则从可选列表中移除对应的类型
    if "think" in used_labels:
        all_types.remove("思考过程")
    if "answer" in used_labels:
        all_types.remove("结果标签")
    if "tool_call" in used_labels:
        all_types.remove("工具标签")
    
    return all_types

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
            json_path = f"rewards/reward_config.json"
            with open(json_path, "w") as f:
                json.dump(config, f, indent=2)
            return config
        
        def export_python_handler(config):
            python_code = generate_reward_python(config)
            # 保存到文件
            os.makedirs("rewards", exist_ok=True)
            py_path = f"rewards/reward_function.py"
            with open(py_path, "w") as f:
                f.write(python_code)
            return python_code
        
        export_json.click(
            fn=export_json_handler,
            outputs=output_json
        )
        
        export_python.click(
            fn=export_python_handler,
            inputs=output_json,
            outputs=output_python
        )
    
    return tab


def create_requirements_ui():
    """创建标签要求配置界面"""
    with gr.Group() as requirements_group:
        with gr.Row(equal_height=True):
            # 要求类型选择
            requirement_type = gr.Dropdown(
                choices=list(REQUIREMENT_TYPE_MAP.values()),
                label="添加要求",
                value=None,
                interactive=True,
                scale=4
            )
            add_count = gr.Button("添加数量限制", visible=False, scale=1)
            add_length = gr.Button("添加长度限制", visible=False, scale=1)
            add_format = gr.Button("添加格式限制", visible=False, scale=1)
        
        # 编辑模式标记
        edit_mode = gr.State({
            "active": False,
            "index": None
        })
        
        # 数量限制配置
        with gr.Group(visible=False) as count_group:
            with gr.Row():
                count_min = gr.Number(label="最小数量", value=1, minimum=0)
                count_max = gr.Number(label="最大数量", value=1, minimum=1)
        
        # 内容长度配置
        with gr.Group(visible=False) as length_group:
            with gr.Row():
                length_min = gr.Number(label="最小长度", value=None, minimum=0)
                length_max = gr.Number(label="最大长度", value=512, minimum=1)
            with gr.Row():
                length_mode = gr.Radio(
                    choices=["均值", "最大值", "最小值", "求和"],
                    label="奖赏计算模式",
                    value="均值"
                )
                length_coefficient = gr.Slider(
                    label="系数",
                    minimum=0.0,
                    maximum=10.0,
                    value=1.0,
                    step=0.1
                )
        
        # 内容格式配置
        with gr.Group(visible=False) as format_group:
            format_type = gr.Radio(
                choices=["json", "xml"],
                label="格式类型",
                value="json"
            )
            format_example = gr.Code(
                label="格式样例",
                language="json"
            )
            with gr.Row():
                format_mode = gr.Radio(
                    choices=["均值", "最大值", "最小值", "求和"],
                    label="奖赏计算模式",
                    value="均值"
                )
                format_coefficient = gr.Slider(
                    label="系数",
                    minimum=0.0,
                    maximum=10.0,
                    value=1.0,
                    step=0.1
                )
        
        # 已添加的要求列表
        requirements_list = gr.State([])
        requirements_display = gr.DataFrame(
            headers=["要求类型", "配置"],
            label="已添加的要求",
            interactive=False,
            visible=True,
            wrap=True
        )
        with gr.Row(equal_height=True):
            edit_button = gr.Button("✏️ 编辑", visible=False, size="sm", scale=1)
            delete_button = gr.Button("🗑️ 删除", visible=False, size="sm", variant="stop", scale=1)
        selected_row = gr.State(None)  # 存储选中的行索引
        
        def update_requirement_groups(req_type: Optional[str], current_reqs: List[Dict], edit_state: Dict) -> Dict:
            """更新要求配置组的可见性"""
            # 如果是编辑模式，不检查唯一性
            if not edit_state["active"]:
                # 检查要求类型是否已存在
                existing_types = {req["type"] for req in current_reqs}
                type_map = {
                    "数量限制": "count",
                    "内容长度": "length",
                    "内容格式": "format"
                }
                
                if req_type and type_map[req_type] in existing_types:
                    gr.Warning(f"{req_type}已经添加过了")
                    return {
                        count_group: gr.update(visible=False),
                        length_group: gr.update(visible=False),
                        format_group: gr.update(visible=False),
                        requirement_type: gr.update(value=None)
                    }
            
            return {
                count_group: gr.update(visible=req_type == "数量限制"),
                length_group: gr.update(visible=req_type == "内容长度"),
                format_group: gr.update(visible=req_type == "内容格式")
            }
        
        def add_or_update_requirement(req_type: str, edit_state: Dict, requirements: List[Dict], config_params: Dict):
            """添加或更新要求"""
            type_map = {
                "数量限制": "count",
                "内容长度": "length",
                "内容格式": "format"
            }
            internal_type = type_map[req_type]
            
            # 创建新的要求配置
            if internal_type == "count":
                new_req = {
                    "type": internal_type,
                    "config": {
                        "min": config_params["count_min"],
                        "max": config_params["count_max"]
                    }
                }
            elif internal_type == "length":
                new_req = {
                    "type": internal_type,
                    "config": {
                        "min": config_params["length_min"],
                        "max": config_params["length_max"],
                        "mode": config_params["length_mode"],
                        "coefficient": config_params["length_coefficient"]
                    }
                }
            else:  # format
                new_req = {
                    "type": internal_type,
                    "config": {
                        "type": config_params["format_type"],
                        "example": config_params["format_example"],
                        "mode": config_params["format_mode"],
                        "coefficient": config_params["format_coefficient"]
                    }
                }
            
            # 如果是编辑模式，替换原有要求
            if edit_state["active"] and edit_state["index"] is not None:
                updated_reqs = requirements.copy()
                updated_reqs[edit_state["index"]] = new_req
                edit_state["active"] = False
                edit_state["index"] = None
            else:
                # 检查是否已存在
                if any(req["type"] == internal_type for req in requirements):
                    gr.Warning(f"{req_type}已经添加过了")
                    return (
                        requirements,
                        [[r["type"], json.dumps(r["config"], ensure_ascii=False, indent=2)] for r in requirements],
                        None,
                        gr.update(choices=get_available_requirement_types(requirements), value=None)
                    )
                updated_reqs = requirements + [new_req]
            
            # 更新显示数据
            display_data = [
                [r["type"], json.dumps(r["config"], ensure_ascii=False, indent=2)]
                for r in updated_reqs
            ]
            
            # 获取更新后的可用要求类型
            available_types = get_available_requirement_types(updated_reqs)
            
            return (
                updated_reqs,
                display_data,
                None,  # 清空要求类型选择
                gr.update(choices=available_types, value=None)  # 更新要求类型的选项
            )
        
        def select_requirement(evt: gr.SelectData, requirements: List[Dict]) -> Dict:
            """选择要求进行编辑或删除"""
            row_index = evt.index[0]
            return {
                edit_button: gr.update(visible=True),
                delete_button: gr.update(visible=True),
                selected_row: row_index
            }
        
        def delete_requirement(row_index: int, requirements: List[Dict]) -> Tuple[List[Dict], List[List[str]], gr.Button, gr.Button]:
            """删除选中的要求"""
            if row_index is None:
                return (
                    requirements,
                    [[r["type"], json.dumps(r["config"], ensure_ascii=False, indent=2)] for r in requirements],
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
            
            updated_reqs = requirements[:row_index] + requirements[row_index + 1:]
            display_data = [
                [r["type"], json.dumps(r["config"], ensure_ascii=False, indent=2)]
                for r in updated_reqs
            ]
            
            return (
                updated_reqs,
                display_data,
                gr.update(visible=False),
                gr.update(visible=False)
            )
        
        def edit_requirement(row_index: int, requirements: List[Dict]) -> Dict:
            """编辑选中的要求"""
            if row_index is None:
                return {}
            
            req = requirements[row_index]
            req_type = req["type"]
            config = req["config"]
            
            # 获取可用的要求类型，包括当前正在编辑的要求类型
            available_types = get_available_requirement_types(requirements, req)
            current_type = REQUIREMENT_TYPE_MAP[req_type]
            if current_type not in available_types:
                available_types.append(current_type)
            
            updates = {}
            
            # 更新要求类型下拉列表
            updates[requirement_type] = gr.update(
                choices=available_types,
                value=current_type
            )
            
            # 隐藏所有配置组
            updates[count_group] = gr.update(visible=False)
            updates[length_group] = gr.update(visible=False)
            updates[format_group] = gr.update(visible=False)
            
            # 根据类型显示和更新对应的配置组
            if req_type == "count":
                updates[count_group] = gr.update(visible=True)
                updates[count_min] = gr.update(value=config["min"])
                updates[count_max] = gr.update(value=config["max"])
            elif req_type == "length":
                updates[length_group] = gr.update(visible=True)
                updates[length_min] = gr.update(value=config["min"])
                updates[length_max] = gr.update(value=config["max"])
                updates[length_mode] = gr.update(value=config["mode"])
                updates[length_coefficient] = gr.update(value=config["coefficient"])
            else:  # format
                updates[format_group] = gr.update(visible=True)
                updates[format_type] = gr.update(value=config["type"])
                updates[format_example] = gr.update(value=config["example"])
                updates[format_mode] = gr.update(value=config["mode"])
                updates[format_coefficient] = gr.update(value=config["coefficient"])
            
            # 更新按钮状态和编辑模式
            updates[edit_button] = gr.update(visible=False)
            updates[delete_button] = gr.update(visible=False)
            updates[edit_mode] = {"active": True, "index": row_index}
            
            return updates
        
        # 绑定事件
        requirement_type.change(
            fn=update_requirement_groups,
            inputs=[requirement_type, requirements_list, edit_mode],
            outputs=[
                count_group,
                length_group,
                format_group,
                add_count,
                add_length,
                add_format,
                requirement_type
            ]
        )
        
        # 添加要求按钮的事件处理
        def wrap_add_count(edit_state: Dict, requirements: List[Dict], count_min: int, count_max: int):
            return add_or_update_requirement(
                "数量限制",
                edit_state,
                requirements,
                {
                    "count_min": count_min,
                    "count_max": count_max
                }
            )
        
        def wrap_add_length(edit_state: Dict, requirements: List[Dict], length_min: Optional[int], length_max: int, length_mode: str, length_coefficient: float):
            return add_or_update_requirement(
                "内容长度",
                edit_state,
                requirements,
                {
                    "length_min": length_min,
                    "length_max": length_max,
                    "length_mode": length_mode,
                    "length_coefficient": length_coefficient
                }
            )
        
        def wrap_add_format(edit_state: Dict, requirements: List[Dict], format_type: str, format_example: str, format_mode: str, format_coefficient: float):
            return add_or_update_requirement(
                "内容格式",
                edit_state,
                requirements,
                {
                    "format_type": format_type,
                    "format_example": format_example,
                    "format_mode": format_mode,
                    "format_coefficient": format_coefficient
                }
            )

        add_count.click(
            fn=wrap_add_count,
            inputs=[
                edit_mode,
                requirements_list,
                count_min,
                count_max
            ],
            outputs=[
                requirements_list,
                requirements_display,
                requirement_type,
                requirement_type  # 添加requirement_type到输出以更新选项
            ]
        )
        
        add_length.click(
            fn=wrap_add_length,
            inputs=[
                edit_mode,
                requirements_list,
                length_min,
                length_max,
                length_mode,
                length_coefficient
            ],
            outputs=[
                requirements_list,
                requirements_display,
                requirement_type,
                requirement_type
            ]
        )
        
        add_format.click(
            fn=wrap_add_format,
            inputs=[
                edit_mode,
                requirements_list,
                format_type,
                format_example,
                format_mode,
                format_coefficient
            ],
            outputs=[
                requirements_list,
                requirements_display,
                requirement_type,
                requirement_type
            ]
        )
        
        # 更新按钮的可见性控制
        def update_button_visibility(req_type: Optional[str]) -> Dict:
            """更新按钮的可见性"""
            return {
                add_count: gr.update(visible=req_type == "数量限制"),
                add_length: gr.update(visible=req_type == "内容长度"),
                add_format: gr.update(visible=req_type == "内容格式")
            }
        
        requirement_type.change(
            fn=update_button_visibility,
            inputs=[requirement_type],
            outputs=[add_count, add_length, add_format]
        )
        
        requirements_display.select(
            fn=select_requirement,
            inputs=[requirements_list],
            outputs=[edit_button, delete_button, selected_row]
        )
        
        delete_button.click(
            fn=delete_requirement,
            inputs=[selected_row, requirements_list],
            outputs=[
                requirements_list,
                requirements_display,
                edit_button,
                delete_button
            ]
        )
        
        edit_button.click(
            fn=edit_requirement,
            inputs=[selected_row, requirements_list],
            outputs=[
                requirement_type,
                count_group,
                length_group,
                format_group,
                count_min,
                count_max,
                length_min,
                length_max,
                length_mode,
                length_coefficient,
                format_type,
                format_example,
                format_mode,
                format_coefficient,
                edit_button,
                delete_button,
                edit_mode
            ]
        )
        
        requirements_group.visible = False
        
        def update_visibility(rule_type: Optional[str]) -> Dict:
            """更新要求配置界面的可见性"""
            show = rule_type is not None
            return {
                requirements_group: gr.update(visible=show)
            }
    
    return {
        "group": requirements_group,
        "update_visibility": update_visibility,
        "requirements_list": requirements_list,
        "requirements_display": requirements_display,
        "requirement_type": requirement_type
    }

def create_rule_definition_tab():
    """规则定义子标签页"""
    with gr.Blocks() as tab:
        gr.Markdown("## 规则定义")
        
        # 获取所有已注册的评分器
        graders = GraderRegistry.list_graders()
        if not graders:
            gr.Markdown("⚠️ 警告：未找到任何已注册的评分器！")
            return {"grader_type": None}
        
        # 评分器选择和测试区域
        with gr.Row():
            with gr.Column(scale=1):
                grader_type = gr.Dropdown(
                    choices=list(graders.keys()),
                    label="评分器类型",
                    interactive=True,
                    value=list(graders.keys())[0] if graders else None
                )
                grader_description = gr.Markdown(
                    value=f"**评分器说明**：{graders[list(graders.keys())[0]]}" if graders else ""
                )
                
                def update_description(grader_name):
                    if grader_name in graders:
                        return f"**评分器说明**：{graders[grader_name]}"
                    return "⚠️ 未选择评分器"
                
                grader_type.change(
                    fn=update_description,
                    inputs=[grader_type],
                    outputs=[grader_description]
                )
            
            with gr.Column(scale=1):
                with gr.Group():
                    test_input = gr.Textbox(
                        label="测试输入",
                        placeholder="输入要测试的内容..."
                    )
                    test_reference = gr.Textbox(
                        label="参考答案",
                        placeholder="输入正确答案..."
                    )
                    test_button = gr.Button("测试评分", variant="primary")
                    test_result = gr.Number(label="评分结果", value=0.0)
                
                def test_grader(grader_name: str, test_input: str, test_reference: str) -> float:
                    if not grader_name:
                        gr.Warning("请先选择一个评分器！")
                        return 0.0
                    if not test_input or not test_reference:
                        gr.Warning("请输入测试内容和参考答案！")
                        return 0.0
                    
                    try:
                        grader_class = GraderRegistry.get(grader_name)
                        grader = grader_class()
                        score = grader.grade(test_input, test_reference)
                        if score == 1.0:
                            gr.Info("完全正确！")
                        elif score > 0:
                            gr.Info(f"部分正确，得分：{score}")
                        else:
                            gr.Warning("答案不正确")
                        return score
                    except Exception as e:
                        gr.Error(f"评分出错：{str(e)}")
                        return 0.0
                
                test_button.click(
                    fn=test_grader,
                    inputs=[grader_type, test_input, test_reference],
                    outputs=[test_result]
                )
        
        # 规则配置区域
        gr.Markdown("### 规则配置")
        
        with gr.Row():
            with gr.Column(scale=2):
                # 规则选择下拉菜单
                rule_type = gr.Dropdown(
                    choices=["思考过程", "结果标签", "工具标签", "自定义标签"],
                    label="新增规则",
                    value=None,
                    interactive=True
                )
            
            with gr.Column(scale=1):
                # 标签名称输入框
                label_name = gr.Textbox(
                    label="标签名称",
                    placeholder="输入标签名称...",
                    interactive=False,
                    visible=False
                )
            
            with gr.Column(scale=1):
                # 验证器选择下拉菜单
                validator_type = gr.Dropdown(
                    choices=list(VALIDATOR_TYPE_MAP.keys()),
                    label="验证器",
                    value="无",
                    interactive=True,
                    visible=False
                )
        
        # 创建标签要求配置界面
        requirements_ui = create_requirements_ui()
        
        with gr.Row():
            add_button = gr.Button("添加规则", variant="primary", visible=False)
        
        # 规则列表显示
        rules_list = gr.State([])  # 存储已添加的规则
        rules_display = gr.DataFrame(
            headers=["规则类型", "标签名称", "验证器", "标签要求"],
            label="已添加的规则",
            interactive=False,
            visible=True,
            wrap=True
        )
        with gr.Row(equal_height=True):
            rule_edit_button = gr.Button("✏️ 编辑", visible=False, size="sm", scale=1)
            rule_delete_button = gr.Button("🗑️ 删除", visible=False, size="sm", variant="stop", scale=1)
        selected_rule_row = gr.State(None)  # 存储选中的规则行索引
        rule_edit_mode = gr.State({  # 规则编辑模式状态
            "active": False,
            "index": None
        })

        def select_rule(evt: gr.SelectData, rules: List[Dict]) -> Dict:
            """选择规则进行编辑或删除"""
            row_index = evt.index[0]
            return {
                rule_edit_button: gr.update(visible=True),
                rule_delete_button: gr.update(visible=True),
                selected_rule_row: row_index
            }
        
        def delete_rule_and_update(row_index: int, rules: List[Dict]) -> Tuple[List[Dict], List[List[str]], gr.Button, gr.Button, gr.Dropdown]:
            """删除规则并更新规则类型选项"""
            if row_index is None:
                return (
                    rules,
                    [[r["type"], r["label"], json.dumps(r["requirements"], ensure_ascii=False, indent=2)] for r in rules],
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(choices=get_available_rule_types(rules, None), value=None)
                )
            
            updated_rules = rules[:row_index] + rules[row_index + 1:]
            display_data = [
                [r["type"], r["label"], json.dumps(r["requirements"], ensure_ascii=False, indent=2)]
                for r in updated_rules
            ]
            
            return (
                updated_rules,
                display_data,
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(choices=get_available_rule_types(updated_rules, None), value=None)
            )
        
        def update_ui_visibility(rule_type: str, rules: List[Dict], edit_state: Dict, current_requirements: List[Dict]) -> Dict:
            """更新界面元素的可见性"""
            if not rule_type:
                return {
                    label_name: gr.update(visible=False),
                    validator_type: gr.update(visible=False),
                    add_button: gr.update(visible=False),
                    requirements_ui["group"]: gr.update(visible=False),
                    requirements_ui["requirements_list"]: [],
                    requirements_ui["requirements_display"]: [],
                    requirements_ui["requirement_type"]: gr.update(choices=list(REQUIREMENT_TYPE_MAP.values()), value=None)
                }
            
            is_custom = rule_type == "自定义标签"
            label_value = "" if is_custom else RULE_TYPE_MAP[rule_type]
            
            # 如果是编辑模式，从当前编辑的规则中获取要求列表和验证器
            if edit_state["active"] and edit_state["index"] is not None and edit_state["index"] < len(rules):
                current_rule = rules[edit_state["index"]]
                requirements = current_rule.get("requirements", [])
                requirements_display = [[r["type"], json.dumps(r["config"], ensure_ascii=False, indent=2)] for r in requirements]
                # 获取可用的要求类型（排除已添加的类型）
                used_types = {REQUIREMENT_TYPE_MAP[r["type"]] for r in requirements}
                available_types = [t for t in REQUIREMENT_TYPE_MAP.values() if t not in used_types]
                # 获取当前验证器
                current_validator = next((k for k, v in VALIDATOR_TYPE_MAP.items() if v == current_rule.get("validator")), "无")
            else:
                requirements = []
                requirements_display = []
                available_types = list(REQUIREMENT_TYPE_MAP.values())
                current_validator = "无"
                # 重置编辑状态
                edit_state["active"] = False
                edit_state["index"] = None
            
            return {
                label_name: gr.update(
                    visible=True,
                    interactive=is_custom,
                    value=label_value,
                    label="自定义标签名称" if is_custom else "标签名称"
                ),
                validator_type: gr.update(
                    visible=True,
                    value=current_validator
                ),
                add_button: gr.update(visible=True),
                requirements_ui["group"]: gr.update(visible=True),
                requirements_ui["requirements_list"]: requirements,
                requirements_ui["requirements_display"]: requirements_display,
                requirements_ui["requirement_type"]: gr.update(choices=available_types, value=None)
            }

        def edit_rule(row_index: int, rules: List[Dict]) -> Dict:
            """编辑选中的规则"""
            if row_index is None:
                return {}
            
            rule = rules[row_index]
            
            # 获取可用的规则类型，包括当前正在编辑的规则类型
            available_types = get_available_rule_types(rules, rule)
            type_map = {
                "think": "思考过程",
                "answer": "结果标签",
                "tool_call": "工具标签"
            }
            current_type = type_map.get(rule["label"], "自定义标签")
            if current_type not in available_types:
                available_types.append(current_type)
            
            # 获取当前规则的要求列表
            current_requirements = rule.get("requirements", [])
            requirements_display = [[r["type"], json.dumps(r["config"], ensure_ascii=False, indent=2)] for r in current_requirements]
            
            # 获取可用的要求类型（排除已添加的类型）
            used_types = {REQUIREMENT_TYPE_MAP[r["type"]] for r in current_requirements}
            available_requirement_types = [t for t in REQUIREMENT_TYPE_MAP.values() if t not in used_types]
            
            # 获取当前验证器
            current_validator = next((k for k, v in VALIDATOR_TYPE_MAP.items() if v == rule.get("validator")), "无")
            
            # 更新界面状态
            updates = {
                rule_type: gr.update(
                    choices=available_types,
                    value=current_type
                ),
                label_name: gr.update(
                    visible=True,
                    interactive=rule["type"] == "自定义标签",
                    value=rule["label"]
                ),
                validator_type: gr.update(
                    visible=True,
                    value=current_validator
                ),
                add_button: gr.update(visible=True),
                rule_edit_button: gr.update(visible=False),
                rule_delete_button: gr.update(visible=False),
                requirements_ui["requirements_list"]: current_requirements,
                requirements_ui["requirements_display"]: requirements_display,
                requirements_ui["requirement_type"]: gr.update(
                    choices=available_requirement_types,
                    value=None
                ),
                requirements_ui["group"]: gr.update(visible=True),
                rule_edit_mode: {"active": True, "index": row_index}
            }
            
            return updates

        # 规则类型改变时更新界面
        rule_type.change(
            fn=update_ui_visibility,
            inputs=[rule_type, rules_list, rule_edit_mode, requirements_ui["requirements_list"]],
            outputs=[
                label_name,
                validator_type,
                add_button,
                requirements_ui["group"],
                requirements_ui["requirements_list"],
                requirements_ui["requirements_display"],
                requirements_ui["requirement_type"]
            ]
        )

        # 绑定规则选择和编辑删除事件
        rules_display.select(
            fn=select_rule,
            inputs=[rules_list],
            outputs=[rule_edit_button, rule_delete_button, selected_rule_row]
        )
        
        rule_delete_button.click(
            fn=delete_rule_and_update,
            inputs=[selected_rule_row, rules_list],
            outputs=[rules_list, rules_display, rule_edit_button, rule_delete_button, rule_type]
        )
        
        rule_edit_button.click(
            fn=edit_rule,
            inputs=[selected_rule_row, rules_list],
            outputs=[
                rule_type,
                label_name,
                add_button,
                rule_edit_button,
                rule_delete_button,
                requirements_ui["requirements_list"],
                requirements_ui["requirements_display"],
                requirements_ui["requirement_type"],
                requirements_ui["group"],
                rule_edit_mode
            ]
        )

        # 修改原有的添加规则函数，支持编辑模式
        def add_or_update_rule(rule_type: str, label_value: str, validator_value: str, requirements: List[Dict], rules: List[Dict], edit_state: Dict) -> tuple:
            """添加或更新规则"""
            if not rule_type:
                return (
                    rules,
                    [[r["type"], r["label"], r.get("validator", "无"), json.dumps(r["requirements"], ensure_ascii=False, indent=2)] for r in rules],
                    None,
                    "",
                    "无",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(choices=get_available_rule_types(rules), value=None)
                )
            
            # 验证标签名称
            if rule_type == "自定义标签":
                if not label_value:
                    gr.Warning("请输入自定义标签名称")
                    return (
                        rules,
                        [[r["type"], r["label"], r.get("validator", "无"), json.dumps(r["requirements"], ensure_ascii=False, indent=2)] for r in rules],
                        rule_type,
                        label_value,
                        validator_value,
                        gr.update(visible=True),
                        gr.update(visible=True),
                        gr.update(visible=True),
                        gr.update(choices=get_available_rule_types(rules), value=rule_type)
                    )
                if not label_value.isidentifier():
                    gr.Warning("标签名称只能包含字母、数字和下划线，且不能以数字开头")
                    return (
                        rules,
                        [[r["type"], r["label"], r.get("validator", "无"), json.dumps(r["requirements"], ensure_ascii=False, indent=2)] for r in rules],
                        rule_type,
                        label_value,
                        validator_value,
                        gr.update(visible=True),
                        gr.update(visible=True),
                        gr.update(visible=True),
                        gr.update(choices=get_available_rule_types(rules), value=rule_type)
                    )
            
            # 如果是编辑模式，使用原有的要求列表
            if edit_state["active"] and edit_state["index"] is not None:
                current_requirements = rules[edit_state["index"]].get("requirements", [])
            else:
                current_requirements = requirements
            
            # 创建新规则
            new_rule = {
                "type": rule_type,
                "label": label_value,
                "validator": VALIDATOR_TYPE_MAP[validator_value],
                "requirements": current_requirements
            }
            
            # 如果是编辑模式，替换原有规则
            if edit_state["active"] and edit_state["index"] is not None:
                # 检查是否有重复的标签名（排除当前编辑的规则）
                other_rules = rules[:edit_state["index"]] + rules[edit_state["index"] + 1:]
                if any(r["label"] == label_value for r in other_rules):
                    gr.Warning(f"标签名称 '{label_value}' 已经存在")
                    return (
                        rules,
                        [[r["type"], r["label"], r.get("validator", "无"), json.dumps(r["requirements"], ensure_ascii=False, indent=2)] for r in rules],
                        rule_type,
                        label_value,
                        validator_value,
                        gr.update(visible=True),
                        gr.update(visible=True),
                        gr.update(visible=True),
                        gr.update(choices=get_available_rule_types(rules), value=rule_type)
                    )
                
                # 更新规则
                updated_rules = rules.copy()
                updated_rules[edit_state["index"]] = new_rule
                # 重置编辑状态
                edit_state["active"] = False
                edit_state["index"] = None
            else:
                # 检查是否已存在相同的标签名
                if any(r["label"] == label_value for r in rules):
                    gr.Warning(f"标签名称 '{label_value}' 已经存在")
                    return (
                        rules,
                        [[r["type"], r["label"], r.get("validator", "无"), json.dumps(r["requirements"], ensure_ascii=False, indent=2)] for r in rules],
                        rule_type,
                        label_value,
                        validator_value,
                        gr.update(visible=True),
                        gr.update(visible=True),
                        gr.update(visible=True),
                        gr.update(choices=get_available_rule_types(rules), value=rule_type)
                    )
                updated_rules = rules + [new_rule]
            
            # 转换为DataFrame显示格式
            display_data = [
                [r["type"], r["label"], r.get("validator", "无"), json.dumps(r["requirements"], ensure_ascii=False, indent=2)]
                for r in updated_rules
            ]
            
            # 清空输入并隐藏控件，更新规则类型选项
            return (
                updated_rules,
                display_data,
                None,  # 清空规则类型
                "",    # 清空标签名称
                "无",  # 重置验证器选择
                gr.update(visible=False),  # 隐藏标签名称输入框
                gr.update(visible=False),  # 隐藏添加按钮
                gr.update(visible=False),  # 隐藏验证器选择
                gr.update(choices=get_available_rule_types(updated_rules), value=None)  # 更新规则类型选项
            )

        # 更新添加规则按钮的事件处理
        add_button.click(
            fn=add_or_update_rule,
            inputs=[
                rule_type,
                label_name,
                validator_type,
                requirements_ui["requirements_list"],
                rules_list,
                rule_edit_mode
            ],
            outputs=[
                rules_list,
                rules_display,
                rule_type,
                label_name,
                validator_type,
                add_button,
                rule_type
            ]
        )
        
        return {
            "grader_type": grader_type,
            "rules": rules_list
        }


def create_model_evaluation_tab():
    """模型评判子标签页"""
    with gr.Blocks() as tab:
        gr.Markdown("## 模型评判")
        # 待补充具体内容
        return {}


def create_validation_tools_tab():
    """验证工具子标签页"""
    with gr.Blocks() as tab:
        gr.Markdown("## 验证工具")
        # 待补充具体内容
        return {}


def generate_reward_json(rule_data: Dict[str, Any], model_data: Dict[str, Any], validation_data: Dict[str, Any]) -> Dict[str, Any]:
    """生成奖赏配置JSON"""
    reward_config = {
        "grader": {
            "type": rule_data["grader_type"]
        }
    }
    return reward_config

def generate_reward_python(reward_config: Dict[str, Any]) -> str:
    """根据配置生成Python奖赏函数"""
    template = f'''
import numpy as np
from components.rewards.graders import GraderRegistry

class RewardFunction:
    def __init__(self):
        # 初始化评分器
        grader_class = GraderRegistry.get("{reward_config["grader"]["type"]}")
        self.grader = grader_class()
        
    def calculate_reward(self, state, action, next_state, info=None):
        """计算奖赏值"""
        if not info or 'reference' not in info:
            return 0.0
        
        return self.grader.grade(next_state, info['reference'])
        
    def reset(self):
        """重置奖赏函数状态"""
        pass
'''
    return template

def add_rule(
    rule_type: str,
    label_value: str,
    requirements: List[Dict],
    rules: List[Dict]
) -> tuple:
    """添加新规则"""
    if not rule_type:
        return rules, None
    
    # 验证标签名称
    if rule_type == "自定义标签":
        if not label_value:
            gr.Warning("请输入自定义标签名称")
            return rules, [[r["type"], r["label"], json.dumps(r["requirements"], ensure_ascii=False, indent=2)] for r in rules]
        if not label_value.isidentifier():
            gr.Warning("标签名称只能包含字母、数字和下划线，且不能以数字开头")
            return rules, [[r["type"], r["label"], json.dumps(r["requirements"], ensure_ascii=False, indent=2)] for r in rules]
    
    # 创建新规则
    new_rule = {
        "type": rule_type,
        "label": label_value,
        "requirements": requirements
    }
    
    # 更新规则列表
    updated_rules = rules + [new_rule]
    
    # 转换为DataFrame显示格式
    display_data = [
        [r["type"], r["label"], json.dumps(r["requirements"], ensure_ascii=False, indent=2)]
        for r in updated_rules
    ]
    
    # 清空输入并隐藏控件
    return (
        updated_rules,
        display_data,
        gr.update(value=None),  # 清空规则类型
        gr.update(value=""),    # 清空标签名称
        gr.update(visible=False),  # 隐藏标签名称输入框
        gr.update(visible=False)   # 隐藏添加按钮
    )
