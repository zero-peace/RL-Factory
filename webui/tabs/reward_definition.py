import gradio as gr
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from components.rewards.graders import GraderRegistry

# 类型映射
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

# 规则配置相关的常量
RESPONSE_POSITIONS = ["首位", "末尾", "整体", "每个"]
LABEL_TYPES = ["思考过程", "结果标签", "工具标签", "自定义"]
RULE_FORMS = ["数量", "长度", "格式", "得分"]

def create_rule_definition_tab():
    """创建规则定义标签页"""
    with gr.Column() as tab:
        # 规则列表状态
        rules_state = gr.State([])
        
        # 添加规则按钮行
        with gr.Row():
            gr.Markdown("### 规则列表")
            add_rule_btn = gr.Button("➕ 添加规则", scale=0)
        
        # 规则编辑区域
        with gr.Group(visible=False) as rule_edit_group:
            with gr.Row():
                rule_description = gr.Textbox(
                    label="规则描述",
                    placeholder="请输入规则描述（不超过20字）",
                    max_lines=1,
                    scale=8
                )
                with gr.Column(scale=2):
                    with gr.Row():
                        save_rule_btn = gr.Button("保存", variant="primary", size="sm")
                        cancel_rule_btn = gr.Button("取消", size="sm")
            
            with gr.Row():
                response_position = gr.Dropdown(
                    choices=RESPONSE_POSITIONS,
                    label="回复选择",
                    value=RESPONSE_POSITIONS[0],
                    scale=2
                )
                
                with gr.Column(scale=3):
                    with gr.Row():
                        label_type = gr.Dropdown(
                            choices=LABEL_TYPES,
                            label="标签类型",
                            value=LABEL_TYPES[0],
                            scale=1
                        )
                        custom_label = gr.Textbox(
                            label="自定义标签",
                            placeholder="请输入自定义标签",
                            visible=False,
                            scale=1
                        )
                
                rule_form = gr.Dropdown(
                    choices=RULE_FORMS,
                    label="规则形式",
                    value=RULE_FORMS[0],
                    scale=2
                )
        
        # 规则列表显示
        rules_list = gr.Dataframe(
            headers=["描述", "回复选择", "标签", "规则形式"],
            datatype=["str", "str", "str", "str"],
            col_count=(4, "fixed"),
            wrap=True,
            row_count=10,
            interactive=False,
            visible=True
        )
        
        # 操作按钮区域
        with gr.Row():
            edit_rule_btn = gr.Button("✏️ 编辑选中规则", visible=False, variant="secondary")
            delete_rule_btn = gr.Button("🗑️ 删除选中规则", visible=False, variant="stop")
        
        # 选中的规则索引
        selected_rule_index = gr.State(None)
        
        # 编辑状态
        edit_state = gr.State({
            "active": False,
            "index": None
        })
        
        def toggle_rule_edit(show: bool, edit_state: Dict = None) -> Tuple[Dict, str, str, str, str, str]:
            """切换规则编辑区域的显示状态"""
            if not show and edit_state:
                edit_state["active"] = False
                edit_state["index"] = None
            
            return (
                gr.update(visible=show),  # rule_edit_group
                "",  # rule_description
                RESPONSE_POSITIONS[0],  # response_position
                LABEL_TYPES[0],  # label_type
                "",  # custom_label
                RULE_FORMS[0]  # rule_form
            )
        
        def update_label_input(label_type: str) -> Dict:
            """更新标签输入区域"""
            return {
                custom_label: gr.update(visible=label_type == "自定义")
            }
        
        def save_rule(description: str, position: str, label_type: str, 
                     custom_label: str, rule_form: str, rules: List[Dict],
                     edit_state: Dict) -> Tuple[List[Dict], List[List], Dict, str, str, str, str, str]:
            """保存规则"""
            if not description or len(description) > 20:
                gr.Warning("请输入有效的规则描述（不超过20字）")
                return (
                    rules,  # rules_state
                    [[r["description"], r["position"], r["label"], r["form"]] for r in rules],  # rules_list
                    gr.update(visible=True),  # rule_edit_group
                    description,  # rule_description
                    position,  # response_position
                    label_type,  # label_type
                    custom_label,  # custom_label
                    rule_form  # rule_form
                )
            
            # 获取实际的标签值
            label = custom_label if label_type == "自定义" else label_type
            
            new_rule = {
                "description": description,
                "position": position,
                "label": label,
                "form": rule_form
            }
            
            # 编辑模式
            if edit_state["active"] and edit_state["index"] is not None:
                rules[edit_state["index"]] = new_rule
                edit_state["active"] = False
                edit_state["index"] = None
            else:
                rules.append(new_rule)
            
            # 更新显示数据
            display_data = [
                [r["description"], r["position"], r["label"], r["form"]]
                for r in rules
            ]
            
            # 清空编辑区域并返回默认值
            return (
                rules,  # rules_state
                display_data,  # rules_list
                gr.update(visible=False),  # rule_edit_group
                "",  # rule_description
                RESPONSE_POSITIONS[0],  # response_position
                LABEL_TYPES[0],  # label_type
                "",  # custom_label
                RULE_FORMS[0]  # rule_form
            )
        
        def select_rule(evt: gr.SelectData, rules: List[Dict]) -> Tuple[int, Dict, Dict]:
            """选择规则"""
            row_index = evt.index[0]
            return (
                row_index,  # selected_rule_index
                gr.update(visible=True),  # edit_rule_btn
                gr.update(visible=True)  # delete_rule_btn
            )
        
        def edit_selected_rule(rule_index: int, rules: List[Dict]) -> Tuple[Dict, str, str, str, str, str, Dict]:
            """编辑选中的规则"""
            if rule_index is None or rule_index >= len(rules):
                return (
                    gr.update(visible=False),  # rule_edit_group
                    "",  # rule_description
                    RESPONSE_POSITIONS[0],  # response_position
                    LABEL_TYPES[0],  # label_type
                    "",  # custom_label
                    RULE_FORMS[0],  # rule_form
                    {"active": False, "index": None}  # edit_state
                )
            
            rule = rules[rule_index]
            return (
                gr.update(visible=True),  # rule_edit_group
                rule["description"],  # rule_description
                rule["position"],  # response_position
                "自定义" if rule["label"] not in LABEL_TYPES else rule["label"],  # label_type
                rule["label"] if rule["label"] not in LABEL_TYPES else "",  # custom_label
                rule["form"],  # rule_form
                {"active": True, "index": rule_index}  # edit_state
            )
        
        def delete_selected_rule(rule_index: int, rules: List[Dict]) -> Tuple[List[Dict], List[List], Dict, Dict, int]:
            """删除选中的规则"""
            if rule_index is None or rule_index >= len(rules):
                return (
                    rules,  # rules_state
                    [[r["description"], r["position"], r["label"], r["form"]] for r in rules],  # rules_list
                    gr.update(visible=False),  # edit_rule_btn
                    gr.update(visible=False),  # delete_rule_btn
                    None  # selected_rule_index
                )
            
            updated_rules = rules[:rule_index] + rules[rule_index + 1:]
            display_data = [
                [r["description"], r["position"], r["label"], r["form"]]
                for r in updated_rules
            ]
            
            return (
                updated_rules,  # rules_state
                display_data,  # rules_list
                gr.update(visible=False),  # edit_rule_btn
                gr.update(visible=False),  # delete_rule_btn
                None  # selected_rule_index
            )
        
        # 绑定事件
        add_rule_btn.click(
            fn=toggle_rule_edit,
            inputs=[gr.State(True)],
            outputs=[
                rule_edit_group,
                rule_description,
                response_position,
                label_type,
                custom_label,
                rule_form
            ]
        )
        
        cancel_rule_btn.click(
            fn=toggle_rule_edit,
            inputs=[gr.State(False), edit_state],
            outputs=[
                rule_edit_group,
                rule_description,
                response_position,
                label_type,
                custom_label,
                rule_form
            ]
        )
        
        label_type.change(
            fn=update_label_input,
            inputs=[label_type],
            outputs=[custom_label]
        )
        
        save_rule_btn.click(
            fn=save_rule,
            inputs=[
                rule_description,
                response_position,
                label_type,
                custom_label,
                rule_form,
                rules_state,
                edit_state
            ],
            outputs=[
                rules_state,
                rules_list,
                rule_edit_group,
                rule_description,
                response_position,
                label_type,
                custom_label,
                rule_form
            ]
        )
        
        rules_list.select(
            fn=select_rule,
            inputs=[rules_state],
            outputs=[
                selected_rule_index,
                edit_rule_btn,
                delete_rule_btn
            ]
        )
        
        edit_rule_btn.click(
            fn=edit_selected_rule,
            inputs=[selected_rule_index, rules_state],
            outputs=[
                rule_edit_group,
                rule_description,
                response_position,
                label_type,
                custom_label,
                rule_form,
                edit_state
            ]
        )
        
        delete_rule_btn.click(
            fn=delete_selected_rule,
            inputs=[selected_rule_index, rules_state],
            outputs=[
                rules_state,
                rules_list,
                edit_rule_btn,
                delete_rule_btn,
                selected_rule_index
            ]
        )
    
    return tab

def create_reward_definition_tab():
    """奖赏定义主标签页"""
    with gr.Blocks() as tab:
        gr.Markdown("# 奖赏定义")
        
        # 创建子标签页
        with gr.Tabs() as subtabs:
            with gr.TabItem("规则定义"):
                rule_tab = create_rule_definition_tab()
            
            with gr.TabItem("模型评判"):
                # TODO: 实现模型评判界面
                pass
            
            with gr.TabItem("验证工具"):
                # TODO: 实现验证工具界面
                pass
    
    return tab
