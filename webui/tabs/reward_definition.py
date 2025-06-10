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
LABEL_TYPES = ["<think>", "<answer>", "<tool_call>", "自定义"]
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
            with gr.Row(equal_height=True):
                rule_description = gr.Textbox(
                    label="规则描述",
                    placeholder="请输入规则描述（不超过20字）",
                    max_lines=1,
                    scale=8
                )
                save_rule_btn = gr.Button("保存", variant="primary", scale=1)
                cancel_rule_btn = gr.Button("取消", scale=1)
            
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
            
            # 规则形式配置区域
            with gr.Group() as rule_form_config:
                # 数量限制配置
                with gr.Group(visible=False) as count_config:
                    with gr.Row():
                        count_min = gr.Number(
                            label="最小数量",
                            value=1,
                            minimum=0,
                            scale=1
                        )
                        count_max = gr.Number(
                            label="最大数量",
                            value=1,
                            minimum=1,
                            scale=1
                        )
                
                # 长度限制配置
                with gr.Group(visible=False) as length_config:
                    with gr.Row():
                        length_min = gr.Number(
                            label="最小长度",
                            value=None,
                            minimum=0,
                            scale=1
                        )
                        length_max = gr.Number(
                            label="最大长度",
                            value=512,
                            minimum=1,
                            scale=1
                        )
                
                # 格式限制配置
                with gr.Group(visible=False) as format_config:
                    with gr.Row():
                        format_type = gr.Radio(
                            choices=["json", "xml"],
                            label="格式类型",
                            value="json"
                        )
                    format_example = gr.Code(
                        label="格式示例",
                        language="json"
                    )
                
                # 得分配置
                with gr.Group(visible=False) as score_config:
                    with gr.Row():
                        grader_type = gr.Dropdown(
                            choices=list(GraderRegistry.list_graders().keys()),
                            label="评分器",
                            value=None
                        )
                        answer_field = gr.Textbox(
                            label="答案字段",
                            placeholder="answer",
                            visible=False
                        )
        
        # 规则列表显示
        rules_list = gr.Dataframe(
            headers=["描述", "回复选择", "标签", "规则形式", "规则内容"],
            datatype=["str", "str", "str", "str", "str"],
            col_count=(5, "fixed"),
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
        
        def update_rule_form_config(form: str) -> Dict:
            """更新规则形式配置区域"""
            updates = {
                count_config: gr.update(visible=form == "数量"),
                length_config: gr.update(visible=form == "长度"),
                format_config: gr.update(visible=form == "格式"),
                score_config: gr.update(visible=form == "得分")
            }
            return updates
        
        def update_grader_config(grader_name: str) -> Dict:
            """更新评分器配置"""
            if not grader_name:
                return {answer_field: gr.update(visible=False)}
            
            grader_class = GraderRegistry.get(grader_name)
            instance = grader_class()
            return {
                answer_field: gr.update(
                    visible=instance.gt_required,
                    value="answer" if instance.gt_required else None
                )
            }
        
        def validate_format_example(example: str, format_type: str) -> Tuple[bool, str]:
            """验证格式示例
            
            Returns:
                Tuple[bool, str]: (是否有效, 错误信息)
            """
            if not example.strip():
                return False, "格式示例不能为空"
            
            try:
                if format_type == "json":
                    json.loads(example)
                    return True, ""
                elif format_type == "xml":
                    # TODO: 添加XML验证
                    return True, ""
                else:
                    return False, f"不支持的格式类型：{format_type}"
            except json.JSONDecodeError as e:
                line_col = f"第 {e.lineno} 行，第 {e.colno} 列" if hasattr(e, 'lineno') and hasattr(e, 'colno') else ""
                return False, f"JSON格式错误 {line_col}：{str(e)}"
            except Exception as e:
                return False, f"格式验证失败：{str(e)}"
        
        def validate_and_show_format_example(example: str, format_type: str) -> str:
            """验证格式示例并显示错误信息
            
            Returns:
                str: 原始示例文本
            """
            is_valid, msg = validate_format_example(example, format_type)
            if not is_valid:
                gr.Warning(msg)
            return example
        
        def toggle_rule_edit(show: bool, edit_state: Dict = None) -> Tuple[Dict, str, str, str, str, str, Dict, Dict, Dict, Dict]:
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
                RULE_FORMS[0],  # rule_form
                gr.update(visible=show and RULE_FORMS[0] == "数量"),  # count_config
                gr.update(visible=show and RULE_FORMS[0] == "长度"),  # length_config
                gr.update(visible=show and RULE_FORMS[0] == "格式"),  # format_config
                gr.update(visible=show and RULE_FORMS[0] == "得分")  # score_config
            )
        
        def update_label_input(label_type: str) -> Dict:
            """更新标签输入区域"""
            return {
                custom_label: gr.update(visible=label_type == "自定义")
            }
        
        def format_rule_config(rule: Dict) -> str:
            """格式化规则配置为显示文本"""
            form = rule["form"]
            config = rule.get("config", {})
            
            if form == "数量":
                return f"最小值: {config.get('min', 1)}, 最大值: {config.get('max', 1)}"
            elif form == "长度":
                min_len = config.get('min')
                max_len = config.get('max', 512)
                if min_len is None:
                    return f"最大长度: {max_len}"
                return f"最小长度: {min_len}, 最大长度: {max_len}"
            elif form == "格式":
                format_type = config.get('type', 'json')
                example = config.get('example', '')
                # 如果示例太长，只显示前50个字符
                if len(example) > 50:
                    example = example[:47] + "..."
                return f"类型: {format_type}, 示例: {example}"
            else:  # 得分
                grader = config.get('grader', '未指定')
                answer_field = config.get('answer_field')
                if answer_field:
                    return f"评分器: {grader}, 答案字段: {answer_field}"
                return f"评分器: {grader}"
        
        def save_rule(
            description: str, position: str, label_type: str, custom_label: str, rule_form: str,
            count_min: int, count_max: int,
            length_min: Optional[int], length_max: int,
            format_type: str, format_example: str,
            grader_type: Optional[str], answer_field: Optional[str],
            rules: List[Dict], edit_state: Dict
        ) -> Tuple[List[Dict], List[List], Dict, str, str, str, str, str]:
            """保存规则"""
            if not description or len(description) > 20:
                gr.Warning("请输入有效的规则描述（不超过20字）")
                return (
                    rules,  # rules_state
                    [[r["description"], r["position"], r["label"], r["form"], format_rule_config(r)] for r in rules],  # rules_list
                    gr.update(visible=True),  # rule_edit_group
                    description,  # rule_description
                    position,  # response_position
                    label_type,  # label_type
                    custom_label,  # custom_label
                    rule_form  # rule_form
                )
            
            # 获取实际的标签值
            label = custom_label if label_type == "自定义" else label_type
            
            # 验证规则形式相关的配置
            config = {}
            if rule_form == "数量":
                if count_max < count_min:
                    gr.Warning("最大数量不能小于最小数量")
                    return (
                        rules, [[r["description"], r["position"], r["label"], r["form"], format_rule_config(r)] for r in rules],
                        gr.update(visible=True), description, position, label_type, custom_label, rule_form
                    )
                config = {"min": count_min, "max": count_max}
            elif rule_form == "长度":
                if length_max < (length_min or 0):
                    gr.Warning("最大长度不能小于最小长度")
                    return (
                        rules, [[r["description"], r["position"], r["label"], r["form"], format_rule_config(r)] for r in rules],
                        gr.update(visible=True), description, position, label_type, custom_label, rule_form
                    )
                config = {"min": length_min, "max": length_max}
            elif rule_form == "格式":
                is_valid, error_msg = validate_format_example(format_example, format_type)
                if not is_valid:
                    gr.Warning(f"格式示例无效：{error_msg}")
                    return (
                        rules, [[r["description"], r["position"], r["label"], r["form"], format_rule_config(r)] for r in rules],
                        gr.update(visible=True), description, position, label_type, custom_label, rule_form
                    )
                config = {"type": format_type, "example": format_example}
            else:  # 得分
                if not grader_type:
                    gr.Warning("请选择评分器")
                    return (
                        rules, [[r["description"], r["position"], r["label"], r["form"], format_rule_config(r)] for r in rules],
                        gr.update(visible=True), description, position, label_type, custom_label, rule_form
                    )
                grader_class = GraderRegistry.get(grader_type)
                instance = grader_class()
                if instance.gt_required and not answer_field:
                    gr.Warning("请填写答案字段")
                    return (
                        rules, [[r["description"], r["position"], r["label"], r["form"], format_rule_config(r)] for r in rules],
                        gr.update(visible=True), description, position, label_type, custom_label, rule_form
                    )
                config = {
                    "grader": grader_type,
                    "answer_field": answer_field if instance.gt_required else None
                }
            
            new_rule = {
                "description": description,
                "position": position,
                "label": label,
                "form": rule_form,
                "config": config
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
                [r["description"], r["position"], r["label"], r["form"], format_rule_config(r)]
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
        
        def edit_selected_rule(rule_index: int, rules: List[Dict]) -> Tuple[Dict, str, str, str, str, str, Dict, int, int, int, int, str, str, str, str]:
            """编辑选中的规则"""
            if rule_index is None or rule_index >= len(rules):
                return (
                    gr.update(visible=False),  # rule_edit_group
                    "",  # rule_description
                    RESPONSE_POSITIONS[0],  # response_position
                    LABEL_TYPES[0],  # label_type
                    "",  # custom_label
                    RULE_FORMS[0],  # rule_form
                    {"active": False, "index": None},  # edit_state
                    1,  # count_min
                    1,  # count_max
                    None,  # length_min
                    512,  # length_max
                    "json",  # format_type
                    "",  # format_example
                    None,  # grader_type
                    ""  # answer_field
                )
            
            rule = rules[rule_index]
            config = rule.get("config", {})
            
            # 根据规则形式准备配置值
            if rule["form"] == "数量":
                count_min = config.get("min", 1)
                count_max = config.get("max", 1)
                length_min = None
                length_max = 512
                format_type = "json"
                format_example = ""
                grader_type = None
                answer_field = ""
            elif rule["form"] == "长度":
                count_min = 1
                count_max = 1
                length_min = config.get("min")
                length_max = config.get("max", 512)
                format_type = "json"
                format_example = ""
                grader_type = None
                answer_field = ""
            elif rule["form"] == "格式":
                count_min = 1
                count_max = 1
                length_min = None
                length_max = 512
                format_type = config.get("type", "json")
                format_example = config.get("example", "")
                grader_type = None
                answer_field = ""
            else:  # 得分
                count_min = 1
                count_max = 1
                length_min = None
                length_max = 512
                format_type = "json"
                format_example = ""
                grader_type = config.get("grader")
                answer_field = config.get("answer_field", "")
            
            return (
                gr.update(visible=True),  # rule_edit_group
                rule["description"],  # rule_description
                rule["position"],  # response_position
                "自定义" if rule["label"] not in LABEL_TYPES else rule["label"],  # label_type
                rule["label"] if rule["label"] not in LABEL_TYPES else "",  # custom_label
                rule["form"],  # rule_form
                {"active": True, "index": rule_index},  # edit_state
                count_min,  # count_min
                count_max,  # count_max
                length_min,  # length_min
                length_max,  # length_max
                format_type,  # format_type
                format_example,  # format_example
                grader_type,  # grader_type
                answer_field  # answer_field
            )
        
        def delete_selected_rule(rule_index: int, rules: List[Dict]) -> Tuple[List[Dict], List[List], Dict, Dict, int]:
            """删除选中的规则"""
            if rule_index is None or rule_index >= len(rules):
                return (
                    rules,  # rules_state
                    [[r["description"], r["position"], r["label"], r["form"], format_rule_config(r)] for r in rules],  # rules_list
                    gr.update(visible=False),  # edit_rule_btn
                    gr.update(visible=False),  # delete_rule_btn
                    None  # selected_rule_index
                )
            
            updated_rules = rules[:rule_index] + rules[rule_index + 1:]
            display_data = [
                [r["description"], r["position"], r["label"], r["form"], format_rule_config(r)]
                for r in updated_rules
            ]
            
            return (
                updated_rules,  # rules_state
                display_data,  # rules_list
                gr.update(visible=False),  # edit_rule_btn
                gr.update(visible=False),  # delete_rule_btn
                None  # selected_rule_index
            )
        
        def show_format_warning(result: Tuple[bool, str]) -> None:
            """显示格式验证警告"""
            is_valid, msg = result
            if not is_valid:
                gr.Warning(msg)
        
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
                rule_form,
                count_config,
                length_config,
                format_config,
                score_config
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
                rule_form,
                count_config,
                length_config,
                format_config,
                score_config
            ]
        )
        
        label_type.change(
            fn=update_label_input,
            inputs=[label_type],
            outputs=[custom_label]
        )
        
        rule_form.change(
            fn=update_rule_form_config,
            inputs=[rule_form],
            outputs=[count_config, length_config, format_config, score_config]
        )
        
        grader_type.change(
            fn=update_grader_config,
            inputs=[grader_type],
            outputs=[answer_field]
        )
        
        format_example.blur(
            fn=validate_and_show_format_example,
            inputs=[format_example, format_type],
            outputs=[format_example]
        )
        
        save_rule_btn.click(
            fn=save_rule,
            inputs=[
                rule_description,
                response_position,
                label_type,
                custom_label,
                rule_form,
                count_min,
                count_max,
                length_min,
                length_max,
                format_type,
                format_example,
                grader_type,
                answer_field,
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
                edit_state,
                count_min,
                count_max,
                length_min,
                length_max,
                format_type,
                format_example,
                grader_type,
                answer_field
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

def create_model_evaluation_tab():
    """创建模型评判标签页"""
    with gr.Column() as tab:
        # System Prompt 设置
        with gr.Group():
            gr.Markdown("### System Prompt")
            system_prompt = gr.TextArea(
                label="系统提示词",
                placeholder="请输入系统提示词，用于指导模型如何进行评判...",
                lines=5
            )
        
        # 字段设置
        with gr.Group():
            gr.Markdown("### 评判字段")
            with gr.Row():
                field_name = gr.Textbox(
                    label="字段名称",
                    placeholder="例如：query",
                    scale=2
                )
                add_field_btn = gr.Button("➕ 添加字段", scale=1)
            
            # 字段列表
            fields_list = gr.Dataframe(
                headers=["字段名称", "示例值"],
                datatype=["str", "str"],
                col_count=(2, "fixed"),
                row_count=5,
                interactive=True,
                visible=True
            )
            
            delete_field_btn = gr.Button("🗑️ 删除选中字段", visible=False)
        
        # User Prompt 设置
        with gr.Group():
            gr.Markdown("### User Prompt")
            gr.Markdown("""
            在文本中使用 `{字段名}` 来引用字段值，例如：
            ```
            请评估这个查询：{query}
            ```
            """)
            user_prompt = gr.TextArea(
                label="用户提示词",
                placeholder="请输入用户提示词，可以使用 {字段名} 来引用字段值...",
                lines=5
            )
            
            # 字段预览
            gr.Markdown("### 预览")
            preview = gr.Markdown("_在此显示预览结果_")
        
        # 字段状态
        fields_state = gr.State([])
        selected_field_index = gr.State(None)
        
        def add_field(name: str, fields: List[Dict]) -> Tuple[List[Dict], List[List], str, Dict]:
            """添加字段"""
            if not name:
                gr.Warning("请输入字段名称")
                return fields, [[f["name"], f["example"]] for f in fields], name, gr.update(visible=False)
            
            if any(f["name"] == name for f in fields):
                gr.Warning("字段名称已存在")
                return fields, [[f["name"], f["example"]] for f in fields], name, gr.update(visible=False)
            
            fields.append({"name": name, "example": ""})
            return (
                fields,  # fields_state
                [[f["name"], f["example"]] for f in fields],  # fields_list
                "",  # field_name
                gr.update(visible=False)  # delete_field_btn
            )
        
        def select_field(evt: gr.SelectData, fields: List[Dict]) -> Tuple[int, Dict]:
            """选择字段"""
            return evt.index[0], gr.update(visible=True)
        
        def delete_field(index: int, fields: List[Dict]) -> Tuple[List[Dict], List[List], Dict, int]:
            """删除字段"""
            if index is None or index >= len(fields):
                return fields, [[f["name"], f["example"]] for f in fields], gr.update(visible=False), None
            
            updated_fields = fields[:index] + fields[index + 1:]
            return (
                updated_fields,  # fields_state
                [[f["name"], f["example"]] for f in updated_fields],  # fields_list
                gr.update(visible=False),  # delete_field_btn
                None  # selected_field_index
            )
        
        def update_field_example(fields_data: List[List], fields: List[Dict]) -> List[Dict]:
            """更新字段示例值"""
            updated_fields = []
            for i, (name, example) in enumerate(fields_data):
                if i < len(fields):
                    field = fields[i].copy()
                    field["example"] = example
                    updated_fields.append(field)
            return updated_fields
        
        def update_preview(prompt: str, fields: List[Dict]) -> str:
            """更新预览"""
            if not prompt:
                return "_请输入用户提示词_"
            
            preview_text = prompt
            for field in fields:
                if field["example"]:
                    preview_text = preview_text.replace(f"{{{field['name']}}}", field["example"])
                else:
                    preview_text = preview_text.replace(f"{{{field['name']}}}", f"<{field['name']}>")
            
            return f"```\n{preview_text}\n```"
        
        # 绑定事件
        add_field_btn.click(
            fn=add_field,
            inputs=[field_name, fields_state],
            outputs=[fields_state, fields_list, field_name, delete_field_btn]
        )
        
        fields_list.select(
            fn=select_field,
            inputs=[fields_state],
            outputs=[selected_field_index, delete_field_btn]
        )
        
        delete_field_btn.click(
            fn=delete_field,
            inputs=[selected_field_index, fields_state],
            outputs=[fields_state, fields_list, delete_field_btn, selected_field_index]
        )
        
        fields_list.change(
            fn=update_field_example,
            inputs=[fields_list, fields_state],
            outputs=[fields_state]
        ).then(
            fn=update_preview,
            inputs=[user_prompt, fields_state],
            outputs=[preview]
        )
        
        user_prompt.change(
            fn=update_preview,
            inputs=[user_prompt, fields_state],
            outputs=[preview]
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
                model_tab = create_model_evaluation_tab()
            
            with gr.TabItem("验证工具"):
                # TODO: 实现验证工具界面
                pass
    
    return tab
