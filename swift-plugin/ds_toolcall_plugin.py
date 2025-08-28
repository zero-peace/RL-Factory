import asyncio
import re
import textwrap
from copy import deepcopy
from typing import Dict, List, Optional

import json
import torch, os, sys
import torch.distributed as dist
from json import JSONDecodeError
from swift.llm import PtEngine, RequestConfig, Template, to_device
from swift.llm.infer.protocol import ChatCompletionResponse
from swift.plugin import multi_turns
from swift.plugin.multi_turn import MultiTurnScheduler
from swift.utils import get_logger
from envs.utils.tool_utils import ToolUtils
from envs import TOOL_ENV_REGISTRY

class ENV(object):
    def __init__(self, config):
        self.cfg = config
        for k,v in config.items():
           setattr(self, k, v)
    def get(self, key, default_value=None):
        if key not in self.cfg:
            return default_value
        return self.cfg[key]

env_dict = {
    "name": "search",
    "mcp_mode": "stdio",
    "tool_manager": "qwen3",
    "enable_thinking": False,
    "config_path": "envs/configs/sse_mcp_tools.pydata",
    "use_process_reward": False,
    "model_type": "null",
    "tool_name_selected": [],      # selected tools for using when call a sse MCP server; default [] will call all tools
    "max_prompt_length": 4096,
    "enable_limiter": False,
    "tool_timeout": 10,            # this is the timeout for each tool call
    "max_concurrency": 10,         # this is the max concurrency for tool manager
    "use_storage_manager": False,
    "max_size": 1000,
    "cache": "single",
    "persist": "false",
    "eviction": "lru",
    "persist_dir": "./cache",
    "local_cache": "null",
    "parallel_sse_tool_call": ENV({
      "is_enabled": True,
      "num_instances": 3
    })
}

from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union
if TYPE_CHECKING:
    from swift.llm.infer.protocol import RolloutResponseChoice
    from swift.llm.template import RolloutInferRequest

class RLFactoryToolcall(MultiTurnScheduler):
    def __init__(self, max_turns=None, *args, **kwargs):

        max_turns = 6
        super().__init__(max_turns, *args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        env = ENV(env_dict)
        self.env_object = TOOL_ENV_REGISTRY[env.name](
            config=env, 
            centralized_actor=False
        )

        # if /log dir exist
        if not os.path.exists('/log'):
            self.log_path = '/log/log_result.txt'
        else:
            self.log_path = './log_result.txt'
        with open(self.log_path, 'w') as f:
            f.write('')

    def check_finished(self, infer_request: 'RolloutInferRequest', result: 'RolloutResponseChoice',
                       current_turn: int) -> bool:

        action, tool = self.env_object.tool_manager.parse_response(infer_request.messages[-1]["content"])
        if action == 'answer':
            # If action is 'answer', end the conversation
            return True

        #length too long or reaches max turn
        return super().check_finished(infer_request, result, current_turn)

    def step(
        self,
        infer_request: 'RolloutInferRequest',
        result: 'RolloutResponseChoice',
        current_turn: int,
    ):
        extra_dict = {}

        # if self.env_object.use_process_reward:
            
        # if user ask for toolcall, toolcall
        #for msg in infer_request.messages:

        # Last request
        msg = infer_request.messages[-1]

        #if not msg['role'] == 'assistant':
        #    continue
        step_score = self.env_object.get_step_reward([infer_request.messages[-1]['content']])
        if '<tool_call>' in msg['content']:
            if self.log_path:
                with open(self.log_path , 'a') as f:
                    f.write("-----\n")
                    f.write(f"length after update: {len(infer_request.messages)}\n")
                    f.write("prior messages: " + str(infer_request.messages) + '\n')

            next_obs, dones, valid_action, is_tool = self.env_object.step([msg['content']])
            infer_request.messages.append(next_obs[-1][-1])

            step_scores_full = self.env_object.get_step_reward(responses=[d['content'] for d in infer_request.messages])

            if self.log_path:
                with open(self.log_path, 'a') as f:
                    f.write("-----\n")
                    f.write(f"length after update: {len(infer_request.messages)}\n")
                    #f.write("updated messages: " + str(infer_request.messages) + '\n')
                    f.write("updated messages: \n")
                    for d in infer_request.messages:
                        f.write("\t" + str(d) + "\n")
                    f.write("\n")
                    f.write("result: " + str(result) + "\n")
                    f.write("-----\n")
                    f.write("====== content: <<<<\n\t" + msg['content'] + "\n>>>>\n")
                    f.write("====== tool: " + str(next_obs) + "\n")
                    f.write("====== dones: " + str(dones) + "\n")
                    f.write("====== valid_action: " + str(valid_action) + "\n")
                    f.write("====== is_tool: " + str(is_tool) + "\n")
                    f.write("----- \n")
                    f.write("====== step_scores: " + str(step_score) + " full score: " + str(step_scores_full) + "\n")
                    f.write("----- \n\n\n\n\n")

        else:
            if self.log_path:
                with open(self.log_path, 'a') as f:
                    f.write("!!-----\n")
                    f.write(f"====== no tool call in:\n {msg['content']}\n")
                    f.write(f"====== full massages:\n")
                    for d in infer_request.messages:
                        f.write("\t" + str(d) + "\n")
                    f.write("\n")
                    f.write("====== step_scores: " + str(step_score) + "\n")
                    f.write("!!----- \n\n\n\n\n")
        extra_dict["rollout_reward"] = step_score

        return infer_request, extra_dict

multi_turns['rlfactory_toolcall'] = RLFactoryToolcall
