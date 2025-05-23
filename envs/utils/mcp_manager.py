# copy from qwen_agent
import time
import json
import uuid
import atexit
import asyncio
import threading
import jsonschema
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from envs.utils.util import json_loads
from typing import Union, List, Optional, Dict
from envs.utils.schema import ContentItem, is_tool_schema, has_chinese_chars


"""
要继承这个类
"""

class BaseTool(ABC):
    name: str = ''
    description: str = ''
    parameters: Union[List[dict], dict] = []

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        if not self.name:
            raise ValueError(
                f'You must set {self.__class__.__name__}.name, either by @register_tool(name=...) or explicitly setting {self.__class__.__name__}.name'
            )
        if isinstance(self.parameters, dict):
            if not is_tool_schema({'name': self.name, 'description': self.description, 'parameters': self.parameters}):
                raise ValueError(
                    'The parameters, when provided as a dict, must confirm to a valid openai-compatible JSON schema.')

    @abstractmethod
    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, dict, List[ContentItem]]:
        """The interface for calling tools.

        Each tool needs to implement this function, which is the workflow of the tool.

        Args:
            params: The parameters of func_call.
            kwargs: Additional parameters for calling tools.

        Returns:
            The result returned by the tool, implemented in the subclass.
        """
        raise NotImplementedError

    def _verify_json_format_args(self, params: Union[str, dict], strict_json: bool = False) -> dict:
        """Verify the parameters of the function call"""
        if isinstance(params, str):
            try:
                if strict_json:
                    params_json: dict = json.loads(params)
                else:
                    params_json: dict = json_loads(params)
            except json.decoder.JSONDecodeError:
                raise ValueError('Parameters must be formatted as a valid JSON!')
        else:
            params_json: dict = params
        if isinstance(self.parameters, list):
            for param in self.parameters:
                if 'required' in param and param['required']:
                    if param['name'] not in params_json:
                        raise ValueError('Parameters %s is required!' % param['name'])
        elif isinstance(self.parameters, dict):
            jsonschema.validate(instance=params_json, schema=self.parameters)
        else:
            raise ValueError
        return params_json

    @property
    def function(self) -> dict:  # Bad naming. It should be `function_info`.
        return {
            # 'name_for_human': self.name_for_human,
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
            # 'args_format': self.args_format
        }

    @property
    def name_for_human(self) -> str:
        return self.cfg.get('name_for_human', self.name)

    @property
    def args_format(self) -> str:
        fmt = self.cfg.get('args_format')
        if fmt is None:
            if has_chinese_chars([self.name_for_human, self.name, self.description, self.parameters]):
                fmt = '此工具的输入应为JSON对象。'
            else:
                fmt = 'Format the arguments as a JSON object.'
        return fmt

    @property
    def file_access(self) -> bool:
        return False


class MCPManager:
    _instance = None  # Private class variable to store the unique instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MCPManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'clients'): # The singleton should only be inited once
            """Set a new event loop in a separate thread"""
            try:
                import mcp
            except ImportError as e:
                raise ImportError('Could not import mcp. Please install mcp with `pip install -U mcp`.') from e

            load_dotenv()  # Load environment variables from .env file
            self.clients: dict = {}
            self.loop = asyncio.new_event_loop()
            self.loop_thread = threading.Thread(target=self.start_loop, daemon=True)
            self.loop_thread.start()

            # A fallback way to terminate MCP tool processes after Qwen-Agent exits
            self.processes = []
            self.monkey_patch_mcp_create_platform_compatible_process()
    
    def monkey_patch_mcp_create_platform_compatible_process(self):
        try:
            import mcp.client.stdio
            target = mcp.client.stdio._create_platform_compatible_process
        except (ModuleNotFoundError, AttributeError) as e:
            raise ImportError(
                'Qwen-Agent needs to monkey patch MCP for process cleanup. '
                'Please upgrade MCP to a higher version with `pip install -U mcp`.'
            ) from e

        async def _monkey_patched_create_platform_compatible_process(*args, **kwargs):
            process = await target(*args, **kwargs)
            self.processes.append(process)
            return process
        mcp.client.stdio._create_platform_compatible_process = _monkey_patched_create_platform_compatible_process

    def start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def is_valid_mcp_servers(self, config: dict):
        """Example of mcp servers configuration:
        {
         "mcpServers": {
            "memory": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-memory"]
            },
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/files"]
            }
         }
        }
        """

        # Check if the top-level key "mcpServers" exists and its value is a dictionary
        if not isinstance(config, dict) or 'mcpServers' not in config or not isinstance(config['mcpServers'], dict):
            return False
        mcp_servers = config['mcpServers']
        # Check each sub-item under "mcpServers"
        for key in mcp_servers:
            server = mcp_servers[key]
            # Each sub-item must be a dictionary
            if not isinstance(server, dict):
                return False
            if 'command' in server:
                # "command" must be a string
                if not isinstance(server['command'], str):
                    return False
                # "args" must be a list
                if 'args' not in server or not isinstance(server['args'], list):
                    return False
            if 'url' in server:
                # "url" must be a string
                if not isinstance(server['url'], str):
                    return False
                # "headers" must be a dictionary
                if 'headers' in server and not isinstance(server['headers'], dict):
                    return False
            # If the "env" key exists, it must be a dictionary
            if 'env' in server and not isinstance(server['env'], dict):
                return False
        return True

    def initConfig(self, config: Dict):
        # logger.info(f'Initializing MCP tools from mcpservers config: {config}')
        if not self.is_valid_mcp_servers(config):
            raise ValueError('Config of mcpservers is not valid')
        # Submit coroutine to the event loop and wait for the result
        future = asyncio.run_coroutine_threadsafe(self.init_config_async(config), self.loop)
        try:
            result = future.result()  # You can specify a timeout if desired
            return result
        except Exception as e:
            # logger.info(f'Failed in initializing MCP tools: {e}')
            raise e

    async def init_config_async(self, config: Dict):
        tools: list = []
        mcp_servers = config['mcpServers']
        for server_name in mcp_servers: # multi-tools的接口。server_name = meituan_search
            client = MCPClient() # sse to do
            server = mcp_servers[server_name]
            await client.connection_server(server)  # Attempt to connect to the server

            client_id = server_name + '_' + str(uuid.uuid4()) # To allow the same server name be used across different running agents
            self.clients[client_id] = client  # Add to clients dict after successful connection
            for tool in client.tools:
                """MCP tool example:
                {
                "name": "read_query",
                "description": "Execute a SELECT query on the SQLite database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                        "type": "string",
                        "description": "SELECT SQL query to execute"
                        }
                    },
                    "required": ["query"]
                }
                """
                parameters = tool.inputSchema
                # The required field in inputSchema may be empty and needs to be initialized.
                if 'required' not in parameters:
                    parameters['required'] = []
                # Remove keys from parameters that do not conform to the standard OpenAI schema
                # Check if the required fields exist
                required_fields = {'type', 'properties', 'required'}
                missing_fields = required_fields - parameters.keys()
                if missing_fields:
                    raise ValueError(f'Missing required fields in schema: {missing_fields}')

                # Keep only the necessary fields
                cleaned_parameters = {
                    'type': parameters['type'],
                    'properties': parameters['properties'],
                    'required': parameters['required']
                }
                register_name = server_name + '-' + tool.name
                agent_tool = self.create_tool_class(register_name=register_name, register_client_id=client_id, 
                                                    tool_name=tool.name, tool_desc=tool.description, tool_parameters=cleaned_parameters)
                tools.append(agent_tool)
        return tools

    def create_tool_class(self, register_name, register_client_id, tool_name, tool_desc, tool_parameters):

        class ToolClass(BaseTool):
            name = register_name
            description = tool_desc
            parameters = tool_parameters
            client_id = register_client_id

            def call(self, params: Union[str, dict], **kwargs) -> str:
                tool_args = json.loads(params)
                # Submit coroutine to the event loop and wait for the result
                manager = MCPManager()
                client = manager.clients[register_client_id]
                future = asyncio.run_coroutine_threadsafe(client.execute_function(tool_name, tool_args), manager.loop)
                try:
                    result = future.result()
                    return result
                except Exception as e:
                    # logger.info(f'Failed in executing MCP tool: {e}')
                    raise e

        ToolClass.__name__ = f'{register_name}_Class'
        return ToolClass()

    def shutdown(self):
        futures = []
        for client_id in list(self.clients.keys()):
            client :MCPClient = self.clients[client_id]
            future = asyncio.run_coroutine_threadsafe(client.cleanup(), self.loop)
            futures.append(future)
            del self.clients[client_id]
        time.sleep(1) # Wait for the graceful cleanups, otherwise fall back
        
        # fallback
        if asyncio.all_tasks(self.loop):
            # logger.info('There are still tasks in `MCPManager().loop`, force terminating the MCP tool processes. There may be some exceptions.')
            for process in self.processes:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass # it's ok, the process may exit earlier
        
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join()


class MCPClient:
    def __init__(self):
        from mcp import ClientSession

        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.tools: list = None
        self.exit_stack = AsyncExitStack()

    async def connection_server(self, mcp_server): # {'url': 'http://10.46.7.131:8080/sse'}
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.client.sse import sse_client
        """Connect to an MCP server and retrieve the available tools."""
        try:
            if 'url' in mcp_server:
                # self._streams_context = sse_client(url=mcp_server.get('url'), headers=mcp_server.get('headers', {"Accept": "text/event-stream"}))
                self._streams_context = sse_client(url=mcp_server.get('url'))

                streams = await self.exit_stack.enter_async_context(self._streams_context)
                self._session_context = ClientSession(*streams)
                self.session = await self.exit_stack.enter_async_context(self._session_context)
            else:
                server_params = StdioServerParameters(
                    command = mcp_server["command"],
                    args = mcp_server["args"],
                    env = mcp_server.get("env", None)
                )
                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                self.stdio, self.write = stdio_transport
                self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
                # logger.info(f'Will initialize a MCP stdio_client, if this takes forever, please check whether the mcp config is correct: {mcp_server}')

            await self.session.initialize()
            list_tools = await self.session.list_tools()
            self.tools = list_tools.tools
        except Exception as e:
            # logger.info(f"Failed in connecting to MCP server: {e}")
            raise e

    async def execute_function(self, tool_name, tool_args: dict):
        response = await self.session.call_tool(tool_name, tool_args)
        texts = []
        for content in response.content:
            if content.type == 'text':
                texts.append(content.text)
        if texts:
            return '\n\n'.join(texts)
        else:
            return 'execute error'

    async def cleanup(self):
        await self.exit_stack.aclose()


def _cleanup_mcp(_sig_num=None, _frame=None):
    if MCPManager._instance is None:
        return
    manager = MCPManager()
    manager.shutdown()


# Make sure all subprocesses are terminated even if killed abnormally:
# If not running in the main thread, (for example run in streamlit)
# register a signal would cause a RuntimeError
if threading.current_thread() is threading.main_thread():
    atexit.register(_cleanup_mcp)