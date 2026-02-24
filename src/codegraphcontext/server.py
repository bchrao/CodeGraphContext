# src/codegraphcontext/server.py
import asyncio
import json
import sys
import traceback

from typing import Any, Dict, Coroutine

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from .prompts import LLM_SYSTEM_PROMPT
from .core import get_database_manager
from .core.jobs import JobManager
from .core.watcher import CodeWatcher
from .tools.graph_builder import GraphBuilder
from .tools.code_finder import CodeFinder
from .utils.debug_log import debug_log, info_logger, error_logger, warning_logger, debug_logger

# Import Tool Definitions and Handlers
from .tool_definitions import TOOLS
from .tools.handlers import (
    analysis_handlers,
    indexing_handlers,
    management_handlers,
    query_handlers,
    watcher_handlers
)

DEFAULT_EDIT_DISTANCE = 2
DEFAULT_FUZZY_SEARCH = False

WORKSPACE_PREFIX = "/workspace/"


def _is_path_key(key: str) -> bool:
    """Check if a dict key represents a file path field.

    Matches keys like 'path', 'clone_path', 'caller_file_path', and also
    Cypher-aliased keys like 'f.path', 'n.caller_file_path'.
    """
    # Strip Cypher alias prefix (e.g. "f.path" -> "path")
    bare = key.rsplit(".", 1)[-1] if "." in key else key
    return bare == "path" or bare.endswith("_path")


def _strip_path_value(value):
    """Strip /workspace/ prefix from a single string value."""
    if isinstance(value, str) and value.startswith(WORKSPACE_PREFIX):
        return value[len(WORKSPACE_PREFIX):]
    return value


def _strip_workspace_prefix(obj):
    """Recursively strip /workspace/ prefix from path values in results."""
    if isinstance(obj, dict):
        return {
            k: _strip_path_value(v) if _is_path_key(k) else _strip_workspace_prefix(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [_strip_workspace_prefix(item) for item in obj]
    return obj


def _dict_to_tool(tool_dict: dict) -> types.Tool:
    """Convert a raw tool definition dict to an mcp types.Tool object."""
    return types.Tool(
        name=tool_dict["name"],
        description=tool_dict.get("description", ""),
        inputSchema=tool_dict.get("inputSchema", {"type": "object", "properties": {}}),
    )


class MCPServer:
    """
    The main MCP Server class.

    This class orchestrates all the major components of the application, including:
    - Database connection management (`DatabaseManager` or `FalkorDBManager`)
    - Background job tracking (`JobManager`)
    - File system watching for live updates (`CodeWatcher`)
    - Tool handlers for graph building, code searching, etc.
    - MCP SDK Server for standards-compliant JSON-RPC communication.
    """

    def __init__(self, loop=None):
        """
        Initializes the MCP server and its components.

        Args:
            loop: The asyncio event loop to use. If not provided, it gets the current
                  running loop or creates a new one.
        """
        try:
            # Initialize the database manager (Neo4j or FalkorDB Lite based on env var)
            # to fail fast if credentials/configuration are wrong.
            self.db_manager = get_database_manager()
            self.db_manager.get_driver()
        except ValueError as e:
            raise ValueError(f"Database configuration error: {e}")

        # Initialize managers for jobs and file watching.
        self.job_manager = JobManager()

        # Get the current event loop to pass to thread-sensitive components like the graph builder.
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        self.loop = loop

        # Initialize all the tool handlers, passing them the necessary managers and the event loop.
        self.graph_builder = GraphBuilder(self.db_manager, self.job_manager, loop)
        self.code_finder = CodeFinder(self.db_manager)
        self.code_watcher = CodeWatcher(self.graph_builder, self.job_manager)

        # Define the tool manifest that will be exposed to the AI assistant.
        self._init_tools()

        # Create the MCP SDK Server instance with system prompt as instructions
        self._mcp = Server(
            "CodeGraphContext",
            instructions=LLM_SYSTEM_PROMPT,
        )
        self._register_handlers()

    def _init_tools(self):
        """
        Defines the complete tool manifest for the LLM.
        """
        self.tools = TOOLS

    def _register_handlers(self):
        """Register list_tools and call_tool handlers with the MCP SDK Server."""

        @self._mcp.list_tools()
        async def list_tools() -> list[types.Tool]:
            return [_dict_to_tool(t) for t in self.tools.values()]

        @self._mcp.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
            try:
                result = await self.handle_tool_call(name, arguments or {})
                result = _strip_workspace_prefix(result)
            except Exception as e:
                error_logger(f"Tool call error ({name}): {e}\n{traceback.format_exc()}")
                return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]

            if "error" in result:
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    def get_database_status(self) -> dict:
        """Returns the current connection status of the Neo4j database."""
        return {"connected": self.db_manager.is_connected()}


    # --- Tool Wrappers ---
    # These methods delegate to the functional handlers, injecting the necessary dependencies.

    def execute_cypher_query_tool(self, **args) -> Dict[str, Any]:
        return query_handlers.execute_cypher_query(self.db_manager, **args)

    def visualize_graph_query_tool(self, **args) -> Dict[str, Any]:
        return query_handlers.visualize_graph_query(self.db_manager, **args)

    def find_dead_code_tool(self, **args) -> Dict[str, Any]:
        return analysis_handlers.find_dead_code(self.code_finder, **args)

    def calculate_cyclomatic_complexity_tool(self, **args) -> Dict[str, Any]:
        return analysis_handlers.calculate_cyclomatic_complexity(self.code_finder, **args)

    def find_most_complex_functions_tool(self, **args) -> Dict[str, Any]:
        return analysis_handlers.find_most_complex_functions(self.code_finder, **args)

    def analyze_code_relationships_tool(self, **args) -> Dict[str, Any]:
        return analysis_handlers.analyze_code_relationships(self.code_finder, **args)

    def find_code_tool(self, **args) -> Dict[str, Any]:
        return analysis_handlers.find_code(self.code_finder, **args)

    def list_indexed_repositories_tool(self, **args) -> Dict[str, Any]:
        return management_handlers.list_indexed_repositories(self.code_finder, **args)

    def delete_repository_tool(self, **args) -> Dict[str, Any]:
        return management_handlers.delete_repository(self.graph_builder, **args)

    def check_job_status_tool(self, **args) -> Dict[str, Any]:
        return management_handlers.check_job_status(self.job_manager, **args)

    def list_jobs_tool(self) -> Dict[str, Any]:
        return management_handlers.list_jobs(self.job_manager)

    def list_watched_paths_tool(self, **args) -> Dict[str, Any]:
        return watcher_handlers.list_watched_paths(self.code_watcher, **args)

    def unwatch_directory_tool(self, **args) -> Dict[str, Any]:
        return watcher_handlers.unwatch_directory(self.code_watcher, **args)

    def index_repository_tool(self, **args) -> Dict[str, Any]:
        return indexing_handlers.index_repository(
            self.graph_builder,
            self.job_manager,
            self.loop,
            self.list_indexed_repositories_tool,
            **args
        )

    def add_code_to_graph_tool(self, **args) -> Dict[str, Any]:
        return indexing_handlers.add_code_to_graph(
            self.graph_builder,
            self.job_manager,
            self.loop,
            self.list_indexed_repositories_tool,
            **args
        )

    def add_package_to_graph_tool(self, **args) -> Dict[str, Any]:
        return indexing_handlers.add_package_to_graph(
            self.graph_builder,
            self.job_manager,
            self.loop,
            self.list_indexed_repositories_tool,
            **args
        )

    def watch_directory_tool(self, **args) -> Dict[str, Any]:
        # watch_directory needs to call metadata tools.
        return watcher_handlers.watch_directory(
            self.code_watcher,
            self.list_indexed_repositories_tool,
            self.add_code_to_graph_tool,
            **args
        )

    def load_bundle_tool(self, **args) -> Dict[str, Any]:
        return management_handlers.load_bundle(self.code_finder, **args)

    def search_registry_bundles_tool(self, **args) -> Dict[str, Any]:
        return management_handlers.search_registry_bundles(self.code_finder, **args)

    def get_repository_stats_tool(self, **args) -> Dict[str, Any]:
        return management_handlers.get_repository_stats(self.code_finder, **args)


    async def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Routes a tool call from the AI assistant to the appropriate handler function.
        """
        tool_map: Dict[str, Coroutine] = {
            "index_repository": self.index_repository_tool,
            "add_package_to_graph": self.add_package_to_graph_tool,
            "find_dead_code": self.find_dead_code_tool,
            "find_code": self.find_code_tool,
            "analyze_code_relationships": self.analyze_code_relationships_tool,
            "watch_directory": self.watch_directory_tool,
            "execute_cypher_query": self.execute_cypher_query_tool,
            "add_code_to_graph": self.add_code_to_graph_tool,
            "check_job_status": self.check_job_status_tool,
            "list_jobs": self.list_jobs_tool,
            "calculate_cyclomatic_complexity": self.calculate_cyclomatic_complexity_tool,
            "find_most_complex_functions": self.find_most_complex_functions_tool,
            "list_indexed_repositories": self.list_indexed_repositories_tool,
            "delete_repository": self.delete_repository_tool,
            "visualize_graph_query": self.visualize_graph_query_tool,
            "list_watched_paths": self.list_watched_paths_tool,
            "unwatch_directory": self.unwatch_directory_tool,
            "load_bundle": self.load_bundle_tool,
            "search_registry_bundles": self.search_registry_bundles_tool,
            "get_repository_stats": self.get_repository_stats_tool
        }
        handler = tool_map.get(tool_name)
        if handler:
            # Run the synchronous tool function in a separate thread to avoid
            # blocking the main asyncio event loop.
            return await asyncio.to_thread(handler, **args)
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    async def run(self):
        """
        Runs the MCP server using the SDK's stdio transport.
        """
        print("MCP Server is running. Waiting for requests...", file=sys.stderr, flush=True)
        self.code_watcher.start()

        # Capture the running event loop so tool handlers can schedule async work
        self.loop = asyncio.get_running_loop()
        # Update graph_builder's loop reference as well
        self.graph_builder.loop = self.loop

        initialization_options = self._mcp.create_initialization_options(
            notification_options=None,
            experimental_capabilities=None,
        )

        async with stdio_server() as (read_stream, write_stream):
            await self._mcp.run(
                read_stream,
                write_stream,
                initialization_options,
            )

    def shutdown(self):
        """Gracefully shuts down the server and its components."""
        debug_logger("Shutting down server...")
        self.code_watcher.stop()
        self.db_manager.close_driver()
