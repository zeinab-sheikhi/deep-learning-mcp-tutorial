import asyncio
import json
import os 
from typing import List, Dict, TypedDict

import nest_asyncio
from anthropic import Anthropic
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


nest_asyncio.apply()


class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: Dict


class MCP_Chatbot:
  
    def __init__(self):

        # initiate session and client objects
        self.sessions: List[ClientSession] = []
        self.exit_stack = AsyncExitStack() 
        self.anthropic = Anthropic()
        self.available_tools: List[ToolDefinition] = []
        self.tool_to_session: Dict[str, ClientSession] = {}

    async def connect_to_server(self, server_name: str, server_config: Dict) -> None:
        """Connect to a single MCP server."""    
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.sessions.append(session)

            # List all available tools for this session
            response = await session.list_tools()
            tools = response.tools
            print(f"Connected to server {server_name} with tools:", [t.name for t in tools])

            for tool in tools:
                self.tool_to_session[tool.name] = session
                self.available_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                })
        
        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")
            raise
    
    async def connect_to_servers(self):
        """Connect to all configured MCP servers"""
        try:
            with open("server_config.json", "r") as f:
                data = json.load(f)
            servers = data.get("mcpServers", {})
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise

    async def cleanup(self):
        """Clean up all connections"""
        await self.exit_stack.aclose()
    
    async def process_query(self, query: str):
        messages = [{"role": "user", "content": query}]
        response = self.anthropic.messages.create(
            max_tokens=1024,
            model="claude-3-7-sonnet-20250219",  # Updated to latest model
            messages=messages,
            tools=self.available_tools
        )
        process = True
        while process:
            assistant_content = []
            for content in response.content:

                if content.type == "text":
                    print(f"Claude says: {content.text}")
                    assistant_content.append(content)
                    if len(response.content) == 1:
                        process = False

                elif content.type == "tool_use":
                    # what should we do here? we should extract the tool id and add a new message and then call the tool 
                    assistant_content.append(content)
                    messages.append({"role": "assistant", "content": assistant_content})
                    tool_id = content.id
                    tool_name = content.name
                    tool_args = content.input

                    print(f"Calling tools {tool_name} with {tool_args}")
                    
                    # call tool
                    session = self.tool_to_session[tool_name]
                    result = await session.call_tool(tool_name, tool_args)
                    messages.append({
                        "role": "user", 
                        "content": [
                            {
                                "type": "tool_result", 
                                "tool_use_id": tool_id, 
                                "content": result.content
                            }
                        ]
                    })
                    response = self.anthropic.messages.create(
                        max_tokens=1024,
                        model="claude-3-7-sonnet-20250219",  # Updated to latest model
                        tools=self.available_tools,
                        messages=messages,
                    )
                    if len(response.content) == 1 and response.content[0].type == "text":
                        print(response.content[0].text)
                        process = False

    async def chat_loop(self):
        """
        Run an interactive chat loop.
        """
        print("\nMCP chatbot started.")
        print("Type your query or 'quit' to exit")
        while True:
            try:
                query = input("\nQuery:").strip()
                if query.lower() == "quit":
                    break
                
                await self.process_query(query)

            except Exception as e:
                print(f"Error: {str(e)}")


async def main():
    chatbot = MCP_Chatbot()
    try:
        # the mcp clients and sessions are not initialized using "with"
        # so the cleanup should be manually handled
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
