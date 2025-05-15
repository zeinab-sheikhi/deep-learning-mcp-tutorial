import asyncio

from typing import List, Dict

import nest_asyncio
from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


nest_asyncio.apply()
load_dotenv(dotenv_path=".env")


class MCP_Chatbot:
  
    def __init__(self):
        
        # initiate session and client objects
        self.session: ClientSession = None
        self.anthropic = Anthropic()
        self.available_tools: List[Dict] = None
    
    async def process_query(self, query: str):
        messages = [{"role": "user", "content": query}]
        response = self.anthropic.messages.create(
            max_tokens=1024, 
            model="claude-3-5-sonnet-20240620",
            messages=messages,
            tools=self.available_tools
        )
        process = True
        while process:
            assistant_content = []
            for content in response.content:
                
                if content.type == "text":
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
                    result = self.session.call_tool(tool_name, tool_args)
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
                        model="claude-3-5-sonnet-20240620",
                        tools=self.available_tools, 
                        messages=messages,
                    )
                    if len(response.content) == 1 and response.content.type == "text":
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
                if query.lower == "quit":
                    break
                
                await self.process_query(query)

            except Exception as e:
                print(f"Error: {str(e)}")
    
    async def connect_to_server_and_run(self):
        # create server params for Stdio connection
        server_aprams = StdioServerParameters(
            command="uv",
            args=["run", "stdio_server.py"],
            env=None,
        )
        async with stdio_client(server_aprams) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                await self.session.initialize()

                # List all the available tools 
                response = await self.session.list_tools()

                tools = response.tools

                print(f"Connected to server with tools: {[tool.name for tool in tools]}")

                self.available_tools = [{
                    "name": tool.name, 
                    "description": tool.description, 
                    "inputSchema": tool.inputSchema,
                } for tool in response.tools]

                await self.chat_loop()


async def main():
    chatbot = MCP_Chatbot()
    await chatbot.connect_to_server_and_run()


if __name__ == "__main__":
    asyncio.run(main())
