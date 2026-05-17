import asyncio
from fastmcp import Client

# Initialize the client pointing to your MCP server
client = Client("http://154.57.164.78:30227/mcp")

async def main():
    async with client:
        # Fetch available tools from the server
        tools = await client.list_tools()
        
        # Call the file storage tool
        result_object = await client.call_tool(
            "store_file", 
            {"file_content": "Hello World!", "file_name": "helloworld"}
        )
        
        # CORRECTED: Access the text within the first item of the content list
        result_text = result_object.content[0].text

        print(f"*** Available Tools:\n{tools}\n*** Tool Result:\n{result_text}\n")

if __name__ == "__main__":
    asyncio.run(main())
