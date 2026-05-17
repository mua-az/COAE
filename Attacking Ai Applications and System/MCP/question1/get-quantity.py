import asyncio
from fastmcp import Client

SERVER_URL = "http://154.57.164.78:30227/mcp"

async def main():
    client = Client(SERVER_URL)
    async with client:
        # Read the logs resource using its URI
        response = await client.read_resource("quantity://banana")
        
        # Extract and print the text content
        logs_text = response[0].text
        print("=== Server Logs ===")
        print(logs_text)

if __name__ == "__main__":
    asyncio.run(main())