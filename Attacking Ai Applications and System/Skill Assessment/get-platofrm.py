import asyncio
from fastmcp import Client

client = Client("http://154.57.164.72:30981/mcp/")

async def main():
    async with client:
        result = await client.read_resource("resource://error_logs")
        print(result[0].text)

asyncio.run(main())