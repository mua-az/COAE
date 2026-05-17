import asyncio
from fastmcp import Client

client = Client("http://154.57.164.72:30981/mcp/")

async def main():
    async with client:
        resources = await client.list_resources()
        for r in resources:
            print(r.uri, "->", r.name)

asyncio.run(main())