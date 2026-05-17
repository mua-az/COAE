import asyncio
from fastmcp import Client

SERVER_URL = "http://154.57.164.78:30227/mcp"

async def main():
    client = Client(SERVER_URL)
    async with client:
        print("Fetching all available items...")
        # Read from the 'get_items' static resource URI
        response = await client.read_resource("resource://items")
        
        # Extract and print the text payload
        items_list = response[0].text
        print("\n=== Available Items ===")
        print(items_list)

if __name__ == "__main__":
    asyncio.run(main())