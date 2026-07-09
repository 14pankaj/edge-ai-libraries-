# tests/test_plugins.py

import asyncio
from model_download_sdk.client import ModelDownloadSDK


async def main():
    client = ModelDownloadSDK()

    plugins = await client.list_plugins()

    print(plugins)

    await client.close()


asyncio.run(main())