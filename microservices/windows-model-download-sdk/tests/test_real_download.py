# tests/test_real_download.py

import asyncio
from model_download_sdk.client import ModelDownloadSDK

async def main():
    client = ModelDownloadSDK()

    result = await client.download_model(
        model_name="bert-base-uncased",
        hub="huggingface",
        download_path="/tmp/models",
        wait=False,
    )

    print(result)

    await client.close()

asyncio.run(main())