# tests/test_download_wait.py

import asyncio

from model_download_sdk.client import ModelDownloadSDK


async def main():
    client = ModelDownloadSDK()

    result = await client.download_model(
        model_name="bert-base-uncased",
        hub="huggingface",
        download_path="/tmp/models",
        wait=True,
        timeout=300,
    )

    print(result)

    if result.successful_jobs:
        print("SUCCESS")
        print(result.successful_jobs[0])

    if result.failed_jobs:
        print("FAILED")
        print(result.failed_jobs[0])


asyncio.run(main())