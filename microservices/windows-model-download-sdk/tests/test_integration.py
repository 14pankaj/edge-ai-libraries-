# from model_download_sdk.client import ModelDownloadSDK, SDKConfig

# config = SDKConfig(
#     base_url="http://localhost:8200"
# )

# client = ModelDownloadSDK(config=config)

# print("SDK created")

# from model_download_sdk.client import ModelDownloadSDK, SDKConfig

# config = SDKConfig(
#     base_url="http://localhost:8200"
# )

# client = ModelDownloadSDK(config=config)

# result = client.health_check()

# print(result)

import asyncio

from model_download_sdk.client import ModelDownloadSDK, SDKConfig


async def main():
    config = SDKConfig(
        base_url="http://localhost:8200"
    )

    client = ModelDownloadSDK(config=config)

    result = await client.health_check()

    print(result)


if __name__ == "__main__":
    asyncio.run(main())