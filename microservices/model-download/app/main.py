import os
import gc
import subprocess
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import List, Optional, TypedDict
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, ValidationError
from huggingface_hub import snapshot_download
from .logger import logger

app = FastAPI(root_path="/api/v1",title="Model Download Service", version="1.0.0")

class ModelPrecision(str, Enum):
    INT8 = "int8"
    FP16 = "fp16"
    FP32 = "fp32"

class DeviceType(str, Enum):
    CPU = "CPU"
    GPU = "GPU"

class OVMSConfig(BaseModel):
    precision: ModelPrecision = ModelPrecision.INT8
    device: DeviceType = DeviceType.CPU
    cache_size: Optional[int] = Field(None, gt=0)

class ModelResult(TypedDict):
    status: str
    model_name: str
    model_path: Optional[str]
    error: Optional[str]
    is_ovms: Optional[bool]

class ModelRequest(BaseModel):
    name: str
    type: Optional[str] = None
    is_ovms: bool = False
    revision: Optional[str] = None
    model_family: Optional[str] = None
    description: Optional[str] = None
    cache_dir: Optional[str] = None
    ovms_config: Optional[OVMSConfig] = None

class ModelDownloadRequest(BaseModel):
    models: List[ModelRequest]
    parallel_downloads: Optional[bool] = False

@app.post("/models/download")
async def download_huggingface_models(
    request: ModelDownloadRequest,
    download_path: str = "models",
    Authorization: str = Header(...)
):
    """
    Unified endpoint to download one or more models from Hugging Face.

    Request format:
    {
        "models": [
            {"name": "model1", "type": "llm", "is_ovms": true},
            {"name": "model2", "type": "llm", "is_ovms": false}
        ],
        "parallel_downloads": true  # Optional, defaults to false
    }

    For single model download, simply include one model in the list:
    {
        "models": [
            {"name": "model1", "type": "llm", "is_ovms": true}
        ]
    }
    """
    try:
        if not Authorization:
            raise HTTPException(
                status_code=401,
                detail="Authorization token is empty."
            )
        # Validate request
        os.environ["HF_TOKEN"] = Authorization

        # Log download request details with configuration
        logger.info(f"Initiating model download for {len(request.models)} model(s)")

        # Process models either in parallel or sequentially
        with ThreadPoolExecutor(max_workers=len(request.models) if request.parallel_downloads else 1) as executor:
            results = list(executor.map(
                lambda model: download_and_process_model(
                    model=model,
                    model_path=download_path,
                    hf_token=Authorization
                ),
                request.models
            ))

        gc.collect()

        response = {
            "message": "Model download completed",
            "results": results
        }

        # For single model requests, maintain backward compatibility in response format
        if len(request.models) == 1:
            result = results[0]
            if result["status"] == "success":
                response.update({
                    "message": "Model downloaded successfully",
                    "model_path": result["model_path"],
                })
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error downloading model: {result['error']}"
                )

        return response

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in model download process: {str(e)}")

def download_and_process_model(model: ModelRequest, model_path: str, hf_token: str) -> ModelResult:
    """
    Download a model from Hugging Face and optionally convert it to OVMS format
    
    Args:
        model: The model request containing name, type, OVMS flag and configurations
        model_path: Base path for model downloads
        hf_token: Hugging Face API token
        
    Returns:
        ModelResult containing the status and details of the model processing
    """
    try:
        logger.info(f"Starting download for model: {model.name}")
        
        # Create model-specific directory
        model_specific_path = os.path.join(model_path, model.name.replace('/', '_'))
        os.makedirs(model_specific_path, exist_ok=True)
        
        # Download model from Hugging Face
        model_downloaded_path = snapshot_download(
            repo_id=model.name,
            token=hf_token,
            local_dir=model_specific_path
        )
        logger.info(f"Model download completed: {model.name}")

        # Convert if OVMS is requested and model type is provided
        if model.is_ovms and model.type:
            logger.info(f"Starting OVMS conversion for model: {model.name}")

            # Use model-specific OVMS config if provided, otherwise initialize default settings
            ovms_config = model.ovms_config if model.ovms_config else OVMSConfig(
                precision=ModelPrecision.INT8,
                device=DeviceType.CPU,
                cache_size=10
            )
            
            # Prepare OVMS configuration
            model_downloaded_path = os.path.join(ovms_config.device.value,model_specific_path)
            os.makedirs(model_downloaded_path, exist_ok=True)
            ovms_params = {
                "model_name": model.name,
                "weight_format": ovms_config.precision.value,
                "target_device": ovms_config.device.value,
                "huggingface_token": hf_token,
                "model_type": model.type,
                "model_directory": model_downloaded_path,
                "cache_size": ovms_config.cache_size or None
            }
            
            # Filter out None values
            ovms_params = {k: v for k, v in ovms_params.items() if v is not None}
            
            convert_to_ovms_format(**ovms_params)
            logger.info(f"OVMS conversion completed for model: {model.name}")

        return ModelResult(
            status="success",
            model_name=model.name,
            model_path=model_downloaded_path,
            is_ovms=model.is_ovms,
            error=None
        )

    except Exception as e:
        logger.error(f"Model processing failed for {model.name}: {str(e)}")
        return ModelResult(
            status="error",
            model_name=model.name,
            model_path=None,
            error=str(e),
            is_ovms=None
        )

def convert_to_ovms_format(
    model_name: str,
    weight_format: str,
    huggingface_token: str,
    model_type: str,
    target_device: str,
    model_directory: str,
    cache_size: Optional[int] = None,
):
    """
    Convert a downloaded model to OpenVINO Model Server (OVMS) format.

    Args:
        model_name (str): The name of the Hugging Face model to download.
        weight_format (str): The weight format for the exported model (e.g., "int4", "fp16").
        huggingface_token (str): The Hugging Face API token for authentication.
        model_type (str): The type of the model (e.g., "llm", "embeddings", "rerank").
        cache_size (int, optional): Cache size for model optimization.

    Returns:
        dict: A success message if the process completes successfully.

    Raises:
        HTTPException: If any step in the process fails.
    """
    try:
        # Map model_type to export type
        export_type_map = {
            "llm": "text_generation",
            "embeddings": "embeddings",
            "rerank": "rerank"
        }

        # Validate model_type
        if model_type not in export_type_map:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type: {type}. Must be one of {list(export_type_map.keys())}."
            )

        export_type = export_type_map[model_type]

        # Step 1: Log in to Hugging Face
        logger.info("Logging in to Hugging Face...")
        result = subprocess.run(["huggingface-cli", "login", "--token", huggingface_token])
        if result.returncode != 0:
            raise HTTPException(status_code=400, detail="Failed to log in to Hugging Face. Please check your token.")

        # Step 2: Download the export_model.py script
        logger.info("Downloading export_model.py script...")
        export_script_url = "https://raw.githubusercontent.com/openvinotoolkit/model_server/refs/heads/releases/2025/0/demos/common/export_models/export_model.py"
        subprocess.run(["curl", "-o", "export_model.py", export_script_url], check=True)

        # Step 3: Export the model
        logger.info(f"Exporting model: {model_name} with weight format: {weight_format} and export type: {export_type}...")
        #models directory
        os.makedirs(model_directory, exist_ok=True)
        
        # Build command with base arguments
        command = [
            "python3", "export_model.py", export_type,
            "--source_model", model_name,
            "--weight-format", weight_format,
            "--config_file_path", f"{model_directory}/config.json",
            "--model_repository_path", model_directory,
            "--target_device", target_device
        ]
        # Add optional parameters if provided
        if cache_size is not None:
            command += ["--cache", str(cache_size)]

        subprocess.run(command, check=True)

        return {"message": f"Model successfully downloaded, converted, and prepared for OVMS deployment as {export_type}."}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during the process: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")