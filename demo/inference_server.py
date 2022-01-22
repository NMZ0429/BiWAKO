import shutil
from pathlib import Path
from typing import Dict, List, Optional

import BiWAKO
import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Query, UploadFile

app = FastAPI()

AVAILABLE_MODELS = BiWAKO.available_models


@app.get("/models")
def get_models() -> Dict[str, List[str]]:
    """Return the list of available models.

    Returns:
        Dict[str, List[str]]: A dictionary of the list of available versions.
    """
    return {"models": AVAILABLE_MODELS}


@app.post("/predict")
async def post_predict(
    files: List[UploadFile] = File(...),
    model_choise: Optional[str] = Query(default="U2Net", description="Model name"),
    backgroud_tasks: BackgroundTasks = None,
) -> Dict[str, str]:
    """Request prediction for a list of files.

    Prediction is done in a background task. Results are saved to the directory "results".

    Args:
        files (List[UploadFile], optional): List of files to predict. Defaults to File(...).
        model_choise (Optional[str], optional): Model name. Defaults to Query(default="U2Net", description="Model name").
        backgroud_tasks (Optional[BackgroundTasks], optional): Background tasks. Defaults to None.

    Returns:
        Dict[str, str]: Message indicating the number of files processed.
    """
    if model_choise not in AVAILABLE_MODELS:
        return {"error": f"Model {model_choise} is not available."}
    else:
        model = eval(f"BiWAKO.{model_choise}()")

    save_path = Path("./results")
    if not save_path.exists():
        save_path.mkdir(parents=True)

    num_predictions = 0
    for file in files:
        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)  # type: ignore
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        mask = model.render(model.predict(img), img)

        print(f"Saving {file.filename}")
        cv2.imwrite(f"./results/{file.filename}", mask)
        num_predictions += 1

    return {"message": f"{num_predictions}  Prediction complete"}


@app.post("/delete")
async def request_delete() -> Dict[str, str]:
    """Delete all prediction results.

    Returns:
        Dict[str, str]: message json indicating success.
    """
    save_path = Path("./results")
    if save_path.exists():
        shutil.rmtree(save_path)
        return {"message": "Prediction results deleted"}

    return {"message": "No prediction results to delete"}


@app.get("/result")
async def get_result() -> Dict[str, str]:
    """Return the prediction result.

    Returns:
        Dict[str, str]: json of prediction index and its path.
    """
    save_path = Path("./results")
    results = [str(p) for p in save_path.glob("**/*")]

    rtn = {}
    for i, r in enumerate(results):
        rtn[f"image_{i}"] = r

    return rtn


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reload", "-r", action="store_true")
    parser.add_argument("--host", "-H", default="127.0.0.1", type=str)
    parser.add_argument("--port", "-p", default=8000, type=int)
    args = parser.parse_args()

    uvicorn.run(
        "inference_server:app", reload=args.reload, host=args.host, port=args.port
    )
