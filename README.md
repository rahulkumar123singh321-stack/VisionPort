
# VisionPort

A vision-based traffic object detection system using YOLOv8, Streamlit and FastAPI.

## Current Version

v0.1 - Pretrained YOLOv8n demo, FastAPI gateway and request trace.

## Codebase Tour

| Part | Location | Will contain |
|---|---|---|
| Data pipeline | data/, train/dataset.py | Loading images and annotations |
| Training loop | train/train.py | Model training, loss and checkpointing |
| Architecture file | train/model.py | YOLO-style backbone, neck and head |
| Evaluation harness | eval/ | mAP, latency and throughput measurements |
| Optimisation script | optimise/ | ONNX export and TensorRT optimisation |
| Triton config | serve/triton/ | Model repository and configuration |

## Project Structure

- app.py - Streamlit application
- env_check.py - Environment verification
- serve/ - FastAPI gateway and request trace
- data/ - Dataset storage
- models/ - Model storage
- eval/ - Model evaluation
- optimise/ - Model optimisation
- train/ - Training pipeline
- logs/ - Performance logs
- docs/ - Project documentation
- failures/ - Model failure cases

## Environment

See env_report.md.

Install dependencies using:

pip install -r requirements.txt

## How to Run

Start the FastAPI gateway:

uvicorn serve.gateway_trace:app --reload

Open the API documentation at:

http://127.0.0.1:8000/docs

Run the Streamlit application:

streamlit run app.py

## Known Limitations

- Uses pretrained YOLOv8n weights.
- CPU-only PyTorch environment currently configured.
- Model is not yet specifically trained on Indian traffic data.
