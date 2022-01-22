# Demo

---

## 1. Asynchronous Inference Server with FastAPI

First, run the backend inference server

```sh
$python inference_server.py
INFO:     Started server process [18987]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Then, on the different process, run the front end web app via `streamlit`. Specify the address of the inference server you booted up above by `-s` option. For example, if the server is booted at `http://127.0.0.1:8000`, pass `8000`.

```sh
$streamlit run demo/webapp_demo.py -- -s  8000
```

---

## 2. Realtime Inference

This requires a webcam connected to your computer.

First, uncomment the mode you want to try and comment all others out. Following example chose `MiDAS` to try.

```python
model = BiWAKO.MiDAS("weights/mono_depth_small")
# model = BiWAKO.U2Net("mobile")
# model = BiWAKO.HumanParsing("human_attribute")
```

Now, just run the script

```sh
$python demo/live_demo.py
```

---

## 3. Video Inference

Sample usage of [VideoPredictor](../api/video_predictor.md). Assign the path to input mp4 to `video` and run the script.

```sh
$python demo/video_inference_demo.py
```
