import requests
import json
import os

url = "http://127.0.0.1:8000/predict"
video_path = r"data/td/80_video.avi"

print(f"Testing {video_path}...")
with open(video_path, "rb") as f:
    files = {"file": (os.path.basename(video_path), f, "video/avi")}
    data = {"model_selection": "ensemble"}
    try:
        response = requests.post(url, files=files, data=data)
        print("Status Code:", response.status_code)
        try:
            print(json.dumps(response.json(), indent=2))
        except:
            print(response.text)
    except Exception as e:
        print("Error pinging localhost:8000 - is app.py running?", e)
