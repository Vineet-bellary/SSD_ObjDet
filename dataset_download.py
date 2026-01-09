from roboflow import Roboflow
from dotenv import load_dotenv
import os

load_dotenv()

# This is to download Dataset 1 from Roboflow
rf = Roboflow(api_key="apUiV7snDplgUzUnTj5Y")
project = rf.workspace("small-object-detections-smart-surveillance-system").project(
    "cgi-weapon-dataset-q6mia"
)
version = project.version(1)
dataset = version.download("tensorflow")

print("Dataset downloaded successfully...")


# This is to download Dataset 2 from Roboflow
rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("small-object-detections-smart-surveillance-system").project(
    "cgi-weapon-dataset-q6mia"
)
version = project.version(2)
dataset = version.download("tensorflow")
print("Dataset downloaded successfully...")
