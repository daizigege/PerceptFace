# PerceptFace
Make Identity Unextractable yet Perceptible: Synthesis-Based Privacy Protection for Subject Faces in Photos

HugFace API: https://huggingface.co/spaces/daizigege/PerceptFace

Pre-trained models and insightface_func models
https://drive.google.com/drive/folders/1nRIuPJ7h8tNTQYud9yVNoKtOA_aS1dvp?usp=drive_link

The face alignment processing code must be employed to resize them to 224 pixels with "data_process". We use the face detection and alignment methods from InsightFace for image preprocessing. Please download the relevant files and unzip them to ./insightface_func/models. pip install insightface==0.2.1 onnxruntime moviepy. See details in  https://github.com/neuralchen/SimSwap/blob/main/docs/guidance/preparation.md

## Acknowledgement

Part of the code is  designed based on https://github.com/neuralchen/SimSwap/tree/main
face detection see 

