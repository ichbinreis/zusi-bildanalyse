# zusi-bildanalyse

## Getting started

```
python -m venv venv

# Unix
source venv/bin/activate

# Windows
.\venv\Scripts\activate

pip install -r requirements.txt
```

## Required files

```
https://huggingface.co/Kleinhe/CAMD/resolve/main/weights/ViT-B-32.pt > clip_model/ViT-B-32.pt
clip_patch/open_clip_pytorch_model.bin
clip_patch/model_encoded.txt
data/Bilder
data/alle_bilder_embeddings.npy
data/alle_bilder_index.csv
data/Objektdatenbank.csv
```
