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
https://huggingface.co/Kleinhe/CAMD/blob/main/weights/ViT-B-32.pt > clip_model/ViT-B-32.pt

https://drive.google.com/file/d/1lzHgA6mSH68tyUa7NNF_nqNO68y9hbes/view?usp=sharing > all the following files:

clip_patch/open_clip_pytorch_model.bin
clip_patch/model_encoded.txt
Daten/Bilder (containing pictures of all objects)
Daten/Objektdatenbank.csv
alle_bilder_embeddings.npy
alle_bilder_index.csv
```
