# zusi-bildanalyse

## Getting started

```
python -m venv venv

# Unix
source venv/bin/activate

# Windows
.\venv\Scripts\activate

pip install -r requirements.txt

pip install "rembg[cpu]"
```

## Required files

```
https://huggingface.co/Kleinhe/CAMD/blob/main/weights/ViT-B-32.pt > clip_model/ViT-B-32.pt

Please note, for v1.1 you will need this link:
https://drive.google.com/file/d/1lzHgA6mSH68tyUa7NNF_nqNO68y9hbes/view?usp=sharing
For v3.0 you will need this link: https://drive.google.com/file/d/1MHL2Cx8hkTieq4Oqk6KR0Uxln86YNFUN/view?usp=sharing

You get all the following files:

clip_patch/open_clip_pytorch_model.bin
clip_patch/model_encoded.txt
Daten/Bilder and Daten/Bilder_2 (containing pictures of all objects)
Daten/Objektdatenbank.csv
alle_bilder_embeddings.npy
alle_bilder_index.csv

```

## Changelog

v3.0
- removed "advanced analysis mode"
- added "remote background" (optional)
- approximately 500 more objects (pictures + advanced database)
- button "copy path" now copies the whole path including the beginning of the path

v1.1
- Reduced from 3 to 2 analysis types (fast and not so fast)
- Now also supports absolute paths (for objects not located in \Routes)
- 597 objects marked as unsuitable for "Geländeformer", warning message integrated in evaluation
- ETA fixed, now shows ETA for both the currently analyzed and the total analysis
- Progress bar fixed
