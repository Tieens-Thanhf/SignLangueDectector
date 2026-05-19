# SignLangueDectector
Sign language detector with Python, OpenCV and MediaPipe

## Pipeline

1) Put your dataset in `data\\<class_name>\\*.jpg`

2) Build landmark features (skips images with no detected hand):

```bash
python create_dataset.py
```

3) Train model:

```bash
python train_classifier.py --data data.pickle --out model.p
```

4) Realtime predict from webcam (press `q` to quit):

```bash
python test_classifier.py
```

## Collect images (optional)

```bash
python collect_imgs.py
```
