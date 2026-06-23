# Cat Breed Classifier

A deep-learning project that classifies cat images into 11 breeds using transfer
learning. Three pretrained convolutional backbones — **ResNet50**,
**DenseNet121**, and **MobileNetV3Large** — are fine-tuned and compared on
accuracy, mean average precision (mAP), and training time.

The entire workflow lives in [`ProjectAI/ProjectAI.ipynb`](ProjectAI/ProjectAI.ipynb):
dataset collection, train/val/test splitting, model training, evaluation, and
visualization.

## Breeds

The model recognizes 11 classes:

| | | |
|---|---|---|
| Abyssinian | Bengal | British Shorthair |
| Maine Coon | Norwegian Forest | Persian |
| Ragdoll | Russian Blue | Scottish Fold |
| Siamese | Sphynx | |

## Pipeline

1. **Data collection** — A Selenium script scrapes ~330 images per breed from
   Bing Images, with hashing to skip duplicates and resume support to avoid
   re-downloading.
2. **Splitting** — Images are shuffled and split per class into
   70% train / 15% validation / 15% test.
3. **Modeling** — Each backbone is loaded with ImageNet weights, frozen, and
   topped with a `GlobalAveragePooling2D` + `Dense(softmax)` head
   (`build_model` in the notebook).
4. **Training** — Models are compiled with Adam (`lr=1e-4`) and categorical
   cross-entropy, then trained for 50 epochs.
5. **Evaluation** — Accuracy, mAP, accuracy/loss curves, confusion matrices,
   and a side-by-side model comparison.

Dataset sizes used during training: **1931** train, **408** validation,
**426** test images.

## Project structure

```
.
├── ProjectAI/
│   ├── ProjectAI.ipynb              # Full pipeline notebook
│   ├── cat_images/                  # Dataset (gitignored)
│   │   ├── <breed>/                 # Raw downloaded images per breed
│   │   ├── train/ val/ test/        # Split datasets
│   ├── model_resnet50.h5            # Trained ResNet50
│   ├── model_densenet121.h5         # Trained DenseNet121
│   └── model_mobilenetv3large.h5    # Trained MobileNetV3Large
├── ProjectAI.html                   # Rendered/exported notebook
└── README.md
```

> **Note:** The `cat_images/` dataset folder and `*.h5` model files are excluded
> from version control via `.gitignore` due to their size.

## Requirements

```bash
pip install tensorflow scikit-learn numpy pandas matplotlib selenium requests
```

The data-collection cells additionally require Google Chrome and a matching
[ChromeDriver](https://chromedriver.chromium.org/) on your `PATH`.

## Usage

1. Open the notebook:

   ```bash
   jupyter notebook ProjectAI/ProjectAI.ipynb
   ```

2. (Optional) Run the scraping cells to build `cat_images/`, or supply your own
   images organized as `cat_images/<breed>/`.
3. Run the splitting cell to generate `train/`, `val/`, and `test/` folders.
4. Run the training cells to fit the three models and save the `.h5` files.
5. Run the evaluation cells to produce accuracy/mAP scores, comparison charts,
   and confusion matrices.

> **Path note:** Several cells contain hard-coded absolute Windows paths
> (e.g. `C:\Users\...\ProjectAI\cat_images`). Update these to match your own
> environment before running.

## Results

DenseNet121 was the strongest performer. Its densely connected architecture
promotes feature reuse and efficient gradient flow, letting it learn well even
with only simple rescaling and no heavy augmentation. ResNet50 trailed with
higher training/validation loss, while MobileNetV3Large — optimized for
lightweight, mobile inference — trained fastest but was less able to separate
visually similar breeds. See the comparison and confusion-matrix cells at the
end of the notebook for exact figures.
