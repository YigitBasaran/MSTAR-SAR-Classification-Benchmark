# SAR ATR — Notebook Edition (asel_proje_v2)

Notebook version of the MSTAR SAR Automatic Target Recognition case study:
**Handcrafted features (HOG+LBP+SVM) vs CNN (ResNet18) vs Vision Transformer
(DeiT-Small)** on the MSTAR SOC 10-class benchmark.

Same logic, metrics, plots and report as the script-based version — reorganized so
each experiment is a self-contained Jupyter notebook, shared code lives in `src/`,
and every experiment writes to its own `results/<experiment>/` folder.

## Structure
```
data/mstar_raw/soc/{train,test}/<CLASS>/*.jpg   # 2747 train / 2425 test (17°/15°)
src/                # shared library
  dataset.py features.py traditional.py cnn.py vit.py evaluate.py
  visualize.py eda_viz.py comparison.py
notebooks/
  01_eda.ipynb 02_traditional.ipynb 03_cnn.ipynb 04_vit.ipynb 05_comparison.ipynb
results/
  eda/ traditional/ cnn/ vit/ comparison/        # per-experiment plots + metrics + npz
models/             # resnet18_best.pth, deit_small_best.pth, traditional_pipeline.joblib
requirements.txt
```

## How to run
Install deps (`pip install -r requirements.txt`), then run the notebooks **in order**
(later notebooks depend on earlier outputs):

1. `01_eda.ipynb` — EDA + writes `results/eda/dataset_stats.json` (normalization stats).
2. `02_traditional.ipynb` — HOG+LBP+intensity → RBF-SVM.
3. `03_cnn.ipynb` — ResNet18 two-phase fine-tuning (needs the EDA stats).
4. `04_vit.ipynb` — DeiT-Small (head warm-up → full fine-tuning; needs the EDA stats).
5. `05_comparison.ipynb` — cross-model plots, Grad-CAM, `final_comparison.json`, `REPORT.md`
   (needs 02–04).

Headless (re-run all):
```
py -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 notebooks/01_eda.ipynb
# ...repeat for 02–05 in order
```

## Results (this run)
| Model | Accuracy | Macro-F1 |
|---|---|---|
| Handcrafted + SVM | 85.98% | 84.64% |
| ResNet18 (CNN) | 92.49% | 91.52% |
| DeiT-Small (ViT) | 95.79% | 95.03% |

Learned features clearly beat handcrafted descriptors, and the ViT (with full
fine-tuning) edges out the CNN. See `results/comparison/REPORT.md` for the full write-up.

## Notes
- Windows: use the `py` launcher; the data pipeline uses `num_workers=0`; keep `numpy < 2`
  (the installed `opencv`/`torch` builds are compiled against numpy 1.x).
- GPU training is not bit-reproducible, so numbers may move by ~1% between runs.
