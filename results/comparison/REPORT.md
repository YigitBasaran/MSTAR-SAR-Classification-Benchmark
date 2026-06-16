# SAR Automatic Target Recognition: Handcrafted Features vs Deep Learning vs Vision Transformers

## 1. Problem Statement
Classify military vehicles from Synthetic Aperture Radar (SAR) imagery on the
MSTAR benchmark. SAR images contain speckle noise and look very different from
optical imagery, which makes target recognition challenging.

## 2. Dataset
MSTAR Standard Operating Condition (SOC), 10 classes, single-channel 128×128
images. Train/test follow the standard depression-angle split (train = 17°,
test = 15°): **2747 train / 2425 test** images. An additional
stratified 15% of the training set is held out for validation of the deep models.

## 3. Methods
1. **Handcrafted + SVM** — intensity histogram + HOG + uniform LBP
   (8190-d) classified with an RBF-SVM.
2. **ResNet18 (CNN)** — ImageNet-pretrained, two-phase fine-tuning (head, then
   last two residual blocks), 128×128 input.
3. **DeiT-Small (ViT)** — ImageNet-pretrained Vision Transformer, two-phase
   fine-tuning (head warm-up, then full-network fine-tuning), 224×224 input.
   ViTs are data-hungry and transfer poorly with a frozen backbone, so unlike
   the CNN they require deeper fine-tuning to adapt to the SAR domain.

## 4. Results
| Metric | Handcrafted+SVM | ResNet18 | DeiT-Small |
|---|---|---|---|
| Accuracy | 85.98% | 92.49% | 95.79% |
| Macro-F1 | 84.64% | 91.52% | 95.03% |
| Macro-Precision | 86.17% | 91.64% | 95.15% |
| Macro-Recall | 84.46% | 91.51% | 95.01% |
| Training time | 1m 36s | 2m 36s | 6m 40s |
| Inference (ms/img) | 50.03 | 6.41 | 10.18 |
| Model size (MB) | 172.22 | 42.73 | 82.72 |
| Parameters | 2,747 SVs | 11.2M | 21.7M |
| Feature dim | 8190 | 512 | 384 |

## 5. Key Findings
- **Where the traditional method breaks.** The handcrafted+SVM pipeline is
  weakest on the most visually similar vehicles, e.g. BMP2 (SVM F1=0.68 → CNN 0.82 → ViT 0.85), BTR60 (SVM F1=0.70 → CNN 0.79 → ViT 0.87), BTR70 (SVM F1=0.74 → CNN 0.84 → ViT 0.89). HOG/LBP encode
  local edge/texture statistics that cannot separate vehicles whose SAR
  signatures differ only in subtle structural detail.
- **What the CNN adds.** ResNet18 lifts accuracy from
  86.0% to 92.5%
  (+6.5 pts) and macro-F1 from
  84.6% to 91.5%. Learned hierarchical
  features are far more discriminative than fixed descriptors, and the t-SNE plots
  show much cleaner class clusters than raw pixels.
- **The transformer's extra edge.** DeiT-Small reaches
  95.8% accuracy / 95.0% macro-F1
  (+3.3 pts vs ResNet18).
  Global self-attention captures long-range spatial relationships across the target.
- **Trade-off (accuracy vs compute).** Higher accuracy costs compute: DeiT-Small
  is the largest (21.7M) and slowest (10.18 ms/img)
  model, ResNet18 is a strong middle ground (6.41 ms/img),
  and the SVM is cheapest to store but bottlenecked by handcrafted-feature extraction
  at inference.

## 6. Conclusion
On MSTAR SOC, moving from handcrafted features to deep learning yields a large,
consistent accuracy gain, and the Vision Transformer matches or exceeds the CNN
while costing the most compute. For SAR ATR, learned representations clearly
dominate fixed descriptors; the CNN is the best accuracy/efficiency compromise,
whereas the ViT is preferable when maximum accuracy justifies the extra cost.
