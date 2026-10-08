# Plan: OCT Segmentation (Maksymilian - Student 2)

## Status: Stage III (Final Results & Interpretation)

### Stage II (Completed)
- [x] Shared data pipeline (RETOUCH 56/7/7 split).
- [x] Baseline SegFormer-B0 implementation.
- [x] Multi-modal & 2.5D context experimentation.

### Stage III: Transformer Optimization & Thesis Reporting (Active)
**Goal:** Finalize "Golden Model" results and generate interpretability data (Attention Maps).

#### 1. Model Experiments & Stability
- [x] **SegFormer-B2 Strategy:** Proven optimal backbone (Test 11/14).
- [x] **Test 16 (The Fusion Record):** Broken project record with **0.7621 mIoU** (at epoch 37). Stable Focal-Tversky + Argmax configuration.
- [ ] **Test 17 (B3 Scale-up):** Final attempt to scale to MiT-B3 using the Test 16 "Stability Fusion" pipeline. **Configuration:** `Dropout=0.3` (overfitting protection), `MIN_REGION=5` (detail preservation), `LR=5e-5`. Status: **ACTIVE**.

#### 2. Interpretability & Analysis
- [x] **Attention Maps:** Integrated automated extraction of Transformer attention heatmaps to identify diagnostic focus areas (RPE, FLuid-Tissue interface).
- [x] **Failure Analysis:** Automated extraction of top 5 worst-performing slices for qualitative discussion in Chapter 5.

#### 3. Final Benchmarking
- [ ] Finalize Comparison Table: **nnUNet (Eryk) vs SegFormer-B2 vs SwinUNETR**.
- [x] HD95 vs Dice trade-off analysis (focusing on clinical boundary precision).

#### 4. Thesis Partial Report (Deadline: 15.06.2025)
- [ ] Description of SegFormer (Self-Attention, MLP Head).
- [ ] Detailed methodology of 2.5D stacking and Percentile Normalization.
- [ ] Quantitative results on RETOUCH.

---

## Log / Diary

### 03.06.2026: Test 16 - The Finał Stability Run
- **Objective:** Consolidate all winning strategies from Tests 11, 12, and 14.
- **Decision:** Critical removal of Boundary Loss. While academically sound, it increased HD95 by 3.0 points on our dataset. Prioritizing stability for the final run.

### 29.05.2025: SwinUNETR Review
- **Conclusion:** Powerful but overfitting (Test 10). MiT-B2 remains the most reliable backbone for clinical deployment in OCTAnnotate.
