1. imagesplit.ipynb: For splitting panoramic images into left/right halves.

2. partitioning.ipynb: Loads CSV or NDJSON file to a pandas dataframe, creates groups for left/right pairs, performs stratified sampling to create train/validation/test splits to ensure the proportion of each class in each split is the same as the in the original dataset.

3. full_pipeline.ipynb: Full end-to-end code for finetuning a pretrained model (including the partitioning from partitioning.ipynb minus the visualizations).

4. csra.ipynb: full_pipeline for finetuning a pretrained model + class-specific residual attention (CSRA)
   - Paper: https://arxiv.org/pdf/2108.02456v2 

5. mldecoder.ipynb: full_pipeline for finetuning a pretrained model + ML Decoder head
   - Paper: https://arxiv.org/pdf/2111.12933v2
