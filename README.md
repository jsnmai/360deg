# Files

**1. imagesplit.ipynb:** For splitting panoramic images into left/right halves.

**2. partitioning.ipynb:** Loads CSV or NDJSON file to a pandas dataframe, creates groups for left/right pairs, performs stratified sampling to create train/validation/test splits to ensure the proportion of each class in each split is the same as the in the original dataset.

**3. datapartition.py:** A DataPartition class that must sit separately as a script and imported to the following Jupyter Notebooks for GPU training to work:

**4. full_pipeline.ipynb:** Full end-to-end code for finetuning a pretrained model (including the partitioning from partitioning.ipynb minus the visualizations).

**5. csra.ipynb:** full_pipeline for finetuning a pretrained model + class-specific residual attention (CSRA)
   - [ (Paper) ](https://arxiv.org/pdf/2108.02456v2 )
   - [ (Official Code) ](https://github.com/Kevinz-code/CSRA/blob/master/pipeline/csra.py)

**6. mldecoder.ipynb:** full_pipeline for finetuning a pretrained model + ML Decoder head
   - [ (Paper) ](https://arxiv.org/pdf/2111.12933v2)
   - [ (Official Code) ](https://github.com/Alibaba-MIIL/ML_Decoder/blob/main/src_files/ml_decoder/ml_decoder.py)

# Resources
- [Implementation guide for finetuning a pretrained model](https://www.kaggle.com/code/gohweizheng/swin-transformer-beginner-friendly-notebook#1.-Introduction) (specifically Swin Transformer)

- [Pretrained Models to choose from timm](https://huggingface.co/models?library=timm)
  - [Benchmarks for choosing a pretrained Swin Transformer model](https://github.com/microsoft/Swin-Transformer/blob/main/MODELHUB.md#imagenet-22k-pretrained-swin-moe-models)
