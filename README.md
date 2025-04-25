# Files

**1. imagesplit.ipynb:** For splitting panoramic images into left/right halves.

**2. partitioning.ipynb:** Loads CSV or NDJSON label file to a pandas dataframe, creates groups for left/right pairs, performs stratified sampling to create train/validation/test splits ensuring the proportion of each class in each split is the same as the in the original dataset, finally plots class distribution for each split for visualization and comparison.

**3. datapartition.py:** A DataPartition class that must sit separately as a script and imported to the following Jupyter Notebooks for GPU training to work:

**4. full_pipeline.ipynb:** Full end-to-end code for finetuning a pretrained model (including the partitioning from partitioning.ipynb minus the visualizations). Current version comes with and is tested for SwinV2 Transformer model. See "Resources" for choosing other models.

**5. csra.ipynb:** full_pipeline for finetuning a pretrained model + class-specific residual attention (CSRA)
   - [ (Paper) ](https://arxiv.org/pdf/2108.02456v2 )
   - [ (Official Code) ](https://github.com/Kevinz-code/CSRA/blob/master/pipeline/csra.py)

**6. mldecoder.ipynb:** full_pipeline for finetuning a pretrained model + ML Decoder head
   - [ (Paper) ](https://arxiv.org/pdf/2111.12933v2)
   - [ (Official Code) ](https://github.com/Alibaba-MIIL/ML_Decoder/blob/main/src_files/ml_decoder/ml_decoder.py)

### Miscellaneous Files
- **demo_360.ndjson:** A dummy Label file mirroring the structure of how our LabelBox labels exports. Used to inspect structure and create appropriate code while images were still being labeled. Notes on the structure can be found [here](https://docs.google.com/document/d/1F8RFUZPHlVVkZzpF3bIoOaOA62DqiIFhOYRDARhMce8/edit?usp=sharing).

# Resources
- [Implementation guide for finetuning a pretrained model](https://www.kaggle.com/code/gohweizheng/swin-transformer-beginner-friendly-notebook#1.-Introduction) (specifically Swin Transformer)

- [Pretrained Models to choose from timm](https://huggingface.co/models?library=timm)
  - [Benchmarks for choosing a pretrained Swin Transformer model](https://github.com/microsoft/Swin-Transformer/blob/main/MODELHUB.md#imagenet-22k-pretrained-swin-moe-models)
    (*benchmarks are for regular image classification on ImageNet dataset)
  - [Benchmarks for choosing/comparing other pretrained models](https://paperswithcode.com/sota/image-classification-on-imagenet)
    (*benchmarks are for regular image classification on ImageNet dataset)

- Optimal Prediction Thresholding Strategies:
  - https://www.evidentlyai.com/classification-metrics/classification-threshold
  - https://www.mathworks.com/help/deeplearning/ug/multilabel-image-classification-using-deep-learning.html
  - https://www.mdpi.com/2076-3417/13/13/7591
    
