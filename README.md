# Usage

1. Clone the repository.
```
git clone https://github.com/LeryLee/vein_segmentation
```
2. Download the datasets used in our paper from [here](https://docs.google.com/forms/d/e/1FAIpQLSflrJTabnsFd7KjLpu4yBJkKg2yimDdjrYU3Hmd_gJiKstXxQ/viewform). The datasets used in our paper are newly collected. Please cite our paper if you use it for your research.

- Organize the file structure as below.
```
|__ vein_segmentation
    |__ code
    |__ data
        |__ LVD2021
```

3. Run pretrain.py and self_train.py to train and test the code.

# Acknowledgement
[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

[PointRend](https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend)

[CE-Net](https://github.com/Guzaiwang/CE-Net)
