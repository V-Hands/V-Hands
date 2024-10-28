# Touchscreen-based Hand Tracking for Remote Whiteboard Interaction

[Paper](https://arxiv.org/pdf/2409.13347) | [Video](https://www.youtube.com/watch?v=YOPJSntBNy0) | [Model](https://drive.google.com/file/d/1vTf2Bgm0zXFxvSCNduzpAk38kw6joG0Y/view?usp=sharing) | [Dataset](https://drive.google.com/file/d/1iAvMGkJdykyBf84KHuXFY4GxDUjUMdZn/view?usp=sharing) | [Homepage](https://v-hands.github.io/) 

<p align="middle">
    <img src=assets/overview.jpg>
</p>

Official implementation of the UIST 2024 paper [Touchscreen-based Hand Tracking for Remote Whiteboard Interaction](https://arxiv.org/pdf/2409.13347). 

## Installation
For inverse kinematics, please download the [MANO parameters](https://mano.is.tue.mpg.de/) and put them as:

    ├── ./                # current (project) directory
    ├── SkeletonModel/
    │   ├── ...
    │   └── mano_v1_2/           
    │       └── models/
    │           ├── MANO_LEFT.pkl   # The file required
    │           └── MANO_RIGHT.pkl  # The file required
    ├── ...

## Dataset

<p align="middle">
    <img src=assets/dataset.jpg>
</p>

The dataset can be accessed [here](https://drive.google.com/file/d/1iAvMGkJdykyBf84KHuXFY4GxDUjUMdZn/view?usp=sharing).

    ├── ./ # current (project) directory
    ├── train/
    │   ├── 20240124125843 
    │   │   ├── clip1
    │   │   │   ├── joints.json
    │   │   │   └── frames/           
    │   │   │       ├── 0.png
    │   │   │       ├── ...
    │   │   ├── ...
    │   ├── ...
    │
    └── test/

## Model Checkpoint

Please download our model checkpoint [here](https://drive.google.com/file/d/1vTf2Bgm0zXFxvSCNduzpAk38kw6joG0Y/view?usp=sharing).

## Evaluation
### Configuration
Please modify the data path in the `config.yaml` file.

### Train
    
```sh
python train.py
```

### Predict Joints
    
```sh
python predict_joints.py
```

### Inverse Kinematics 
    
```sh
python inverse_kinematics.py
```

### Evaluate the Results

```sh
python eval_results.py
```

## Results
<p align="middle">
    <img src=assets/results.jpg>
</p>

## Citation
If you find our code or paper helps, please consider citing:
```
@inproceedings{10.1145/3654777.3676412,
    author = {Liu, Xinshuang and Zhang, Yizhong and Tong, Xin},
    title = {Touchscreen-based Hand Tracking for Remote Whiteboard Interaction},
    year = {2024},
    isbn = {9798400706288},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3654777.3676412},
    doi = {10.1145/3654777.3676412},
    booktitle = {Proceedings of the 37th Annual ACM Symposium on User Interface Software and Technology},
    articleno = {71},
    numpages = {14},
    keywords = {collaboration, input techniques, touch},
    location = {Pittsburgh, PA, USA},
    series = {UIST '24}
}
```