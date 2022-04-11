# Deep learning data augmentation for Raman spectroscopy cancer tissue classification

This repository contains the author's implementation in Python for the scientific reports "Deep learning data augmentation for Raman spectroscopy cancer tissue classifcation" (https://www.nature.com/articles/s41598-021-02687-0). Since the dataset is private, we will not make it public.

## Abstract
Recently, Raman Spectroscopy (RS) was demonstrated to be a non-destructive way of cancer diagnosis, due to the uniqueness of RS measurements in revealing molecular biochemical changes between cancerous vs. normal tissues and cells. In order to design computational approaches for cancer detection, the quality and quantity of tissue samples for RS are important for accurate prediction. In reality, however, obtaining skin cancer samples is difficult and expensive due to privacy and other constraints. With a small number of samples, the training of the classifier is difficult, and often results in overfitting. Therefore, it is important to have more samples to better train classifiers for accurate cancer tissue classification. To overcome these limitations, this paper presents a novel generative adversarial network based skin cancer tissue classification framework. Specifically, we design a data augmentation module that employs a Generative Adversarial Network (GAN) to generate synthetic RS data resembling the training data classes. The original tissue samples and the generated data are concatenated to train classification modules. Experiments on real-world RS data demonstrate that (1) data augmentation can help improve skin cancer tissue classification accuracy, and (2) generative adversarial network can be used to generate reliable synthetic Raman spectroscopic data.

## Dependencies
1. pytorch
2. tensorflow
3. sklearn
4. numpy

## Struture

| Floder   | File                     | Description                                |
| -------- | --------------------------- | ------------------------------------ |
| aug      | config.py                   | Parameters                   |
|          | discriminator.py            | Discriminator model                           |
|          | generateData.py             | Generate new data                         |
|          | generator.py                | Generator model                          |
|          | data_loader.py              | Data loader                 |
|          | netTG.pt                    | Generator model after data augmentation      |
|          | train.py                    | Data augmentation training                    |
|          | visualize.py                | Data visualization  |
|          | aug_data.npz                | Data after data augmentation                         |
| medicine | data_processing_medicine.py | Data process       |
|          | medicine_leaveOne.py        | DL_DA model with original dataset  |
|          | medicine_leaveOne_GAN.py    | DL_DA model with both original and augmented data |

## Data augmentation

All files related to data augmentation are in aug folder. Run train.py to train generator and discriminator.

```python
python train.py
```

Save trained gerenator model as .pt file.

Then run generateData.py to create new data.

~~~
python generateData.py
~~~

## Train

Use medicine_leaveOne.py to train original data

```
python medicine_leaveOne.py
```

Use medicine_leaveOne_GAN.py train original and augmented data.

~~~
python medicine_leaveOne_GAN.py
~~~

## Citation

```
@article{wu2021deep,
  title={Deep learning data augmentation for Raman spectroscopy cancer tissue classification},
  author={Wu, Man and Wang, Shuwen and Pan, Shirui and Terentis, Andrew C and Strasswimmer, John and Zhu, Xingquan},
  journal={Scientific reports},
  volume={11},
  number={1},
  pages={1--13},
  year={2021},
  publisher={Nature Publishing Group}
}
```
