# DL_DA



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

