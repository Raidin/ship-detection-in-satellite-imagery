# Ship Detection In Satellite Imagery

## ENVIRONMENTS
 * Ubuntu 16.04 LTS
 * TensorFlow 1.12.0
 * Keras 2.2.4
 * Python 2.7
 * CUDA 8.0
 * cuDNN 6.0

## PYTHON LIBRARY
 * image==1.5.27
 * matplotlib==2.2.3
 * numpy==1.15.4
 * opencv-python==4.0.0.21
 * Pillow==5.4.1
 * scikit-learn==0.20.2
 * tqdm==4.29.0

## DATASET
 * Kaggle-Ship-Detection :: [Ships in satellite Imagery](https://www.kaggle.com/rhammell/ships-in-satellite-imagery/home)

## DIRECTORY STRUCTURE
```
|--- [Project Directory]
|--- [data]
|--- [exmaples]
        |--- [ImageDataGenerator]
|--- [script]
|--- [src]
|--- [work]
        |--- [Job Name(as Network Model Name)]
                |--- [model]
                |--- [output]
```

## SCRIPT DETAILS
 * generation_model.py :: Generating Network Model
 * get_data_info.py :: ship detection dataset [json] file read and information display
 * test.py :: shipe detection test script
 * train.py :: shipe detection train script