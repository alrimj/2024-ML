# Looking Back `2024 Machine Learning Team Project`

## Topic of this project :  Emotion recognition uising voice recognition
our team is composed with 4 people that include me.
I take a task which is 
1. making KNN based model
2. making Presentation resource
and other team members made (Decision tree, XGBoost, SVM based model).

## Our Goal
We labeled the emotions which include angry, sad, neutral, happy to numbers(each of 0, 2, 4, 5). So its totall 4!
And the dataset for machine learning we used is available at kaggle. (almost 1300..? I remember)

our project aims to get a successful number of Accuracy which is close to 99.9%.
and we had to using only machine learning based model.

@ before I learn specific
so couldn't use deep learning model and also had a problem with data preprocessing and modeling.
I had no idea that time to make reasonably preprocessed data and models.

## Data preprocessing trial
1. librosa.features ( MFCC, spectral_contrast, rms, Spectral Centroid, Zero Crossing Rate, Spectral Bandwidth)
2. Variation sampling rate
3. Data Scaling
4. Adjusting Audio array length by using Padding

our team used all of techniques above.
We extracted lot of audio features, by using these, performed classifying the data.

## Modeling things

### KNN > 0.433 Acc
KNN is Nonparametic and lazy learning algorithm.
It uses the Distance measure and have to decide the hyperparameter 'K' which means the number of neighbors we want to consider.

We could do cross Validation for finding proper K or Normalization if feature's scale is different.
used distance measurement was Manhattan.
K (n-neighbors) is 20.

![image](https://github.com/user-attachments/assets/2dc3620e-93bb-4d90-9c72-b3b711e19b36)

### RF > 0.478 Acc
Tuning by regulatung n_estimators and max_depth.
And its scale was not that important than other models.

model's variable was n_mfcc, n_estimators, max_depth.
n_mfcc : 5 ~ 30
n_estimators : 100 ~ 400
max_depth : 20 ~ 100
Model : RandomForestClassifier

![image](https://github.com/user-attachments/assets/b36b08f4-9c39-4cc0-9af4-40bec97338d9)

![image](https://github.com/user-attachments/assets/0c96211e-3f26-40cb-b402-e72f20c01fc7)

### XGBoost > 0.43 Acc
In first, we got 43 accuracy using xgboost.
Used extra features, After that, that get us much better accuracy.

Extra features was

![image](https://github.com/user-attachments/assets/7ffa47ea-7963-4919-bf49-c83564f2f761)

### SVM > 0.679 ACC
used sklearn.svm.SVC
In scilit-learn, multi-class support was handled according to OvO scheme.

First chosen kernel was linear, but, changed by rbf which made better accuracy.

![image](https://github.com/user-attachments/assets/67f1ddcc-1272-4428-9fc9-fbb741faff89)

## Conclusion
We checked the result by using our own voice dataset.
and the result (highest accuracy) was 0.67.

If included the next things to our project, maybe we could get better results.

1. data augmentation
2. model ensemble
3. NLP
4. applying more professional Speech recognition Technology

## Future Tasks
So, I'll learn Speech recognition Technology, MLP. 

- [ ]  Speech signal processing and analysis
- [ ]  Speech features
- [ ]  Hidden Markov Model(Fundamental concept and evaluation, Decoding and learning)
- [ ]  HMM for ASR
- [ ]  DNN-HMM
- [ ]  E2E-CTC

- [ ]  Words Representation
- [ ]  Sequence Modeling with RNNs
- [ ]  Sequence Modeling with Transformers
- [ ]  Instruction Tuning
- [ ]  Preference Optimization
- [ ]  Build NLP Projects with Hugging Face
- [ ]  Prompting
- [ ]  Parameter Efficient Fine-Tuning
- [ ]  Distillation, Quantization, Pruning
- [ ]  Mixture of Experts, Retrieval-Augmented Generation
