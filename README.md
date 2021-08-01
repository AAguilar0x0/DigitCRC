# DigitCRC (Digit Character Recognition Canvas)

DigitCRC is an application that allows the user to write anything on a canvas. 
The application then tries to predict what is written on the canvas. 
However, the application does not predict multiple or non-digit character written on the canvas. 
As it is designed to only predict a single digit character written on the canvas at a time.

<br/><br/>

## Technical Details

The application has three models predicting handwriting independently. The first model was based on Logistic Regression, the second one is based on Neural Network, while the third one is based on SVM. 
Since both the first and the third are not inherently multiclass classification algorithms. A one-vs-all approach is used for it to be able to predict multiple classes.  

To collect or create a personal dataset is very laborious and time-consuming. And so, the models were instead trained with MNIST dataset consisting of 60,000 data samples. This meant that predicting the target (the content of the canvas) using those models would not work. To make it work as much as possible the target needs to be preprocessed. This also meant that the result of the prediction of those models is not guaranteed.  

Because of hardware limitations, certain decisions were made. One of which is to limit the training of the models, the first and the second model were trained only with one epoch of the dataset. While the third one was trained only one-sixth of the dataset. As such, the pursuit of including CNN was discontinued.

<br/><br/>

## Cost History

### Neural Network
!["Neural Network"](<resources/Neural Network Cost History.png>)

### Logistic Regression
!["Logistic Regression"](<resources/Logistic Regression Cost History.png>)

<br/><br/>

## Confusion Matrix

### Neural Network
```
[[ 963    0    0    2    2    2    8    2    1    0]
 [   0 1115    1    6    1    1    3    1    7    0]
 [  11    4  916   35   14    1   15    9   24    3]
 [   2    0   12  951    0   22    1    5    8    9]
 [   2    0    2    1  924    0    8    0    3   42]
 [   9    2    0   24   10  805   11    3   16   12]
 [  10    4    2    1    4   18  914    0    5    0]
 [   5   13   26    8    7    3    0  910    2   54]
 [   5   13    4   37   15   22   10    4  847   17]
 [   8    4    1   14   15    6    0    6    3  952]]
```

### Logistic Regression
```
[[ 963    0    1    2    1    1    4    1    4    3]
 [   0 1117    2    2    1    0    4    1    8    0]
 [   6   23  872   35   10    5   18   11   47    5]
 [   4    2   17  919    1   28    3   15   10   11]
 [   1    0    4    2  881    0   11    1   16   66]
 [  13    3    0   39   10  747   20    4   38   18]
 [  10    3    4    0    3   22  907    0    9    0]
 [   2   15   23    7    7    1    0  909    7   57]
 [   8   21    8   32   13   39    7   10  811   25]
 [   8    5    2   14   28    4    1   15    8  924]]
```

### Support Vector Machine
```
[[ 971    1    1    0    0    1    4    1    1    0] 
 [   0 1130    1    2    0    0    2    0    0    0] 
 [  15   14  960    9    1    1    4   23    5    0] 
 [   1    1    4  945    1   23    3   12    9   11] 
 [   0   11    0    0  915    0    8    6    2   40] 
 [   8    4    0   27    3  829   10    2    2    7] 
 [   9    3    0    0    3    4  938    0    1    0] 
 [   0   27    8    1    4    1    0  971    1   15] 
 [   9    5    6   32    4   21    7    8  863   19] 
 [   6    6    3    7   20    4    1   17    4  941]]
```

<br/><br/>

## Classification Report

### Neural Network
```
              precision    recall  f1-score   support

           0       0.95      0.98      0.97       980
           1       0.97      0.98      0.97      1135
           2       0.95      0.89      0.92      1032
           3       0.88      0.94      0.91      1010
           4       0.93      0.94      0.94       982
           5       0.91      0.90      0.91       892
           6       0.94      0.95      0.95       958
           7       0.97      0.89      0.92      1028
           8       0.92      0.87      0.90       974
           9       0.87      0.94      0.91      1009

    accuracy                           0.93     10000
   macro avg       0.93      0.93      0.93     10000
weighted avg       0.93      0.93      0.93     10000
```

### Logistic Regression
```
              precision    recall  f1-score   support

           0       0.95      0.98      0.97       980
           1       0.94      0.98      0.96      1135
           2       0.93      0.84      0.89      1032
           3       0.87      0.91      0.89      1010
           4       0.92      0.90      0.91       982
           5       0.88      0.84      0.86       892
           6       0.93      0.95      0.94       958
           7       0.94      0.88      0.91      1028
           8       0.85      0.83      0.84       974
           9       0.83      0.92      0.87      1009

    accuracy                           0.91     10000
   macro avg       0.91      0.90      0.90     10000
weighted avg       0.91      0.91      0.90     10000
```

### Support Vector Machine
```
              precision    recall  f1-score   support

           0       0.95      0.99      0.97       980
           1       0.94      1.00      0.97      1135
           2       0.98      0.93      0.95      1032
           3       0.92      0.94      0.93      1010
           4       0.96      0.93      0.95       982
           5       0.94      0.93      0.93       892
           6       0.96      0.98      0.97       958
           7       0.93      0.94      0.94      1028
           8       0.97      0.89      0.93       974
           9       0.91      0.93      0.92      1009

    accuracy                           0.95     10000
   macro avg       0.95      0.95      0.95     10000
weighted avg       0.95      0.95      0.95     10000
```