

```
## Warning: package 'psych' was built under R version 3.1.1
## Warning: package 'randomForest' was built under R version 3.1.1
```





##Practical Machine Learning Course Project Writeup


###Executive Summary
In this analysis we attempt to build a machine learning model capable of predicting which of 5 possible ways a person was performing a dumbell curl exercise.  Accelerometers were placed on the belt, forearm, arm and dumbell of 6 participants and 160 measurements were taken.

A random forest was tuned to the data and was able to generate predictions with an out-of-bag error rate of 99.3%.  This was also confirmed against a validation set.  This model was then used to generate 20 predictions for the Submission portion of the Practical Machine Learning Course Project with 100% accuracy.




###Exploratory Data Analysis
The training data set contains 19,622 observations of 160 measurements.  The measurements are based on six different participants in the study. Initial investigation of the data by building plots of the "roll_belt" variable for: (a) all participants, (b) user "carlitos" and, (c) user "charles".

![plot of chunk unnamed-chunk-4](./pml_Project_files/figure-html/unnamed-chunk-4.png) 

The first plot on the data for all participants suggests that measurements may not have been taken in a consistent manner.  See the two groupings of measurements that are offset to eachother (vertically shifted).

The plots for "carlitos" suggest that the Roll Belt measurement may be highly predictive of classe "E".

Roll Belt appears to be able to uniquely distinguish classe "E" for charles, but in a different way than carlitos.  Also, we can see that the measurements for Roll Belt are very different when comparing charles and carlitos.  See below for means and standard deviations.


```
## [1] "Roll Belt Mean for Charles =  122.027714932127"
```

```
## [1] "Roll Belt Mean for Carlitos =  1.16774100257069"
```

```
## [1] "Roll Belt Standard Deviation for Charles =  10.6845111749933"
```

```
## [1] "Roll Belt Standard Deviation for Carlitos =  2.86174100190023"
```

We expect that this may cause model-fitting issues as measurements will be correlated with user_name and we do not want to build a model that relies on the username as a parameter (a model to predict measurements on the exact same "charles" will have limited value).  Hopefully, other measurements working in tandem with eachother will allow us to fit a model accurately.


###Data Cleaning
Both the training data and test data supplied at the course website contain 160 columns of information.


```
## [1] 19622   160
```

```
## [1]  20 160
```

We will strip out the first seven columns of the data (X, user name, raw timestamp part 1, raw timestamp part 2, cvtd timestamp, new window, num window) as these do not appear to be accelerometer measurements.  These seven columns appear to be metadata inputs and a working model should only take accelerometer measurements to predict classe on future unseen datasets.  This will reduce the number of columns in our dataframe to 153.



In the next step, we will use the describe() function from the "psych" package to remove observations that are dominated by "NA".  These are sparse features and are unlikely to be predictive.  This will reduce the number of columns in our dataframe to 86.



In the final data cleaning step, we will apply the nearZeroVar() function from the "caret" package to remove features that have a near zero variance as, again, these are unlikely to be predictive.  This will reduce the number of columns in our final dataframe to 53.


```
## [1] 19622    53
```

```
## [1] 20 53
```


###Model Selection
We will split the training data into a "training" set and a "validation" set.  The training subset will include 80% of the observations and the remaining will be used to calculate the out-of-sample error.



Initially, we will fit a decision tree to gain some intuition in the structure of the data.  While easy to interpret, decision trees are known to be fairly poor predictors and we shall instead fit a random forest (an ensemble of decision treees).

Below we have fit a decision tree using the caret package.  While we are already conducting cross-validation by splitting the training data into a training/validation subsets, the caret package allows us to easily conduct 10-fold cross validation in the model training call.


```
## CART 
## 
## 15699 samples
##    52 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 14129, 14130, 14131, 14129, 14127, 14129, ... 
## 
## Resampling results across tuning parameters:
## 
##   cp    Accuracy  Kappa  Accuracy SD  Kappa SD
##   0.04  0.5       0.4    0.02         0.03    
##   0.06  0.4       0.2    0.07         0.1     
##   0.1   0.3       0.06   0.04         0.06    
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.04.
```

Despite a 10-fold cross validation process, the decision tree accurately predicts the training data only 50.9% of the time.  While this is better than a purely random selection of the "classe" response, it's likely that we can find a model that predicts better.

![plot of chunk unnamed-chunk-12](./pml_Project_files/figure-html/unnamed-chunk-12.png) 

Interestingly, the initial node of the fitted decision tree splits on the roll belt feature to predict the E classe.  We saw this relationship in our exploratory analysis. Also note that the D classe cannot be found at any of the terminal nodes so the fitted decision tree will never predict a D classe.  Hopefully we can find a leaner that can do better than that!


```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1025   20   69    0    2
##          B  310  271  178    0    0
##          C  325   23  336    0    0
##          D  290  102  251    0    0
##          E  115   95  184    0  327
## 
## Overall Statistics
##                                         
##                Accuracy : 0.499         
##                  95% CI : (0.484, 0.515)
##     No Information Rate : 0.526         
##     P-Value [Acc > NIR] : 1             
##                                         
##                   Kappa : 0.345         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.496   0.5303   0.3301       NA   0.9939
## Specificity             0.951   0.8570   0.8802    0.836   0.8904
## Pos Pred Value          0.918   0.3570   0.4912       NA   0.4535
## Neg Pred Value          0.629   0.9241   0.7894       NA   0.9994
## Prevalence              0.526   0.1303   0.2595    0.000   0.0839
## Detection Rate          0.261   0.0691   0.0856    0.000   0.0834
## Detection Prevalence    0.284   0.1935   0.1744    0.164   0.1838
## Balanced Accuracy       0.724   0.6937   0.6051       NA   0.9421
```

Above is a confusion matrix comparing the true responses in the 20% validation set against the predicted reposnses from the validation set.  As you can see the accuracy is approximately 49.9%.  The accuracy is lower on the validation set than the training set due to slight overfitting of the training set by the classifier.

Below we have fit a random forest using parallel cores to reduce computation time.  We have tuned the random forest by setting the "mtry" variable to either 2, 4, 8, 16 or 32.  The "mtry" variable is the number of features that are randomly chosen at each node for each bootstrapped decision tree.  If mtry is set to 4, eacho node for each decision tree that is fit to a bootstrapped sample will have be split on one of 4 randomly selected features.


```
## Parallel Random Forest 
## 
## 15699 samples
##    52 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 15699, 15699, 15699, 15699, 15699, 15699, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      0.002        0.002   
##   4     1         1      0.001        0.002   
##   8     1         1      0.002        0.002   
##   20    1         1      0.001        0.002   
##   30    1         1      0.002        0.002   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 8.
```

Accuracy is nearly 100% for each of the tuning parameters but mtry = 8 was the optimal value (accuracy of 99.3%).  This accuracy is an out-of-bag accuracy rate and is the accuracy rate one would expect to achieve when applying the model against an unseen test set.  

![plot of chunk unnamed-chunk-15](./pml_Project_files/figure-html/unnamed-chunk-15.png) 

A plot showing the maximum accuracy for the random forest model at mtry = 8 is shown above.


```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    0    0    0    0
##          B    3  755    1    0    0
##          C    0    5  679    0    0
##          D    0    0    1  642    0
##          E    0    0    0    0  721
## 
## Overall Statistics
##                                         
##                Accuracy : 0.997         
##                  95% CI : (0.995, 0.999)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.997         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.997    0.993    0.997    1.000    1.000
## Specificity             1.000    0.999    0.998    1.000    1.000
## Pos Pred Value          1.000    0.995    0.993    0.998    1.000
## Neg Pred Value          0.999    0.998    0.999    1.000    1.000
## Prevalence              0.285    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.192    0.173    0.164    0.184
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.999    0.996    0.998    1.000    1.000
```

The accuracy generated by the random forest model call (99.3%) is an out-of-bag error rate so technically an additional cross validation isn't necessary (http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#ooberr).  That said, we had already split the training set into a training/validation subset for our decision tree so to remain consistent we've used the same split for the random forest.  A confusion matrix for the validation set is shown above.  The accuracy for predictions against the validation set are approximately 99.7%.

Below we have made a function call to varImp() in the caret package.  This shows the importance of each variable.  Roll Belt and Yaw Belt are the most important variables in the dataset.

![plot of chunk unnamed-chunk-17](./pml_Project_files/figure-html/unnamed-chunk-17.png) 


###Conclusion
The random forest learner is well suited to generating accurate predictions for this specific dataset.  While a single stratification of the feature set (decision tree) wasn't able to accurately predict the response, a bootstrapped ensemble of decision trees were.

Applying the random forest model to predict the 20 test cases from the Submission portion of the Course Project resulted in a 100% classification rate.  
