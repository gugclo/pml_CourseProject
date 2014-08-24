setwd("/Users/GaryLo/Desktop/Data Science Specialization/Practical Machine Learning/Course Project/Code")

library(caret)
library(ggplot2)
library(psych)
library(knitr)
library(randomForest)
library(foreach)
library(doMC)
registerDoMC(cores = 4)
library(pROC)

#download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",destfile="pml-training.csv",method="curl")
#download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",destfile="pml-testing.csv",method="curl")

trainData = read.csv("pml-training.csv")
testData = read.csv("pml-testing.csv")

# Exploratory Analysis of four variables for user_name == Charles.
# adelmo, pedro, charles
gg1 = ggplot(data=trainData[trainData$user_name=="charles",],aes(seq_along(roll_belt),roll_belt)) + geom_point(aes(colour=trainData[trainData$user_name=="charles",]$classe)) + labs(colour="Classe",title="Roll Belt Measurement against Index",y="Roll Belt Measurement",x="Index")
gg2 = ggplot(data=trainData[trainData$user_name=="charles",],aes(seq_along(yaw_belt),yaw_belt)) + geom_point(aes(colour=trainData[trainData$user_name=="charles",]$classe)) + labs(colour="Classe",title="Yaw Belt Measurement against Index",y="Yaw Belt Measurement",x="Index")
gg3 = ggplot(data=trainData[trainData$user_name=="charles",],aes(seq_along(pitch_belt),pitch_belt)) + geom_point(aes(colour=trainData[trainData$user_name=="charles",]$classe)) + labs(colour="Classe",title="Pitch Belt Measurement against Index",y="Pitch Belt Measurement",x="Index")
gg4 = ggplot(data=trainData[trainData$user_name=="charles",],aes(seq_along(pitch_forearm),pitch_forearm)) + geom_point(aes(colour=trainData[trainData$user_name=="charles",]$classe)) + labs(colour="Classe",title="Pitch Forearm Measurement against Index",y="Pitch Forearm Measurement",x="Index")
multiplot(gg1,gg2,gg3,gg4,cols=2)

# Exploratory Analysis of four variables for user_name == carlitos
# carlitos, jeremy, eurico
gg5 = ggplot(data=trainData[trainData$user_name=="carlitos",],aes(seq_along(roll_belt),roll_belt)) + geom_point(aes(colour=trainData[trainData$user_name=="carlitos",]$classe)) + labs(colour="Classe",title="Roll Belt Measurement against Index",y="Roll Belt Measurement",x="Index")
gg6 = ggplot(data=trainData[trainData$user_name=="carlitos",],aes(seq_along(yaw_belt),yaw_belt)) + geom_point(aes(colour=trainData[trainData$user_name=="carlitos",]$classe)) + labs(colour="Classe",title="Yaw Belt Measurement against Index",y="Yaw Belt Measurement",x="Index")
gg7 = ggplot(data=trainData[trainData$user_name=="carlitos",],aes(seq_along(pitch_belt),pitch_belt)) + geom_point(aes(colour=trainData[trainData$user_name=="carlitos",]$classe)) + labs(colour="Classe",title="Pitch Belt Measurement against Index",y="Pitch Belt Measurement",x="Index")
gg8 = ggplot(data=trainData[trainData$user_name=="carlitos",],aes(seq_along(pitch_forearm),pitch_forearm)) + geom_point(aes(colour=trainData[trainData$user_name=="carlitos",]$classe)) + labs(colour="Classe",title="Pitch Forearm Measurement against Index",y="Pitch Forearm Measurement",x="Index")
multiplot(gg5,gg6,gg7,gg8,cols=2)

# Exploratory Analysis of four variables for all user_name
gg9 = ggplot(data=trainData,aes(seq_along(roll_belt),roll_belt)) + geom_point(aes(colour=trainData$classe)) + labs(colour="Classe",title="Roll Belt Measurement against Index",y="Roll Belt Measurement",x="Index")
gg10 = ggplot(data=trainData,aes(seq_along(yaw_belt),yaw_belt)) + geom_point(aes(colour=trainData$classe)) + labs(colour="Classe",title="Yaw Belt Measurement against Index",y="Yaw Belt Measurement",x="Index")
gg11 = ggplot(data=trainData,aes(seq_along(pitch_belt),pitch_belt)) + geom_point(aes(colour=trainData$classe)) + labs(colour="Classe",title="Pitch Belt Measurement against Index",y="Pitch Belt Measurement",x="Index")
gg12 = ggplot(data=trainData,aes(seq_along(pitch_forearm),pitch_forearm)) + geom_point(aes(colour=trainData$classe)) + labs(colour="Classe",title="Pitch Forearm Measurement against Index",y="Pitch Forearm Measurement",x="Index")
multiplot(gg9,gg10,gg11,gg12,cols=2)

# Removing timestamp variables (non measurement related)
# While certain exercises/measurements may have been done on specific days, there's
# no reason to believe that those same measurements will be conducted on those days
# for future predictions and datasets.
newTrainData = trainData[,-c(1:7)]
newTestData = testData[,-c(1:7)]

# Removing features heavily populated with NA
descTrainData = describe(newTrainData)
invalidFeatures = descTrainData[descTrainData$n<=500,][,1]
newTrainData = newTrainData[,-invalidFeatures]
newTestData = newTestData[,-invalidFeatures]
dim(newTrainData)
dim(newTestData)

# Removing features with near-zero-variance
nearZeroVarFeatures = nearZeroVar(newTrainData)
newTrainData = newTrainData[,-nearZeroVarFeatures]
newTestData = newTestData[,-nearZeroVarFeatures]
dim(newTrainData)
dim(newTestData)

# Split the training data into a training and validation subset.
set.seed(1234)
trainingIndex = createDataPartition(newTrainData$classe, p = 0.80,list=FALSE)
training = newTrainData[trainingIndex,]
testing = newTrainData[-trainingIndex,]

# First, fit a simple decision tree
set.seed(1234)
modelFit1 = train(classe~., method = "rpart", data = training)
modelFit1
confusionMatrix(training$classe,predict(modelFit1,newdata=training))
confusionMatrix(testing$classe,predict(modelFit1,newdata=testing))
predict(modelFit1,newdata=newTestData)

# Second, fit a decision tree with 10-fold CV
set.seed(1234)
control = trainControl(method = "cv", number = 10)
modelFit2 = train(classe~., method = "rpart", data = training, trControl = control)
modelFit2
confusionMatrix(training$classe,predict(modelFit2,newdata=training))
confusionMatrix(testing$classe,predict(modelFit2,newdata=testing))
predict(modelFit2,newdata=newTestData)

# Third, fit a parallel random forest.
set.seed(1234)
modelFit3 = train(classe~., method = "parRF", data = training, tuneGrid = data.frame(mtry=c(2,4,8,16,32)))
modelFit3
confusionMatrix(training$classe,predict(modelFit3,newdata=training))
confusionMatrix(testing$classe,predict(modelFit3,newdata=testing))
#predict(modelFit3,newdata=newTestData)

plot(varImp(modelFit3))
plot(modelFit3)

## Course Project Submission Scripts
answers = as.character(predict(modelFit3,newdata=newTestData))
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}
pml_write_files(answers)


# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
    require(grid)
    
    # Make a list from the ... arguments and plotlist
    plots <- c(list(...), plotlist)
    
    numPlots = length(plots)
    
    # If layout is NULL, then use 'cols' to determine layout
    if (is.null(layout)) {
        # Make the panel
        # ncol: Number of columns of plots
        # nrow: Number of rows needed, calculated from # of cols
        layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                         ncol = cols, nrow = ceiling(numPlots/cols))
    }
    
    if (numPlots==1) {
        print(plots[[1]])
        
    } else {
        # Set up the page
        grid.newpage()
        pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
        
        # Make each plot, in the correct location
        for (i in 1:numPlots) {
            # Get the i,j matrix positions of the regions that contain this subplot
            matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
            
            print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                            layout.pos.col = matchidx$col))
        }
    }
}