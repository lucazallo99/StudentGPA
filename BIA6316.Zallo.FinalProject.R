#######DATA PREPROCESSING########
DATA <- read.csv("C:/Users/lucaz/Downloads/archive (2)/Student_performance_data _.csv", stringsAsFactors=TRUE)
View(DATA)
DATA$Gender <- factor(DATA$Gender, 
                      levels = c(0, 1), 
                      labels = c("Male", "Female"))

DATA$Ethnicity <- factor(DATA$Ethnicity, 
                         levels = c(0, 1, 2, 3), 
                         labels = c("Caucasian", "African American", "Asian", "Other"))

DATA$ParentalEducation <- factor(DATA$ParentalEducation, 
                                 levels = c(0, 1, 2, 3, 4), 
                                 labels = c("None", "High School", "Some College", "Bachelor's", "Higher"))

DATA$Tutoring <- factor(DATA$Tutoring, 
                        levels = c(0, 1), 
                        labels = c("No", "Yes"))

DATA$ParentalSupport <- factor(DATA$ParentalSupport, 
                               levels = c(0, 1, 2, 3, 4), 
                               labels = c("None", "Low", "Moderate", "High", "Very High"))

DATA$Extracurricular <- factor(DATA$Extracurricular, 
                               levels = c(0, 1), 
                               labels = c("No", "Yes"))

DATA$Sports <- factor(DATA$Sports, 
                      levels = c(0, 1), 
                      labels = c("No", "Yes"))

DATA$Music <- factor(DATA$Music, 
                     levels = c(0, 1), 
                     labels = c("No", "Yes"))

DATA$Volunteering <- factor(DATA$Volunteering, 
                            levels = c(0, 1), 
                            labels = c("No", "Yes"))

DATA$GradeClass <- factor(DATA$GradeClass, 
                          levels = c(0, 1, 2, 3, 4), 
                          labels = c("A", "B", "C", "D", "F"))
head(DATA)
#check for NA values
any(is.na(DATA))

attach(DATA)
str(DATA)
names(DATA)
options(scipen = 999)
dim(DATA)

##########EDA##############
library(psych)
describe(DATA)
library(ggplot2)
library(dplyr)

# Plot frequency distribution of age
ggplot(DATA, aes(x = Age)) + 
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black") + 
  labs(title = "Frequency Distribution of Age", x = "Age", y = "Frequency") + 
  theme_minimal()
#Frequency distribution of Study Time Weekly
ggplot(DATA, aes(x = StudyTimeWeekly)) + 
  geom_histogram(binwidth = 1, fill = "green", color = "black") + 
  labs(title = "Frequency Distribution of Study Time Weekly", x = "Study Time Weekly", y = "Frequency") + 
  theme_minimal()
#Frequency distribution of Absences
ggplot(DATA, aes(x = Absences)) + 
  geom_histogram(binwidth = 1, fill = "red", color = "black") + 
  labs(title = "Frequency Distribution of Absences", x = "Absences", y = "Frequency") + 
  theme_minimal()
#Frequency distribution of GPA
ggplot(DATA, aes(x = GPA)) + 
  geom_histogram(binwidth = 0.1, fill = "yellow", color = "black") + 
  labs(title = "Frequency Distribution of GPA", x = "GPA", y = "Frequency") + 
  theme_minimal()
# Gender
barplot(table(Gender), 
        main = "Distribution of Gender", 
        xlab = "Gender", 
        ylab = "Frequency", 
        col = "green", 
        border = "black")

# Ethnicity
barplot(table(Ethnicity), 
        main = "Distribution of Ethnicity", 
        xlab = "Ethnicity", 
        ylab = "Frequency", 
        col = "orange", 
        border = "black")

# ParentalEducation
barplot(table(ParentalEducation), 
        main = "Distribution of Parental Education", 
        xlab = "Parental Education", 
        ylab = "Frequency", 
        col = "red", 
        border = "black")

# Tutoring
barplot(table(Tutoring), 
        main = "Distribution of Tutoring", 
        xlab = "Tutoring", 
        ylab = "Frequency", 
        col = "grey", 
        border = "black")

# ParentalSupport
barplot(table(ParentalSupport), 
        main = "Distribution of Parental Support", 
        xlab = "Parental Support", 
        ylab = "Frequency", 
        col = "pink", 
        border = "black")

# Extracurricular
barplot(table(Extracurricular), 
        main = "Distribution of Extracurricular Activities", 
        xlab = "Extracurricular Activities", 
        ylab = "Frequency", 
        col = "violet", 
        border = "black")

# Sports
barplot(table(Sports), 
        main = "Distribution of Sports Participation", 
        xlab = "Sports Participation", 
        ylab = "Frequency", 
        col = "purple", 
        border = "black")

# Music
barplot(table(Music), 
        main = "Distribution of Music Participation", 
        xlab = "Music Participation", 
        ylab = "Frequency", 
        col = "blue", 
        border = "black")

# Volunteering
barplot(table(Volunteering), 
        main = "Distribution of Volunteering", 
        xlab = "Volunteering", 
        ylab = "Frequency", 
        col = "green", 
        border = "black")

# GradeClass
barplot(table(GradeClass), 
        main = "Distribution of Grade Class", 
        xlab = "Grade Class", 
        ylab = "Frequency", 
        col = "skyblue", 
        border = "black")

#Correlation Matrix
numerical_data <- DATA[, sapply(DATA, is.numeric)]
# Exclude the first column (ID column)
numerical_data <- numerical_data[, -1]
# Calculate the correlation matrix for numerical variables
correlation_matrix <- cor(numerical_data)
library(corrplot)
corrplot(correlation_matrix, method = "pie", type = "lower", tl.col = "black", tl.srt = 45, tl.cex = 0.7)
#GPA IS HIGHLY NEGATIVELY CORRELATED WITH ABSENCES
#t-testing GPA
#Gender
var.test(GPA ~ Gender, DATA)
t.test(GPA ~ Gender, DATA, var.equal=T)
#Sports
var.test(GPA ~ Sports, DATA)
t.test(GPA ~ Sports, DATA, var.equal=T) #significant
#Volunteering
var.test(GPA ~ Volunteering, DATA)
t.test(GPA ~ Volunteering, DATA, var.equal=T)
#extracurricular
var.test(GPA ~ Extracurricular, DATA)
t.test(GPA ~ Extracurricular, DATA, var.equal=T) #significant
#Tutoring
var.test(GPA ~ Tutoring, DATA)
t.test(GPA ~ Tutoring, DATA, var.equal=T) #significant
#Anova across parental educational level
aovEDU <- aov(GPA~ParentalEducation, DATA)
anova(aovEDU)
coef(aovEDU)
confint(aovEDU)
TukeyHSD(aovEDU)
#Anova across parental support
aovSUP <- aov(GPA~ParentalSupport, DATA)
anova(aovSUP)
coef(aovSUP)
confint(aovSUP)
TukeyHSD(aovSUP) #Significant
#GPA will be the outcome variable, let's assess normality
QQPLOT <- qqnorm(GPA)
qqline(GPA)
library(tseries)
jarque.bera.test(GPA)

#############QUANTILE REGRESSION######
######MEDICAL EXPENDITURE QUANTILES#######

quantile(GPA)
quantile(GPA, seq(0,1, by = 0.10))

#######OLS REG#######
OLS <- lm(GPA~.-StudentID-GradeClass, data = DATA)
summary(OLS)
#Stepwise regression
STEP_REG <- step(OLS, direction = "both", trace = FALSE)
summary(STEP_REG)
BEST_MODEL <- lm(GPA ~ Age + Gender + StudyTimeWeekly + Absences + 
                   Tutoring + ParentalSupport + Extracurricular + Sports + Music, 
                 data = DATA)

#way of checking heteroskedasticity
FITTED <- fitted(BEST_MODEL)
RESIDUALS <- resid(BEST_MODEL)
plot(GPA, RESIDUALS)

#####TEST FOR HETEROSKEDASTICITY#####
library(lmtest)
BP <- bptest(BEST_MODEL)
BP #there is homoskedasticity

#####QUANTILE REGRESSION#####
library(quantreg)
library(stargazer)
QR10 <- rq(GPA ~ Age + Gender + StudyTimeWeekly + Absences +
             Tutoring + ParentalSupport + Extracurricular + Sports + Music,
           data = DATA, tau = 0.10)
summary(QR10)
QR25<- rq(GPA ~ Age + Gender + StudyTimeWeekly + Absences +
            Tutoring + ParentalSupport + Extracurricular + Sports + Music,
          data = DATA, tau = 0.25)
summary(QR25)
QR90 <- rq(GPA ~ Age + Gender + StudyTimeWeekly + Absences +
             Tutoring + ParentalSupport + Extracurricular + Sports + Music,
           data = DATA, tau = 0.90)
summary(QR90)
stargazer(BEST_MODEL, QR10, QR25, QR90, type = "text")

# Perform Wald test
anova_results <- anova.rq(QR10, QR25, QR90, test = "Wald")
print(anova_results) #no significant difference, linear regression works fine

ALL_QR <- rq(GPA ~ Age + Gender + StudyTimeWeekly + Absences +
               Tutoring + ParentalSupport + Extracurricular + Sports + Music,
             data = DATA,
             tau = seq(0.05, 0.95, by = 0.05))
summary(ALL_QR)
plot(summary(ALL_QR))

#LINEAR REGRESSION MODEL WORKS FINE TO PREDICT GPA 

#############LOGISTIC REGRESSION############
str(DATA)
DATA$GradeClass <- NULL
# Recode GradeClass 
DATA$GradeClass <- ifelse(DATA$GPA >= 2.5, "C or more", "Below C")
DATA$GradeClass <- factor(DATA$GradeClass, levels = c("Below C", "C or more"))
table(DATA$GradeClass)
levels(DATA$GradeClass)
attach(DATA)
table(Gender)
TABLE1 <- addmargins(table(GradeClass, Gender))
TABLE1

TABLE2 <- addmargins(table(GradeClass, Sports))
TABLE2

TABLE3 <- addmargins(table(GradeClass, Tutoring))
TABLE3
#########SUBSETTING DATA########
library(psych)
C_OR_MORE <- subset(DATA, GradeClass == "C or more")
describe(C_OR_MORE)

BELOW_C <- subset(DATA, GradeClass == "Below C")
describe(BELOW_C)
########TEST SET AND TRAIN SET####
# Set seed for reproducibility
set.seed(123)

# Split the data into training (70%) and testing (30%)
train_index <- sample(seq_len(nrow(DATA)), size = 0.7 * nrow(DATA), replace = F)

TRAIN_SET <- DATA[train_index, ]
TEST_SET <- DATA[-train_index, ]
dim(TRAIN_SET)
dim(TEST_SET)
str(DATA)
attach(DATA)
#######LOGISTIC REGRESSION MODEL#####
attach(TRAIN_SET)
FULL_MODEL <- glm(GradeClass~.-StudentID-GPA, family = binomial(), data = TRAIN_SET)
summary(FULL_MODEL)
#######MODEL SELECTION########
library(MASS)
stepAIC(FULL_MODEL)

BEST_MODEL <- glm(GradeClass ~ Age + StudyTimeWeekly + Absences + 
                    Tutoring + ParentalSupport + Extracurricular + Sports + Music, 
                  family = binomial(), data = TRAIN_SET)
stargazer(FULL_MODEL, BEST_MODEL, type = "text")
#testing full model against reduced model
REDUCED_MODEL = BEST_MODEL
anova(FULL_MODEL, REDUCED_MODEL)
library(lmtest)
lrtest(FULL_MODEL, REDUCED_MODEL) #there is no significant difference between the models
######APPLY BEST MODEL TO TEST SET#######
attach(TEST_SET)
table(GradeClass)

PREDICTED_PROBS <- predict(BEST_MODEL, newdata = TEST_SET, type = "response")
PREDICTED_PROBS
describe(PREDICTED_PROBS)

#####ASSESS MODEL PERFORMANDE USING CONFUSION MATRIX#####
PREDICTED_CLASS <- ifelse(PREDICTED_PROBS>0.5, "C or more", "Below C")
CONFUSION_MATRIX <- table(PREDICTED_CLASS, TEST_SET$GradeClass)
CONFUSION_MATRIX
############CONF MATRIX USING CARET########
library(caret)
confusionMatrix(as.factor(PREDICTED_CLASS), as.factor(TEST_SET$GradeClass),
                positive = "C or more")
#########ROC & AUC########
library(pROC)
#we are using predicted probs, NOT pred class
ROC <- roc(TEST_SET$GradeClass~PREDICTED_PROBS, positive = "C or more")
plot(ROC, legacy.axes = TRUE)
AUC <- auc(ROC)
AUC

#setting the best probability threshold
YOUDEN_STAT <- coords(ROC, "best")
YOUDEN_STAT
#reviewed model performance with the Youden Stat
PREDICTED_PROBS <- predict(BEST_MODEL, newdata = TEST_SET, type = "response")
PREDICTED_PROBS
describe(PREDICTED_PROBS)

PREDICTED_CLASS <- ifelse(PREDICTED_PROBS>0.410736, "C or more", "Below C")
CONFUSION_MATRIX <- table(PREDICTED_CLASS, TEST_SET$GradeClass)
CONFUSION_MATRIX

library(caret)
confusionMatrix(as.factor(PREDICTED_CLASS), as.factor(TEST_SET$GradeClass),
                positive = "C or more")

#######NAIVE BAYES#####
str(TRAIN_SET)
library(e1071)
NAIVE_BAYES <- naiveBayes(GradeClass~.-StudentID-GPA, data = TRAIN_SET)
NAIVE_BAYES
#now with laplace
NAIVE_BAYES <- naiveBayes(GradeClass~.-StudentID-GPA,, data = TRAIN_SET, laplace = 2)
NAIVE_BAYES
#######APPLY TO TEST_SET#########
PREDICTED_CLASS1 <- predict(NAIVE_BAYES, TEST_SET)
PREDICTED_CLASS1

PROBS1 <- predict(NAIVE_BAYES, TEST_SET, "raw")
PROBS1
PROBABILITIES1 <- PROBS1[,2]

GRADECLASS <-TEST_SET$GradeClass
GRADECLASS
table(PREDICTED_CLASS1, GRADECLASS)
#########ROC & AUC############
ROC <- roc(GRADECLASS, PROBABILITIES1, positive = "C or more")
ROC
plot(ROC, legacy.axes = TRUE)

AUC <- auc(ROC)
AUC

#Similar performances as the log reg model but the latter is better performing.