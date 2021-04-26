# IPL First Innings Score Prediction

### Deployed at: https://ipl-s3.herokuapp.com/

### Problem Statement:
As we always watch T-20 or ODI after playing few overs there’s a projected score given for First Innings based on their current performance to predict that after the end of the first innings the final score will something around this runs with this wicket. So here our task is to determine the live first innings Score / projected score based on input features such as (overs bowled till now, runs, wickets, runs in previous 5 overs, wickets in previous 5 overs) and also based on their previous Stats / records/ performance throughout the IPL (i.e. 2008 - 2017) as in given dataset. It’s a Regression Problem 
 
### Workflow:
![image](https://user-images.githubusercontent.com/61958476/116069677-4a3b0d80-a6a9-11eb-8e08-34d63a6a0907.png)

### Modelling:
#### 1. Linear Regression: Linear regression is used for finding linear relationship between target and one or more predictors. There are two types of linear regression- Simple and Multiple. Here we are working on multiple Linear regression in which task is to predict final projected score at the end of innings i.e. regression task.

#### 2. Ridge Regression: Ridge regression is a model tuning method that is used to analyse any data that suffers from multicollinearity. This method performs L2 regularization. When the issue of multicollinearity occurs, least-squares are unbiased, and variances are large, this results in predicted values to be far away from the actual values.

#### 3. Lasso Regression: It is used over regression methods for a more accurate prediction. This model uses shrinkage. Shrinkage is where data values are shrunk towards a central point as the mean. The lasso procedure encourages simple,sparse models

### Code Snippets:
![image](https://user-images.githubusercontent.com/61958476/116069958-a0a84c00-a6a9-11eb-97b4-6dde4161efb0.png)

![image](https://user-images.githubusercontent.com/61958476/116070007-b584df80-a6a9-11eb-8596-c3e83277bcd8.png)

![image](https://user-images.githubusercontent.com/61958476/116070091-cdf4fa00-a6a9-11eb-8e57-95a6e0a10792.png)

### Conclusion: Here Linear regression as well as Ridge and Lasso regression works fine with MSE of 250 +

### Final Output will look something like this
