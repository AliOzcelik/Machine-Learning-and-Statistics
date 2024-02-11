library("readxl")
library("ggplot2")
library("ggcorrplot")
library("olsrr")
library("qcc")

df <- read_excel("/Users/desidero/Desktop/Dersler/IE266/data.xlsx")

df_male <- df[df$Gender == "Male", ]
df_female <- df[df$Gender == "Female", ]


#Scatter Plot
plot(df$Age, df$`Revenue($)` ,
     main = "Age vs `Revenue($)`  w.r.t Gender", 
     xlab = "Age",
     ylab = "`Revenue($)` ")
points(df$Age[df$Gender == 'Male'], df$`Revenue($)`[df$Gender == 'Male'], col='blue')
points(df$Age[df$Gender == 'Female'], df$`Revenue($)`[df$Gender == 'Female'], col='magenta')
abline(lm(df_female$`Revenue($)` ~ df_female$Age), col = 'magenta')
abline(lm(df_male$`Revenue($)`  ~ df_male$Age), col = 'steelblue')
lines(lowess(df$Age, df$`Revenue($)` ), col = "orange")

plot(df$`Pages Visited`, df$`Revenue($)`,
     main = "Pages Visited vs `Revenue($)`  w.r.t Gender", 
     xlab = "Pages Visited",
     ylab = "`Revenue($)` ")
points(df$`Pages Visited`[df$Gender == 'Male'], df$`Revenue($)` [df$Gender == 'Male'], col='blue')
points(df$`Pages Visited`[df$Gender == 'Female'], df$`Revenue($)` [df$Gender == 'Female'], col='magenta')
abline(lm(df_female$`Revenue($)`  ~ df_female$`Pages Visited`), col = 'magenta')
abline(lm(df_male$`Revenue($)`  ~ df_male$`Pages Visited`), col = 'steelblue')
lines(lowess(df$`Pages Visited`, df$`Revenue($)` ), col = "orange")

plot(df$`Time Spent (min)`, df$`Revenue($)` ,
     main = "Time Spent (min) vs `Revenue($)`  w.r.t Gender", 
     xlab = "Time Spent (min)",
     ylab = "`Revenue($)` ")
points(df$`Time Spent (min)`[df$Gender == 'Male'], df$`Revenue($)` [df$Gender == 'Male'], col='blue')
points(df$`Time Spent (min)`[df$Gender == 'Female'], df$`Revenue($)` [df$Gender == 'Female'], col='magenta')
abline(lm(df_female$`Revenue($)`  ~ df_female$`Time Spent (min)`), col = 'magenta')
abline(lm(df_male$`Revenue($)`  ~ df_male$`Time Spent (min)`), col = 'steelblue')
lines(lowess(df$`Time Spent (min)`, df$`Revenue($)` ), col = "orange")


plot(df$`Products Purchased`, df$`Revenue($)` ,
     main = "Products Purchased vs `Revenue($)`  w.r.t Gender", 
     xlab = "Products Purchased",
     ylab = "`Revenue($)` ")
points(df$`Products Purchased`[df$Gender == 'Male'], df$`Revenue($)` [df$Gender == 'Male'], col='blue')
points(df$`Products Purchased`[df$Gender == 'Female'], df$`Revenue($)` [df$Gender == 'Female'], col='magenta')
abline(lm(df_female$`Revenue($)`  ~ df_female$`Products Purchased`), col = 'magenta')
abline(lm(df_male$`Revenue($)`  ~ df_male$`Products Purchased`), col = 'steelblue')
#lines(lowess(df$`Products Purchased`, df$`Revenue($)` ), col = "orange")

df$Gender <- ifelse(df$Gender == "Male", 1, 0)


plot(df$`Products Purchased`, df$`Revenue($)` ,
     main = "Products Purchased vs `Revenue($)` ", 
     xlab = "Products Purchased",
     ylab = "`Revenue($)` ")
lines(lowess(df$`Products Purchased`, df$`Revenue($)` ), col = "orange")

plot(df$Age, df$`Revenue($)` ,
     main = "Age vs `Revenue($)` ", 
     xlab = "Age",
     ylab = "`Revenue($)` ")
lines(lowess(df$Age, df$`Revenue($)` ), col = "orange")

plot(df$`Pages Visited`, df$`Revenue($)` ,
     main = "Pages Visited vs `Revenue($)` ", 
     xlab = "Pages Visited",
     ylab = "`Revenue($)` ")
lines(lowess(df$`Pages Visited`, df$`Revenue($)` ), col = "orange")

plot(df$`Time Spent (min)`, df$`Revenue($)` ,
     main = "Time Spent (min) vs `Revenue($)` ", 
     xlab = "Time Spent (min)",
     ylab = "`Revenue($)` ")
lines(lowess(df$`Time Spent (min)`, df$`Revenue($)` ), col = "orange")

df["Gender_TimeSpent"] <- df$Gender*df$`Time Spent (min)`
df["Gender_PagesVisited"] <- df$Gender*df$`Pages Visited`
df["ProductsPurchased2"] <- df$`Products Purchased`^(2)

model <- lm(formula = `Revenue($)`  ~ Age + Gender + `Pages Visited` + `Time Spent (min)` + `Products Purchased` + 
                 Gender_TimeSpent + Gender_PagesVisited +  ProductsPurchased2, data = df)
summary(model)
plot(model,1)
plot(model,2)
plot(model,3)


df2 <- df[-c(8,18,35,50),]
model_c1 <- lm(formula = `Revenue($)`  ~ Age + Gender + `Pages Visited` + `Time Spent (min)` + `Products Purchased` + 
                 Gender_TimeSpent + Gender_PagesVisited +  ProductsPurchased2, data = df2)
summary(model_c1)
plot(model_c1,1)
plot(model_c1,2)
plot(model_c1,3)

model_c6 <- lm(formula = `Revenue($)`  ~ Age + Gender + `Pages Visited` + `Time Spent (min)` + `Products Purchased` + 
                    Gender_TimeSpent + ProductsPurchased2, data = df2)
summary(model_c6)

# "both" for both forward selection and backward elimination
step_model <- step(model_c1, direction = "both", trace = 1)  
adj_r2_values <- c(summary(model_c1)$adj.r.squared, summary(step_model)$adj.r.squared)
summary(step_model)

print(adj_r2_values)

summary(step_model)
anova(step_model)

df2["`Revenue($)` "] <- (df2$`Revenue($)` )
#Now, we are ready to answer the D part of the 2nd question
plot(df2$Age + df2$Gender + df2$`Pages Visited` + df2$`Products Purchased` + 
          df2$Gender_TimeSpent + df2$ProductsPurchased2,df2$`Revenue($)` ,
     main = "Regression Model vs Response", 
     xlab = "Regression Model",
     ylab = "Response")
lines(lowess(df2$Age + df2$Gender + df2$`Time Spent (min)` + df2$`Products Purchased` + 
                  df2$Gender_TimeSpent + df2$ProductsPurchased2, df2$`Revenue($)` ), col = "darkblue")



#Answer to the 3rd question
a <- data.frame(Age=25, Gender=0, `Pages Visited`=2, `Time Spent (min)`=15, `Products Purchased`=4, Gender_TimeSpent=0, 
                Gender_PagesVisited=0, ProductsPurchased2=16)
predict(step_model, newdata = a, interval = 'confidence',level = 0.95)


#Answer to the 4th question
b <- data.frame(Age=32, Gender=1, `Pages Visited`=4, `Time Spent (min)`=21, `Products Purchased`=2, Gender_TimeSpent=21, 
                Gender_PagesVisited=4, ProductsPurchased2=4)
predict(step_model, newdata = b, interval = 'prediction',level = 0.95)
