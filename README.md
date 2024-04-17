# What drives the price of a car?
### Johennie Helton
#### April, 2024

This project is the Practical assignment for module 11to identify what drives the price of a car. 
The analysis and findings were performed using python; additionally, sklearn, matplotlib.pyplot, seaborn, pandas, and numpy libraries were used for computation and visualization.

Notebook: https://github.com/johennie/what_drives_a_car_price/blob/main/notebooks/prompt_II.ipynb

Data: https://github.com/johennie/what_drives_a_car_price/tree/main/data

## Summary for car salesmen
The most important features driving the price of a used car are odometer, year, type, fuel, and manufacturer. 
Where the higher the odometer reading, the lower the price. Similarly, the older the car manufacturing 
year, the lower the price. The fuel type also affects the price, with electric having the higher price, 
followed by hybrid and then gas. The manufacturer that drives the price up are the ferri cars. That is

1- target newer cars with low odometer readings 

2- target luxury cars manufacturers such as ferrari and aston-martin

3- target trucks with diesel

## Report
### Problem Statement
To provide car salesmen with which features affect the car prices for used cars based on the associated dataset found in the data/vehicle.csv file. 

### 1. Outcomes and Predictions
We are going to read and clean the vehicle data, then do an 80-20 split into train and test so the identified independent columns will be used by the models to predict the price.

The X_train and y_train data are then fit into the models, used to predict the price. Based on the r^2 and error values we will be able to identify the best model to use for these predictions, and based on the coefficient values of this model we will be able to identify the features that affect the price the most.

**My hypothesis** is that the age, condition and manufacturer are the top three factors affecting the price of a vehicle.

Note: We decided to fit X with the manufacturer, year, odometer, condition, fuel, transmission, size, and type columns; and set independent variable y to the price column which is the target of the predictions.

### 2. Data Acquisition and Data Understanding
The provided dataset contains information on 426K cars, we need to examine it and determine outliers, 
NAN, invalid and unique values for each column.
During this phase we are to describe and explore the data to make sure it can be used for analysis 
and visualization in understanding car prices.

The following columns are in the raw data:
       Column       	Description	
       id       	  	Unique record identifier
       region  	    	String representing the region
       price      	  	Car price in $
       year        	  	Year of manufacture: 1980, 2020…
       manufacturer  	Car Manufacturer: nissan, ford, kia…
       model  	  	    Car Model: silverado, tacoma…
       condition 	  	Car condition: excelent, fair, good,  like new, new, salvage
       cylinders  	  	Car cylinders: 3 cylinders, 4 cylinders…
       fuel            	Car Fuel type: gas, electric…
      odometer 	    	Car miles
      title_status   	Car Title Status: lien, clean …
      transmission 	  	Car Transmission: manual, automatic, other
      VIN         	  	Unique Car identifier
      drive      	  	Car Wheel Drive: 4wd, fwd, rwd
      size        	 	Car Size: compact, full-size…
      type       	 	Car Type: bus, pickup…
      paint_color  	 	Car Paint Color: black, blue…
      state      	 	State: az, ny…
<br>
2.1. checked for missing values, duplicates and outliers
Based on this information we decide to drop and modify data
![fig2.png](images%2Ffig2.png)
<br>
2.2. analyzed the data present in the columns and counted the unique values
This is to help us identify categorical columns which are the non numeric columns that have few unique possibilities
![fig3.png](images%2Ffig3.png)
<br>
There are 426880 records, which seem to have NAN values which will give us issues during model data fiting. The id and VIN columns won't affect the price, which leaves year, price and odometer as the numerical columns that we can start with. In addition condition, cylinders, fuel, title_status, transmission, drive, size columns as potential categorical data columns.
<br>
## 3. Data Preparation
3.1. created a new dataframe: updated_df
<br>
3.2. replaced NaN, null and/or missing entries with 0 or column.mean
Note: while fitting the data to the model we kept getting NaNs and Inf errors. This was one of the most challenging parts of data preparation for this data set and we identified the NaNs and Inf values, drop or fill them a default value.
<br>
3.3. analyzed the numerical data first: year, odometer, and price columns
During this step, we cleaned up bad data format or impossible values (such as looking for cars older than 1850). We also removed outliers.
<br>
3.4. identified the categorical columns which may be selected for modeling. We identify them looking at the unique values of non numerical columns
The categorical columns are 'year','odometer','condition', 'fuel', 'transmission', 'size', 'type'
In addition I chose to use manufacturer as categorical because I think it is an important feature in car prices.
<br>
3.5 Finally, we scaled and normalized the data for analysis

The correlation matrix shows that price is affected by year, odometer and type as well as some manufacturers
![fig4.png](images%2Ffig4.png)

The luxury car manufacturers have the higher price
![fig5.png](images%2Ffig5.png)

... aa well as the newer cars.
![fig6.png](images%2Ffig6.png)

### Modeling
section to execute selected models, and collect any statistical information to aid in selecting the best model to use<br>     
    calculate X and y<br>
    <br>
    - 5.1. Linear Regression models<br>    
        -- 5.1.1. Model1: Linear regression<br>
        -- 5.1.2. Model2: Ridge regression <br>
        -- 5.1.3. Model3: Lasso regression <br>

                        model	Model r^2	MSE on train set	MSE on testing set
0	regressor=LinearRegression	    0.241071	6.515457e+07	6.783581e+07
1	regressor=Ridge \n (alpha=0.1)	0.241071	6.515457e+07	6.783581e+07
2	regressor=Lasso \n (alphas=0.1)	0.241071	6.515457e+07	6.783581e+07
    <br>
     - 5.2. Linear models with regularization <br>     
         -- 5.2.1. Model1: RidgeCV with alphas=np.logspace(-10, 10, 21) <br>

![fig8.png](images%2Ffig8.png)
                                                model	selected alpha	Model r^2	MSE on train set	MSE on testing set
0	regressor=Ridge (alpha=np.logspace(-10, 10, 21))	0.1	0.569665	3.621024e+07	3.846489e+07
     <br>
    - 5.3. GridSearchCV<br>

                        model	selected alpha	Model r^2	MSE on train set	MSE on testing set
model__alpha	GridSearchCV 	      0.01      0.562382	3.699130e+07	    3.911588e+07
       with Ridge and alphas=[1,
10, 20, 50, 100, 150, 200, 250, 500, 10


![fig7.png](images%2Ffig7.png)
<br>
### Model Evaluation
Base on the r^2 values the best model is Ridge with alpha = 0.01 which gives us an r^2 = 0.562382 
which is not very high at all, I attempted to increase that value by removing some columns but it did not work. 
The next steps would be to try different models and data transformations.
Meanwhile, using the results of this model I would recommend for the car salesmen to 

1- target newer cars with low odometer readings 

2- target luxury cars manufacturers such as ferrari and aston-martin

3- target trucks with diesel

This project disproved my hypothesis that the age of the car, the condition and the manufacturer where going to be the driving factors affecting the price of a vehicle. Instead we have odometer as the leading feature, followed by the type of car and then the year.
