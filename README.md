# G5 End Term Report

## Introduction

In this report, we present the findings and methodologies employed in our End Term project. Building upon the progress made in the mid-term report where we designed a reinforcement learning (RL) algorithm based on the actor-critic model using NIFTY50 data, we shifted our focus to time series analysis. Specifically, we delved into the implementation and evaluation of the Autoregressive Integrated Moving Average (ARIMA) model.

## ARIMA Model

### Overview
Autoregressive Integrated Moving Averages (ARIMA) is a popular statistical analysis model widely used for time series forecasting. ARIMA combines autoregressive (AR), differencing (I), and moving average (MA) components to model time series data.

### Components of ARIMA
1. **Autoregressive (AR) Component**: 
   - The AR component captures the relationship between an observation and a number of lagged observations (its own past values).
   
2. **Differencing (I) Component**:
   - The differencing component aims to make the time series data stationary by differencing consecutive observations.
   
3. **Moving Average (MA) Component**:
   - The MA component captures the relationship between an observation and a residual error from a moving average model applied to lagged observations.

let’s take the example the of the number of the light bulbs sold or the manufactured! L sub t or the number of light bulbs that is going to be created this month is going to be equal to some coefficient beta sub 0 this is just a constant. That's the constant beta sub 1 is a different coefficient and then it gets interesting which is L sub t minus 1 which is the number of light bulbs created last month so this is the autoregressive bit which is if we were to just stop here then this would be a AR 1 model; because it basically means that how many light bulbs is needed to be created this month is a function of how many light bulbs were created last month. But of course we also have this ma one bit which is this part here which says that not only is it a function of the number of light bulbs created last year. It's also a function of this coefficient this V sub one is a coefficient and
it's a function of epsilon sub T minus one which is the error from the previous time period from last month.

L<sub>t</sub> = &beta;<sub>0</sub> + &beta;<sub>1</sub>L<sub>t-1</sub> + &oslash;<sub>1</sub>&epsilon;<sub>t-1</sub> + &epsilon;<sub>t</sub>

This basically says that last month the prediction about how many light bulbs to. Whether the prediction to this was positive or negative this error is epsilon sub t minus one is how much it was off by in the previous period. So the error is incorporated here that how much it was wronged by into the new prediction for this month and there's a little bit distinction to be made. Here this would be the process itself and the epsilon sub T is there at the end so this is the error from this month.

L'<sub>t</sub> = &beta;<sub>0</sub> + &beta;<sub>1</sub>L<sub>t-1</sub> + &oslash;<sub>1</sub>&epsilon;<sub>t-1</sub>
L sub T hat as we remember in statistics or time series L sub T hat or anything hat is the pur predicted value whose real process is given by the thing above but of course the real process is unknown because if it was predicted the error from this month then with all the information the perfect prediction can be made which is obviously not true. So the predicted value for light bulbs created this month is going to be equal to this coefficient beta sub naught plus the coefficient beta sub 1 L sub t minus 1. The exact number of light bulbs needed last month is present because last month is in the past and the past knowledge is with us plus V sub 1 epsilon sub T minus 1. We also have access to the error from last month because it's past knowledge but we don't have access to the error from this month because it hasn't happened yet so our prediction would basically be given by this function right here.

The differenced series is the change between consecutive observations in the original series, and can be written as y′t = yt – yt−1. A closely related model allows the differences to have a non-zero mean. Then
y<sub>t</sub> - y<sub>t-1</sub> = c + &epsilon;<sub>t</sub> or y<sub>t</sub> = c + y<sub>t-1</sub> + &epsilon;<sub>t</sub>.

There are various types of the differencing too like the seasonal, second order. A moving average model do not uses the past values of the forecast variable in a regression rather it uses the errors in the values that is the differences in the predicted and the real values.


### ARIMA Equation
The general ARIMA(p, d, q) model can be represented by the equation:

y_t = c + &epsilon;_t + &theta;_1 &epsilon;_{t-1} + &theta;_2 &epsilon;_{t-2} + ... + &theta;_q &epsilon;_{t-q}

where:
Y<sub>t</sub> is the observed value at time t.  
c is a constant.  
&phi;<sub>i</sub> are the autoregressive coefficients.  
&theta;<sub>i</sub> are the moving average coefficients.  
&epsilon;<sub>t</sub> is white noise.

### Model Parameters
- **p**: Number of lag observations in the model.
- **d**: Number of times the raw observations are differenced to achieve stationarity.
- **q**: Size or order of the moving average window.

## Application of ARIMA

### Stationarity
Stationarity is a critical concept in time series analysis. A stationary series has statistical properties such as mean, variance, and autocorrelation that do not vary with time. Achieving stationarity often involves differencing the data. Here in order to remove the stationarity from the data to be used in the ARIMA model I will be using the Dickey Fuller Test which is a type of the unit root test. A unit root is a feature of some stochastic processes that can cause problems in statistical inference involving time series models. In simple terms, the unit root is non-stationary but does not always have a trend component.

#### Dickey Fuller Test
It uses an autoregressive model and optimizes an information criterion across multiple different lag values. A Dickey-Fuller test is a unit root test that tests the null hypothesis that α=1 in the following model equation. alpha is the coefficient of the first lag on Y. We employed the Dickey Fuller Test, a type of unit root test, to assess stationarity. The null hypothesis of this test is that the time series data has a unit root, indicating it is non-stationary. A rejection of the null hypothesis suggests stationarity.

## Evaluation Metrics

To evaluate the performance of our ARIMA model, we used the MSE and RMSE metrics.



