# Stock-HIGH-LOW-Prediction.
To predict stock’s day HIGH and LOW prices based on its OPEN and historical daily price by LSTM model.

# Problem

To predict stock’s day HIGH and LOW prices based on its OPEN and historical daily price.


## Data Description

The data was downloaded from the Yahoo Finance. The underlying stock is Tencent (700.hk). I renamed the column names as the following:

**DATE**: Trading date

**OPEN**:Current day open price

**CLOSE**:Current day last traded price recorded on exchange

**HIGH**:Current day highest trade price

**LOW**:Current day lowest trade price

**VOLUME**:Current day trade volume


## Data File Path

The data file has been uploaded as https://mryinglee.github.io/DailyPrices.csv



```
DATA_PATH='https://mryinglee.github.io/DailyPrices.csv'
```

# Academic Review

The time-varying volatility and volatility clustering are often observed in the financial market. High and low price can be treated as an indicator of volatility. There are some papers to research on prediction of high and low price. 

## High and Low Prices Research Review
(Mainly digested from  **High and Low Intraday Commodity
Prices: A Fractional Integration and
Cointegration Approach**
Yaya, OlaOluwa S and Gil-Alana, Luis A.
University of Navarra, Spain, University of Ibadan, Nigeria
, 5 December 2018)


In financial economics, the difference between high and low intraday or daily prices is known
as the range. Volatility can be expected to be higher if the range is wider. Parkinson (1980)
showed that, in fact, the price range is a more efficient volatility estimator than alternatives
such as the return-based estimator. It is also frequently used in technical analysis by traders in
financial markets (see, e.g., Taylor and Allen, 1992). However, as pointed out by Cheung et al.
(2009), focusing on the range itself might be useful if one’s only purpose is to obtain an
efficient proxy for the underlying volatility, but it also means discarding useful information
about price behaviour that can be found in its components. Therefore, in their study, Cheung
et al. (2009) analyse simultaneously both the range and daily highs and lows using daily data
for various stock market indices. Because of the observation that the latter two variables
generally do not diverge significantly over time, having found that they both exhibit unit roots
by carrying out Augmented Dickey-Fuller (ADF) tests (Dickey and Fuller, 1979), they model
their behaviour using a cointegration framework as in Johansen (1991) and Johansen and
Juselius (1990) to investigate whether they are linked through a long-run equilibrium
relationship, and interpreting the range as a stationary error correction term. They then show
that such a model has better in-sample properties than rival ARMA specifications but does not
clearly outperform them in terms of its out-of-sample properties.

Following on from Cheung et al. (2009), the present study makes a twofold contribution
to the literature. First, it uses fractional integration and cointegration methods that are more
general than the standard framework based on the I(0) versus I(1) dichotomy. According to the
efficient market hypothesis (EMH), asset prices should be unpredictable and follow a random
walk (see Fama, 1970), i.e. they should be integrated of order 1 or I(1). However, the choice
between stationary I(0) and nonstationary I(1) processes is too restrictive for most financial
series (Barunik and Dvorakova, 2015). Diebold and Rudebusch (1991) and Hasslers and
3
Wolters (1994) showed that in fact unit root tests have very low power in the context of
fractional integration. Therefore our analysis below allows the differencing parameter for the
individual series to take fractional values. Moreover, we adopt a fractional cointegration
approach to test for the long-run relationships. Fiess and MacDonald (2002), Cheung (2007)
and Cheung et al. (2009) all modelled high and low prices together with the range in a
cointegration framework to analyse the foreign exchange and stock markets, respectively.
However, their studies restrict the cointegrating parameter to unity (even though this is not
imposed in Granger’s (1986) seminal paper). 
Fractional cointegration models (see also Robinson and Yajima, 2002; Nielsen and
Shimotsu, 2007; etc.) are more general and have already been shown to be more suitable for
many financial series (see, e.g., Caporale and Gil-Alana, 2014 and Erer et al., 2016). The
FCVAR model in particular has a number of advantages over the fractional cointegration setup
of Robinson and Marrinuci (2003): it allows for multiple time series and long-run
equilibrium relationships to be determined using the statistical test of MacKinnon and Nielsen
(2014), and it jointly estimates the adjustment coefficients and the cointegrating relations.1
Nielsen and Popiel (2018) provide a Matlab package for the calculation of the estimators and
test statistics. Dolatabadi et al. (2016) applied the FCVAR model to analyse the relationship
between spot and futures prices in future commodity markets and found more support for
cointegration compared to the case when the cointegration parameters are restricted to unity.

## On the predictability of stock prices: a case for high and low prices
Caporin, M., Ranaldo, A. and Santucci de Magistris, P. (2013). On the predictability of stock
prices: a case for high and low prices. Journal of Banking and Finance, 37(12), 5132–5146. argued:

*Contrary to the common wisdom that asset prices are hardly possible to forecast, we
show that high and low prices of equity shares are largely predictable. We propose to model
them using a simple implementation of a fractional vector autoregressive model with error
correction (FVECM). This model captures two fundamental patterns of high and low prices:
their cointegrating relationship and the long memory of their difference (i.e. the range),
which is a measure of realized volatility. Investment strategies based on FVECM predictions
of high/low US equity prices as exit/entry signals deliver a superior performance even on a
risk-adjusted basis.*

## Some FCVAR related implementation

[LeeMorinUCF/FCVAR](https://github.com/LeeMorinUCF/FCVAR): Fractionally Cointegrated VAR Model based on Matlab.

[Tinkat/FCVAR](https://github.com/Tinkat/FCVAR): Fractionally cointegrated vector autoregressive model based on R.



# Why Machine Learning?

Machine learning is a morden tools to process stock data. 

## Why LSTM model?
LSTM is a good model to deal with process with memory effect. I have a summary on LSTM model on Github ([Stock-Price-Specific-LSTM](https://github.com/MRYingLEE/Stock-Price-Specific-LSTM))

# Setup Environment

We use Keras + Tensorflow along with other Python ecosystem packages, such as numpy, pandas and matplotlib.


```
!pip install tensorflow-gpu
```

```
!pip install mpl_finance
```

```
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.python.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential

%matplotlib inline

rcParams['figure.figsize'] = 28, 16

```

# Load Data and Overview Data


```
df=pd.read_csv(DATA_PATH,parse_dates=['DATE'])
df.set_index('DATE',inplace=True)
df = df.sort_values('DATE')
df.head
```




    <bound method NDFrame.head of              OPEN   HIGH    LOW  CLOSE  ADJ CLOSE    VOLUME
    DATE                                                       
    2015-01-21  126.2  128.9  125.2  128.7      127.1  33788560
    2015-01-22  130.5  132.0  129.7  131.6      130.0  39063807
    2015-01-23  134.6  134.9  131.1  132.7      131.1  29965533
    2015-01-26  136.7  137.1  134.3  137.0      135.3  34952624
    2015-01-27  138.0  138.0  133.0  136.0      134.3  24455759
    ...           ...    ...    ...    ...        ...       ...
    2020-01-14  410.0  413.0  396.6  400.4      400.4  26827634
    2020-01-15  397.2  403.0  396.2  398.8      398.8  15938138
    2020-01-16  399.0  403.0  396.4  400.0      400.0  13770626
    2020-01-17  400.0  400.6  396.0  399.0      399.0  13670846
    2020-01-20  405.0  405.0  396.0  396.0      396.0  13282412
    
    [1231 rows x 6 columns]>




```
df.shape
```




    (1231, 6)




```
df.dropna(inplace=True)
```


```
df.shape
```




    (1231, 6)




```
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc
df_candle = df[['OPEN', 'HIGH', 'LOW', 'CLOSE']]

df_candle.reset_index(inplace=True)
df_candle['DATE'] = df_candle['DATE'].map(mdates.date2num)
# df_candle.drop(columns='DATE',inplace=True)
df_candle=df_candle.astype(float)
df_candle=df_candle[['DATE','OPEN', 'HIGH', 'LOW', 'CLOSE']]
df_candle.head
```

    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      





    <bound method NDFrame.head of           DATE   OPEN   HIGH    LOW  CLOSE
    0     735619.0  126.2  128.9  125.2  128.7
    1     735620.0  130.5  132.0  129.7  131.6
    2     735621.0  134.6  134.9  131.1  132.7
    3     735624.0  136.7  137.1  134.3  137.0
    4     735625.0  138.0  138.0  133.0  136.0
    ...        ...    ...    ...    ...    ...
    1226  737438.0  410.0  413.0  396.6  400.4
    1227  737439.0  397.2  403.0  396.2  398.8
    1228  737440.0  399.0  403.0  396.4  400.0
    1229  737441.0  400.0  400.6  396.0  399.0
    1230  737444.0  405.0  405.0  396.0  396.0
    
    [1231 rows x 5 columns]>




```

# Define a new subplot
ax = plt.subplot()

# Plot the candlestick chart and put date on x-Axis
candlestick_ohlc(ax, df_candle.values, width=5, colorup='g', colordown='r')
ax.xaxis_date()

# Turn on grid
ax.grid(True)

# Show plot
plt.show()
```


![png](Stock_HIGH_LOW_Prediction_files/Stock_HIGH_LOW_Prediction_19_0.png)


Obviously, there are **gaps** in the candle chart. 
In other words, the current day's price range[LOW, HIGH] has no overlap to that of the previous day.

### OPEN price instead of the previous prices is a better indicator for HIGH/LOW
1. In theory, market price is to response to market news, which most likely is released between trading days. And news could trigger the price jump up or down, which makes the previous price range cannot be a good leading indicator for the current price range. So that the OPNE price has priced in more news.
2. Even for the situation of dividends, the OPEN price as an indicator can deal with, which is better than previous prices.
3. The OPEN is always in the range of [LOW, HIGH].
4. It's pratical to predict HIGH/LOW after the market is open. Especially for a day trader.
5. Technically if we use OPEN as a benchmark, OPEN/HIGH and LOW/OPEN are both belongs to (0,1], which is good for deep learning to optimize.

### If we don't use OPEN as an indicator, in feature engineering, we should use other capping tricks to make sure the relative of HIGH and LOW are within (0,1].

## To Check the Tick Movement Frequency Distribution

We need to check the tick distribution to standardize the tick data


```
TICK_UNIT=0.2
```


```
df['HIGH_tick']=(np.round((df['HIGH']-df['OPEN'])/TICK_UNIT)).astype('int32')
df['LOW_tick']=(np.round((df['OPEN']-df['LOW'])/TICK_UNIT)).astype('int32')

df['RANGE_TICKS']=df['HIGH_tick']+df['LOW_tick'] ## ticks between day low and day high
```


```
df.max()
```




    OPEN                 474.0
    HIGH                 476.6
    LOW                  466.8
    CLOSE                474.6
    ADJ CLOSE            472.3
    VOLUME         308436765.0
    HIGH_tick            102.0
    LOW_tick              84.0
    RANGE_TICKS          108.0
    dtype: float64




```
df.mean()
```




    OPEN           2.691491e+02
    HIGH           2.717566e+02
    LOW            2.661037e+02
    CLOSE          2.688868e+02
    ADJ CLOSE      2.676347e+02
    VOLUME         2.027254e+07
    HIGH_tick      1.303574e+01
    LOW_tick       1.522177e+01
    RANGE_TICKS    2.825751e+01
    dtype: float64




```
df[["HIGH_tick","LOW_tick"]].groupby("HIGH_tick").count().head(40)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LOW_tick</th>
    </tr>
    <tr>
      <th>HIGH_tick</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>183</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48</td>
    </tr>
    <tr>
      <th>2</th>
      <td>64</td>
    </tr>
    <tr>
      <th>3</th>
      <td>55</td>
    </tr>
    <tr>
      <th>4</th>
      <td>63</td>
    </tr>
    <tr>
      <th>5</th>
      <td>77</td>
    </tr>
    <tr>
      <th>6</th>
      <td>28</td>
    </tr>
    <tr>
      <th>7</th>
      <td>42</td>
    </tr>
    <tr>
      <th>8</th>
      <td>48</td>
    </tr>
    <tr>
      <th>9</th>
      <td>43</td>
    </tr>
    <tr>
      <th>10</th>
      <td>40</td>
    </tr>
    <tr>
      <th>11</th>
      <td>33</td>
    </tr>
    <tr>
      <th>12</th>
      <td>32</td>
    </tr>
    <tr>
      <th>13</th>
      <td>33</td>
    </tr>
    <tr>
      <th>14</th>
      <td>40</td>
    </tr>
    <tr>
      <th>15</th>
      <td>36</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
    </tr>
    <tr>
      <th>17</th>
      <td>24</td>
    </tr>
    <tr>
      <th>18</th>
      <td>22</td>
    </tr>
    <tr>
      <th>19</th>
      <td>28</td>
    </tr>
    <tr>
      <th>20</th>
      <td>19</td>
    </tr>
    <tr>
      <th>21</th>
      <td>16</td>
    </tr>
    <tr>
      <th>22</th>
      <td>12</td>
    </tr>
    <tr>
      <th>23</th>
      <td>11</td>
    </tr>
    <tr>
      <th>24</th>
      <td>17</td>
    </tr>
    <tr>
      <th>25</th>
      <td>12</td>
    </tr>
    <tr>
      <th>26</th>
      <td>12</td>
    </tr>
    <tr>
      <th>27</th>
      <td>10</td>
    </tr>
    <tr>
      <th>28</th>
      <td>11</td>
    </tr>
    <tr>
      <th>29</th>
      <td>11</td>
    </tr>
    <tr>
      <th>30</th>
      <td>6</td>
    </tr>
    <tr>
      <th>31</th>
      <td>1</td>
    </tr>
    <tr>
      <th>32</th>
      <td>15</td>
    </tr>
    <tr>
      <th>33</th>
      <td>8</td>
    </tr>
    <tr>
      <th>34</th>
      <td>10</td>
    </tr>
    <tr>
      <th>35</th>
      <td>9</td>
    </tr>
    <tr>
      <th>36</th>
      <td>4</td>
    </tr>
    <tr>
      <th>37</th>
      <td>7</td>
    </tr>
    <tr>
      <th>38</th>
      <td>6</td>
    </tr>
    <tr>
      <th>39</th>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```
df[["HIGH_tick","LOW_tick"]].groupby("LOW_tick").count().head(40)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HIGH_tick</th>
    </tr>
    <tr>
      <th>LOW_tick</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>108</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
    </tr>
    <tr>
      <th>2</th>
      <td>50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>47</td>
    </tr>
    <tr>
      <th>5</th>
      <td>58</td>
    </tr>
    <tr>
      <th>6</th>
      <td>39</td>
    </tr>
    <tr>
      <th>7</th>
      <td>60</td>
    </tr>
    <tr>
      <th>8</th>
      <td>53</td>
    </tr>
    <tr>
      <th>9</th>
      <td>47</td>
    </tr>
    <tr>
      <th>10</th>
      <td>57</td>
    </tr>
    <tr>
      <th>11</th>
      <td>37</td>
    </tr>
    <tr>
      <th>12</th>
      <td>45</td>
    </tr>
    <tr>
      <th>13</th>
      <td>43</td>
    </tr>
    <tr>
      <th>14</th>
      <td>21</td>
    </tr>
    <tr>
      <th>15</th>
      <td>42</td>
    </tr>
    <tr>
      <th>16</th>
      <td>19</td>
    </tr>
    <tr>
      <th>17</th>
      <td>29</td>
    </tr>
    <tr>
      <th>18</th>
      <td>21</td>
    </tr>
    <tr>
      <th>19</th>
      <td>25</td>
    </tr>
    <tr>
      <th>20</th>
      <td>32</td>
    </tr>
    <tr>
      <th>21</th>
      <td>20</td>
    </tr>
    <tr>
      <th>22</th>
      <td>20</td>
    </tr>
    <tr>
      <th>23</th>
      <td>19</td>
    </tr>
    <tr>
      <th>24</th>
      <td>11</td>
    </tr>
    <tr>
      <th>25</th>
      <td>20</td>
    </tr>
    <tr>
      <th>26</th>
      <td>15</td>
    </tr>
    <tr>
      <th>27</th>
      <td>13</td>
    </tr>
    <tr>
      <th>28</th>
      <td>8</td>
    </tr>
    <tr>
      <th>29</th>
      <td>10</td>
    </tr>
    <tr>
      <th>30</th>
      <td>16</td>
    </tr>
    <tr>
      <th>31</th>
      <td>11</td>
    </tr>
    <tr>
      <th>32</th>
      <td>8</td>
    </tr>
    <tr>
      <th>33</th>
      <td>11</td>
    </tr>
    <tr>
      <th>34</th>
      <td>12</td>
    </tr>
    <tr>
      <th>35</th>
      <td>3</td>
    </tr>
    <tr>
      <th>36</th>
      <td>4</td>
    </tr>
    <tr>
      <th>37</th>
      <td>11</td>
    </tr>
    <tr>
      <th>38</th>
      <td>10</td>
    </tr>
    <tr>
      <th>39</th>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```
df_low_freq=pd.DataFrame({'count' : df[["HIGH_tick","LOW_tick"]].groupby("LOW_tick").size()}).reset_index()
plt.bar(df_low_freq["LOW_tick"], df_low_freq["count"])
```




    <BarContainer object of 72 artists>




![png](Stock_HIGH_LOW_Prediction_files/Stock_HIGH_LOW_Prediction_29_1.png)



```
df_high_freq=pd.DataFrame({'count' : df[["HIGH_tick","LOW_tick"]].groupby("HIGH_tick").size()}).reset_index()
plt.bar(df_high_freq["HIGH_tick"], df_high_freq["count"])
```




    <BarContainer object of 73 artists>




![png](Stock_HIGH_LOW_Prediction_files/Stock_HIGH_LOW_Prediction_30_1.png)



```
df_range_freq=pd.DataFrame({'count' : df[["RANGE_TICKS","LOW_tick"]].groupby("RANGE_TICKS").size()}).reset_index()
plt.bar(df_range_freq["RANGE_TICKS"], df_range_freq["count"])
```




    <BarContainer object of 89 artists>




![png](Stock_HIGH_LOW_Prediction_files/Stock_HIGH_LOW_Prediction_31_1.png)



```
MAX_RANGE_TICKS=100 # We use this to standardize the RANGE_TICKS
```

### Findings on Tick Movement Frequency Distribution

Based on OPEN data, the up/down tick movement can be 0-100.

For the situations of outliers, we may miss some better opportunity to make money, but the model can be more stable if we ignore the outliers.

## Overview Conclusion
There are no date conflicts, such as duplicate dates.
There are no N/A values.
But there are many price gaps. So even both of the next HIGH/LOW could be out of the range of the current day.

**OPEN price instead of the previous prices is a better indicator for HIGH/LOW.**


# Feature Engineering


```
df['CLOSE-1']=df['CLOSE'].shift(1)       # The LAST price of the previous day
df['OPEN/LAST-1']=df['OPEN']/df['CLOSE-1'] # The ratio of the current OPEN with the LAST price of the prevous day
df['OPEN+1']=df['OPEN/LAST-1'].shift(-1) # The next OPEN price
df['OPEN/HIGH']=df['OPEN']/df['HIGH'] # The ratio of OPEN with HIGH, which belongs to (0,1]
df['LOW/OPEN']=df['LOW']/df['OPEN'] # The ratio of LOW with OPEN, which belongs to (0,1]
```


```
df.dropna(inplace=True)
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OPEN</th>
      <th>HIGH</th>
      <th>LOW</th>
      <th>CLOSE</th>
      <th>ADJ CLOSE</th>
      <th>VOLUME</th>
      <th>HIGH_tick</th>
      <th>LOW_tick</th>
      <th>RANGE_TICKS</th>
      <th>CLOSE-1</th>
      <th>OPEN/LAST-1</th>
      <th>OPEN+1</th>
      <th>OPEN/HIGH</th>
      <th>LOW/OPEN</th>
    </tr>
    <tr>
      <th>DATE</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-22</th>
      <td>130.5</td>
      <td>132.0</td>
      <td>129.7</td>
      <td>131.6</td>
      <td>130.0</td>
      <td>39063807</td>
      <td>8</td>
      <td>4</td>
      <td>12</td>
      <td>128.7</td>
      <td>1.013986</td>
      <td>1.022796</td>
      <td>0.988636</td>
      <td>0.993870</td>
    </tr>
    <tr>
      <th>2015-01-23</th>
      <td>134.6</td>
      <td>134.9</td>
      <td>131.1</td>
      <td>132.7</td>
      <td>131.1</td>
      <td>29965533</td>
      <td>2</td>
      <td>18</td>
      <td>20</td>
      <td>131.6</td>
      <td>1.022796</td>
      <td>1.030143</td>
      <td>0.997776</td>
      <td>0.973997</td>
    </tr>
    <tr>
      <th>2015-01-26</th>
      <td>136.7</td>
      <td>137.1</td>
      <td>134.3</td>
      <td>137.0</td>
      <td>135.3</td>
      <td>34952624</td>
      <td>2</td>
      <td>12</td>
      <td>14</td>
      <td>132.7</td>
      <td>1.030143</td>
      <td>1.007299</td>
      <td>0.997082</td>
      <td>0.982443</td>
    </tr>
    <tr>
      <th>2015-01-27</th>
      <td>138.0</td>
      <td>138.0</td>
      <td>133.0</td>
      <td>136.0</td>
      <td>134.3</td>
      <td>24455759</td>
      <td>0</td>
      <td>25</td>
      <td>25</td>
      <td>137.0</td>
      <td>1.007299</td>
      <td>1.006618</td>
      <td>1.000000</td>
      <td>0.963768</td>
    </tr>
    <tr>
      <th>2015-01-28</th>
      <td>136.9</td>
      <td>137.0</td>
      <td>134.7</td>
      <td>136.9</td>
      <td>135.2</td>
      <td>16216906</td>
      <td>0</td>
      <td>11</td>
      <td>11</td>
      <td>136.0</td>
      <td>1.006618</td>
      <td>0.993426</td>
      <td>0.999270</td>
      <td>0.983930</td>
    </tr>
  </tbody>
</table>
</div>



## Outliers Processing by a customized scaler
To standardize the tick distribution

We will use our customized scale function to turn the tick data to the value of (0,1]. All outliers will be assign 1. 


```
def capped_with_1(tick_value):
        return min(1, tick_value)
        
df["HIGH_tick_std"] = df.apply(lambda x: capped_with_1(x["HIGH_tick"]/MAX_RANGE_TICKS),axis=1)
df["LOW_tick_std"] = df.apply(lambda x: capped_with_1(x["LOW_tick"]/MAX_RANGE_TICKS),axis=1)
df["RANGE_TICKS_std"]=df.apply(lambda x: capped_with_1(x["RANGE_TICKS"]/MAX_RANGE_TICKS),axis=1)
#df["LOW_tick_std"].max()
```


```
df.dropna(inplace=True) # Due to shift(1), the first row has N/A data
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OPEN</th>
      <th>HIGH</th>
      <th>LOW</th>
      <th>CLOSE</th>
      <th>ADJ CLOSE</th>
      <th>VOLUME</th>
      <th>HIGH_tick</th>
      <th>LOW_tick</th>
      <th>RANGE_TICKS</th>
      <th>CLOSE-1</th>
      <th>OPEN/LAST-1</th>
      <th>OPEN+1</th>
      <th>OPEN/HIGH</th>
      <th>LOW/OPEN</th>
      <th>HIGH_tick_std</th>
      <th>LOW_tick_std</th>
      <th>RANGE_TICKS_std</th>
    </tr>
    <tr>
      <th>DATE</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-22</th>
      <td>130.5</td>
      <td>132.0</td>
      <td>129.7</td>
      <td>131.6</td>
      <td>130.0</td>
      <td>39063807</td>
      <td>8</td>
      <td>4</td>
      <td>12</td>
      <td>128.7</td>
      <td>1.013986</td>
      <td>1.022796</td>
      <td>0.988636</td>
      <td>0.993870</td>
      <td>0.08</td>
      <td>0.04</td>
      <td>0.12</td>
    </tr>
    <tr>
      <th>2015-01-23</th>
      <td>134.6</td>
      <td>134.9</td>
      <td>131.1</td>
      <td>132.7</td>
      <td>131.1</td>
      <td>29965533</td>
      <td>2</td>
      <td>18</td>
      <td>20</td>
      <td>131.6</td>
      <td>1.022796</td>
      <td>1.030143</td>
      <td>0.997776</td>
      <td>0.973997</td>
      <td>0.02</td>
      <td>0.18</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>2015-01-26</th>
      <td>136.7</td>
      <td>137.1</td>
      <td>134.3</td>
      <td>137.0</td>
      <td>135.3</td>
      <td>34952624</td>
      <td>2</td>
      <td>12</td>
      <td>14</td>
      <td>132.7</td>
      <td>1.030143</td>
      <td>1.007299</td>
      <td>0.997082</td>
      <td>0.982443</td>
      <td>0.02</td>
      <td>0.12</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>2015-01-27</th>
      <td>138.0</td>
      <td>138.0</td>
      <td>133.0</td>
      <td>136.0</td>
      <td>134.3</td>
      <td>24455759</td>
      <td>0</td>
      <td>25</td>
      <td>25</td>
      <td>137.0</td>
      <td>1.007299</td>
      <td>1.006618</td>
      <td>1.000000</td>
      <td>0.963768</td>
      <td>0.00</td>
      <td>0.25</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>2015-01-28</th>
      <td>136.9</td>
      <td>137.0</td>
      <td>134.7</td>
      <td>136.9</td>
      <td>135.2</td>
      <td>16216906</td>
      <td>0</td>
      <td>11</td>
      <td>11</td>
      <td>136.0</td>
      <td>1.006618</td>
      <td>0.993426</td>
      <td>0.999270</td>
      <td>0.983930</td>
      <td>0.00</td>
      <td>0.11</td>
      <td>0.11</td>
    </tr>
  </tbody>
</table>
</div>



# Model 1: Return based LSTM model

If we can predict relative HIGH, LOW based on OPEN price, we can get the correponding real HIGH and LOW.

**Range Prediction**

The most difficluty for this model is that the task is too predict one HIGH and one LOW value, which define a range. If there is no constraint on the prediction, the predicted LOW could be higher the predicted HIGH. So we have to predict a reasonable range.

The trick we do is that we use OPEN/HIGH and OPEN/LOW as the indicators and the data to be predicted. These 2 generated columns have the range of (0,1]. And all combines of OPEN/HIGH and OPEN/LOW means a reasonable range. The hinted HIGH is always higher or equal to the LOW.

**The critical activation function for the output layer**

For the predicted values lies on (0,1], and they have no direct relation between them. So that sigmoid aactivation function suites to the output layer.

![Sigmoid](https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg)

Of course, there are a few possible activation cadidates also. Such as: TanH, SQNL，Gaussian and SQ-RBF. But Sigmoid has good feature that the predicted values concentrate to 1, which fits our situation.

The detail on activation function can be found at:

[Activation function](https://en.wikipedia.org/wiki/Activation_function)



## Preprocess


```
history_values = df[["RANGE_TICKS_std","OPEN/HIGH",	"LOW/OPEN"]].values
history_values.shape
```




    (1229, 3)




```
SEQ_LEN = 20 # about 1 month


def to_sequences(data, seq_len):
    d = []

    for index in range(len(data) - seq_len):
        d.append((data[index: index + seq_len]))
        #d.append(data[index: index + seq_len])

    na=np.array(d)
    #print(na.shape)
    #print(na[:2])
    return na

def preprocess(data_raw, seq_len, train_split):

    data = to_sequences(data_raw, seq_len)

    num_train = int(train_split * data.shape[0])

    X_train = data[:num_train, :-1,: ]
    y_train = data[:num_train, -1, -2:]

    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:,-1, -2:]

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = preprocess(history_values, SEQ_LEN, train_split = 0.95)
```


```
X_train.shape
```




    (1148, 19, 3)



## Model Define
This model is copied from a model to predict close price of a crypto currency.
This model is selectd for the author argued the model has some good technical features.

A common LSTM should be fine too.

The difference is that here are 2 outputs with sigmoid as activation function.


```
DROPOUT = 0.2 # In case overfitting
WINDOW_SIZE = (SEQ_LEN - 1)

model = keras.Sequential()

model.add(Bidirectional(CuDNNLSTM(WINDOW_SIZE, return_sequences=True),
                        input_shape=(WINDOW_SIZE, X_train.shape[-1])))
model.add(Dropout(rate=DROPOUT))

model.add(Bidirectional(CuDNNLSTM((WINDOW_SIZE * 2), return_sequences=True)))
model.add(Dropout(rate=DROPOUT))

model.add(Bidirectional(CuDNNLSTM(WINDOW_SIZE, return_sequences=False)))

model.add(Dense(units=2))

model.add(Activation('sigmoid')) # So the value output is in the (0,1] 
```

## Training


```
model.compile(
    loss='mean_squared_error', 
    optimizer=keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-09,# We choose a much smaller value for the predicted value is always near 1, so the loss can be always very small. In order to distinguish 1 tick difference, we need a very small epsilon.
    amsgrad=False,
    name='Adam')
)
```


```
BATCH_SIZE = 64

history = model.fit(
    X_train, 
    y_train, 
    epochs=100, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    validation_split=0.1
)
```

    Train on 1033 samples, validate on 115 samples
## Training Log


```
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```


![png](Stock_HIGH_LOW_Prediction_files/Stock_HIGH_LOW_Prediction_53_0.png)



```
plt.plot(history.history['loss'][-50:])
plt.plot(history.history['val_loss'][-50:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```


![png](Stock_HIGH_LOW_Prediction_files/Stock_HIGH_LOW_Prediction_54_0.png)



```
model.evaluate(X_test, y_test)
```

    61/61 [==============================] - 0s 293us/sample - loss: 4.5586e-05





    4.558617681633208e-05



## Model Predict


```
X_test.shape[0]
```




    61




```
y_hat = model.predict(X_test)
df_test= df[-X_test.shape[0]:].reset_index()
df_test
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DATE</th>
      <th>OPEN</th>
      <th>HIGH</th>
      <th>LOW</th>
      <th>CLOSE</th>
      <th>ADJ CLOSE</th>
      <th>VOLUME</th>
      <th>HIGH_tick</th>
      <th>LOW_tick</th>
      <th>RANGE_TICKS</th>
      <th>CLOSE-1</th>
      <th>OPEN/LAST-1</th>
      <th>OPEN+1</th>
      <th>OPEN/HIGH</th>
      <th>LOW/OPEN</th>
      <th>HIGH_tick_std</th>
      <th>LOW_tick_std</th>
      <th>RANGE_TICKS_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-10-21</td>
      <td>329.6</td>
      <td>330.2</td>
      <td>324.8</td>
      <td>324.8</td>
      <td>324.8</td>
      <td>13947162</td>
      <td>3</td>
      <td>24</td>
      <td>27</td>
      <td>331.0</td>
      <td>0.995770</td>
      <td>1.000616</td>
      <td>0.998183</td>
      <td>0.985437</td>
      <td>0.03</td>
      <td>0.24</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-10-22</td>
      <td>325.0</td>
      <td>327.8</td>
      <td>324.8</td>
      <td>327.6</td>
      <td>327.6</td>
      <td>10448427</td>
      <td>14</td>
      <td>1</td>
      <td>15</td>
      <td>324.8</td>
      <td>1.000616</td>
      <td>0.991453</td>
      <td>0.991458</td>
      <td>0.999385</td>
      <td>0.14</td>
      <td>0.01</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-10-23</td>
      <td>324.8</td>
      <td>325.8</td>
      <td>319.6</td>
      <td>320.0</td>
      <td>320.0</td>
      <td>19855257</td>
      <td>5</td>
      <td>26</td>
      <td>31</td>
      <td>327.6</td>
      <td>0.991453</td>
      <td>0.996875</td>
      <td>0.996931</td>
      <td>0.983990</td>
      <td>0.05</td>
      <td>0.26</td>
      <td>0.31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-10-24</td>
      <td>319.0</td>
      <td>320.6</td>
      <td>316.6</td>
      <td>319.0</td>
      <td>319.0</td>
      <td>18472498</td>
      <td>8</td>
      <td>12</td>
      <td>20</td>
      <td>320.0</td>
      <td>0.996875</td>
      <td>1.004389</td>
      <td>0.995009</td>
      <td>0.992476</td>
      <td>0.08</td>
      <td>0.12</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-10-25</td>
      <td>320.4</td>
      <td>320.4</td>
      <td>316.6</td>
      <td>316.6</td>
      <td>316.6</td>
      <td>15789881</td>
      <td>0</td>
      <td>19</td>
      <td>19</td>
      <td>319.0</td>
      <td>1.004389</td>
      <td>0.999368</td>
      <td>1.000000</td>
      <td>0.988140</td>
      <td>0.00</td>
      <td>0.19</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>56</th>
      <td>2020-01-13</td>
      <td>400.0</td>
      <td>406.4</td>
      <td>397.4</td>
      <td>406.4</td>
      <td>406.4</td>
      <td>27570261</td>
      <td>32</td>
      <td>13</td>
      <td>45</td>
      <td>398.6</td>
      <td>1.003512</td>
      <td>1.008858</td>
      <td>0.984252</td>
      <td>0.993500</td>
      <td>0.32</td>
      <td>0.13</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>57</th>
      <td>2020-01-14</td>
      <td>410.0</td>
      <td>413.0</td>
      <td>396.6</td>
      <td>400.4</td>
      <td>400.4</td>
      <td>26827634</td>
      <td>15</td>
      <td>67</td>
      <td>82</td>
      <td>406.4</td>
      <td>1.008858</td>
      <td>0.992008</td>
      <td>0.992736</td>
      <td>0.967317</td>
      <td>0.15</td>
      <td>0.67</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th>58</th>
      <td>2020-01-15</td>
      <td>397.2</td>
      <td>403.0</td>
      <td>396.2</td>
      <td>398.8</td>
      <td>398.8</td>
      <td>15938138</td>
      <td>29</td>
      <td>5</td>
      <td>34</td>
      <td>400.4</td>
      <td>0.992008</td>
      <td>1.000502</td>
      <td>0.985608</td>
      <td>0.997482</td>
      <td>0.29</td>
      <td>0.05</td>
      <td>0.34</td>
    </tr>
    <tr>
      <th>59</th>
      <td>2020-01-16</td>
      <td>399.0</td>
      <td>403.0</td>
      <td>396.4</td>
      <td>400.0</td>
      <td>400.0</td>
      <td>13770626</td>
      <td>20</td>
      <td>13</td>
      <td>33</td>
      <td>398.8</td>
      <td>1.000502</td>
      <td>1.000000</td>
      <td>0.990074</td>
      <td>0.993484</td>
      <td>0.20</td>
      <td>0.13</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>60</th>
      <td>2020-01-17</td>
      <td>400.0</td>
      <td>400.6</td>
      <td>396.0</td>
      <td>399.0</td>
      <td>399.0</td>
      <td>13670846</td>
      <td>3</td>
      <td>20</td>
      <td>23</td>
      <td>400.0</td>
      <td>1.000000</td>
      <td>1.015038</td>
      <td>0.998502</td>
      <td>0.990000</td>
      <td>0.03</td>
      <td>0.20</td>
      <td>0.23</td>
    </tr>
  </tbody>
</table>
<p>61 rows × 18 columns</p>
</div>




```

df_hat = pd.DataFrame(data =y_hat,columns=['OPEN/HIGH<hat>','LOW/OPEN<hat>'])
df_hat
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>OPEN/HIGH&lt;hat&gt;</th>
      <th>LOW/OPEN&lt;hat&gt;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.990460</td>
      <td>0.988908</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.990460</td>
      <td>0.988910</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.990456</td>
      <td>0.988902</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.990457</td>
      <td>0.988905</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.990456</td>
      <td>0.988903</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>56</th>
      <td>0.990469</td>
      <td>0.988909</td>
    </tr>
    <tr>
      <th>57</th>
      <td>0.990469</td>
      <td>0.988913</td>
    </tr>
    <tr>
      <th>58</th>
      <td>0.990467</td>
      <td>0.988908</td>
    </tr>
    <tr>
      <th>59</th>
      <td>0.990469</td>
      <td>0.988905</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.990483</td>
      <td>0.988921</td>
    </tr>
  </tbody>
</table>
<p>61 rows × 2 columns</p>
</div>




```
df_compare=pd.concat([df_test, df_hat], axis=1)

df_compare
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DATE</th>
      <th>OPEN</th>
      <th>HIGH</th>
      <th>LOW</th>
      <th>CLOSE</th>
      <th>ADJ CLOSE</th>
      <th>VOLUME</th>
      <th>HIGH_tick</th>
      <th>LOW_tick</th>
      <th>RANGE_TICKS</th>
      <th>CLOSE-1</th>
      <th>OPEN/LAST-1</th>
      <th>OPEN+1</th>
      <th>OPEN/HIGH</th>
      <th>LOW/OPEN</th>
      <th>HIGH_tick_std</th>
      <th>LOW_tick_std</th>
      <th>RANGE_TICKS_std</th>
      <th>OPEN/HIGH&lt;hat&gt;</th>
      <th>LOW/OPEN&lt;hat&gt;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-10-21</td>
      <td>329.6</td>
      <td>330.2</td>
      <td>324.8</td>
      <td>324.8</td>
      <td>324.8</td>
      <td>13947162</td>
      <td>3</td>
      <td>24</td>
      <td>27</td>
      <td>331.0</td>
      <td>0.995770</td>
      <td>1.000616</td>
      <td>0.998183</td>
      <td>0.985437</td>
      <td>0.03</td>
      <td>0.24</td>
      <td>0.27</td>
      <td>0.990460</td>
      <td>0.988908</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-10-22</td>
      <td>325.0</td>
      <td>327.8</td>
      <td>324.8</td>
      <td>327.6</td>
      <td>327.6</td>
      <td>10448427</td>
      <td>14</td>
      <td>1</td>
      <td>15</td>
      <td>324.8</td>
      <td>1.000616</td>
      <td>0.991453</td>
      <td>0.991458</td>
      <td>0.999385</td>
      <td>0.14</td>
      <td>0.01</td>
      <td>0.15</td>
      <td>0.990460</td>
      <td>0.988910</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-10-23</td>
      <td>324.8</td>
      <td>325.8</td>
      <td>319.6</td>
      <td>320.0</td>
      <td>320.0</td>
      <td>19855257</td>
      <td>5</td>
      <td>26</td>
      <td>31</td>
      <td>327.6</td>
      <td>0.991453</td>
      <td>0.996875</td>
      <td>0.996931</td>
      <td>0.983990</td>
      <td>0.05</td>
      <td>0.26</td>
      <td>0.31</td>
      <td>0.990456</td>
      <td>0.988902</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-10-24</td>
      <td>319.0</td>
      <td>320.6</td>
      <td>316.6</td>
      <td>319.0</td>
      <td>319.0</td>
      <td>18472498</td>
      <td>8</td>
      <td>12</td>
      <td>20</td>
      <td>320.0</td>
      <td>0.996875</td>
      <td>1.004389</td>
      <td>0.995009</td>
      <td>0.992476</td>
      <td>0.08</td>
      <td>0.12</td>
      <td>0.20</td>
      <td>0.990457</td>
      <td>0.988905</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-10-25</td>
      <td>320.4</td>
      <td>320.4</td>
      <td>316.6</td>
      <td>316.6</td>
      <td>316.6</td>
      <td>15789881</td>
      <td>0</td>
      <td>19</td>
      <td>19</td>
      <td>319.0</td>
      <td>1.004389</td>
      <td>0.999368</td>
      <td>1.000000</td>
      <td>0.988140</td>
      <td>0.00</td>
      <td>0.19</td>
      <td>0.19</td>
      <td>0.990456</td>
      <td>0.988903</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>56</th>
      <td>2020-01-13</td>
      <td>400.0</td>
      <td>406.4</td>
      <td>397.4</td>
      <td>406.4</td>
      <td>406.4</td>
      <td>27570261</td>
      <td>32</td>
      <td>13</td>
      <td>45</td>
      <td>398.6</td>
      <td>1.003512</td>
      <td>1.008858</td>
      <td>0.984252</td>
      <td>0.993500</td>
      <td>0.32</td>
      <td>0.13</td>
      <td>0.45</td>
      <td>0.990469</td>
      <td>0.988909</td>
    </tr>
    <tr>
      <th>57</th>
      <td>2020-01-14</td>
      <td>410.0</td>
      <td>413.0</td>
      <td>396.6</td>
      <td>400.4</td>
      <td>400.4</td>
      <td>26827634</td>
      <td>15</td>
      <td>67</td>
      <td>82</td>
      <td>406.4</td>
      <td>1.008858</td>
      <td>0.992008</td>
      <td>0.992736</td>
      <td>0.967317</td>
      <td>0.15</td>
      <td>0.67</td>
      <td>0.82</td>
      <td>0.990469</td>
      <td>0.988913</td>
    </tr>
    <tr>
      <th>58</th>
      <td>2020-01-15</td>
      <td>397.2</td>
      <td>403.0</td>
      <td>396.2</td>
      <td>398.8</td>
      <td>398.8</td>
      <td>15938138</td>
      <td>29</td>
      <td>5</td>
      <td>34</td>
      <td>400.4</td>
      <td>0.992008</td>
      <td>1.000502</td>
      <td>0.985608</td>
      <td>0.997482</td>
      <td>0.29</td>
      <td>0.05</td>
      <td>0.34</td>
      <td>0.990467</td>
      <td>0.988908</td>
    </tr>
    <tr>
      <th>59</th>
      <td>2020-01-16</td>
      <td>399.0</td>
      <td>403.0</td>
      <td>396.4</td>
      <td>400.0</td>
      <td>400.0</td>
      <td>13770626</td>
      <td>20</td>
      <td>13</td>
      <td>33</td>
      <td>398.8</td>
      <td>1.000502</td>
      <td>1.000000</td>
      <td>0.990074</td>
      <td>0.993484</td>
      <td>0.20</td>
      <td>0.13</td>
      <td>0.33</td>
      <td>0.990469</td>
      <td>0.988905</td>
    </tr>
    <tr>
      <th>60</th>
      <td>2020-01-17</td>
      <td>400.0</td>
      <td>400.6</td>
      <td>396.0</td>
      <td>399.0</td>
      <td>399.0</td>
      <td>13670846</td>
      <td>3</td>
      <td>20</td>
      <td>23</td>
      <td>400.0</td>
      <td>1.000000</td>
      <td>1.015038</td>
      <td>0.998502</td>
      <td>0.990000</td>
      <td>0.03</td>
      <td>0.20</td>
      <td>0.23</td>
      <td>0.990483</td>
      <td>0.988921</td>
    </tr>
  </tbody>
</table>
<p>61 rows × 20 columns</p>
</div>



## From Prediction to Ticks (Practical Price)

The prediction is actually OPEN based HIGH and LOW return. We need to convert it back to real price in the units of TICK.




```
df_compare['HIGH_hat']=np.floor((df_compare['OPEN']/df_compare['OPEN/HIGH<hat>'])/TICK_UNIT)*TICK_UNIT
df_compare['LOW_hat']=np.ceil((df_compare['OPEN']*df_compare['LOW/OPEN<hat>'])/TICK_UNIT)*TICK_UNIT
df_compare['HIGH_diff']=df_compare['HIGH_hat']-df_compare['HIGH']
df_compare['LOW_diff']=df_compare['LOW_hat']-df_compare['LOW']
df_compare['HIGH_diff_tick']=(np.round(df_compare['HIGH_diff']/TICK_UNIT)).astype('int32')
df_compare['LOW_diff_tick']=(np.round(df_compare['LOW_diff']/TICK_UNIT)).astype('int32')
df_compare['RANGE_TICKS_hat']=(np.round((df_compare['HIGH_hat']-df_compare['LOW_hat'])/TICK_UNIT)).astype('int32')
```


```
df_compare[['DATE', 'OPEN', 'CLOSE', 'HIGH', 'HIGH_hat','HIGH_diff_tick', 'LOW','LOW_hat','LOW_diff_tick','RANGE_TICKS','RANGE_TICKS_hat']]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DATE</th>
      <th>OPEN</th>
      <th>CLOSE</th>
      <th>HIGH</th>
      <th>HIGH_hat</th>
      <th>HIGH_diff_tick</th>
      <th>LOW</th>
      <th>LOW_hat</th>
      <th>LOW_diff_tick</th>
      <th>RANGE_TICKS</th>
      <th>RANGE_TICKS_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-10-21</td>
      <td>329.6</td>
      <td>324.8</td>
      <td>330.2</td>
      <td>332.6</td>
      <td>12</td>
      <td>324.8</td>
      <td>326.0</td>
      <td>6</td>
      <td>27</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-10-22</td>
      <td>325.0</td>
      <td>327.6</td>
      <td>327.8</td>
      <td>328.0</td>
      <td>1</td>
      <td>324.8</td>
      <td>321.4</td>
      <td>-17</td>
      <td>15</td>
      <td>33</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-10-23</td>
      <td>324.8</td>
      <td>320.0</td>
      <td>325.8</td>
      <td>327.8</td>
      <td>10</td>
      <td>319.6</td>
      <td>321.2</td>
      <td>8</td>
      <td>31</td>
      <td>33</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-10-24</td>
      <td>319.0</td>
      <td>319.0</td>
      <td>320.6</td>
      <td>322.0</td>
      <td>7</td>
      <td>316.6</td>
      <td>315.6</td>
      <td>-5</td>
      <td>20</td>
      <td>32</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-10-25</td>
      <td>320.4</td>
      <td>316.6</td>
      <td>320.4</td>
      <td>323.4</td>
      <td>15</td>
      <td>316.6</td>
      <td>317.0</td>
      <td>2</td>
      <td>19</td>
      <td>32</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>56</th>
      <td>2020-01-13</td>
      <td>400.0</td>
      <td>406.4</td>
      <td>406.4</td>
      <td>403.8</td>
      <td>-13</td>
      <td>397.4</td>
      <td>395.6</td>
      <td>-9</td>
      <td>45</td>
      <td>41</td>
    </tr>
    <tr>
      <th>57</th>
      <td>2020-01-14</td>
      <td>410.0</td>
      <td>400.4</td>
      <td>413.0</td>
      <td>413.8</td>
      <td>4</td>
      <td>396.6</td>
      <td>405.6</td>
      <td>45</td>
      <td>82</td>
      <td>41</td>
    </tr>
    <tr>
      <th>58</th>
      <td>2020-01-15</td>
      <td>397.2</td>
      <td>398.8</td>
      <td>403.0</td>
      <td>401.0</td>
      <td>-10</td>
      <td>396.2</td>
      <td>392.8</td>
      <td>-17</td>
      <td>34</td>
      <td>41</td>
    </tr>
    <tr>
      <th>59</th>
      <td>2020-01-16</td>
      <td>399.0</td>
      <td>400.0</td>
      <td>403.0</td>
      <td>402.8</td>
      <td>-1</td>
      <td>396.4</td>
      <td>394.6</td>
      <td>-9</td>
      <td>33</td>
      <td>41</td>
    </tr>
    <tr>
      <th>60</th>
      <td>2020-01-17</td>
      <td>400.0</td>
      <td>399.0</td>
      <td>400.6</td>
      <td>403.8</td>
      <td>16</td>
      <td>396.0</td>
      <td>395.6</td>
      <td>-2</td>
      <td>23</td>
      <td>41</td>
    </tr>
  </tbody>
</table>
<p>61 rows × 11 columns</p>
</div>




```
df_compare
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DATE</th>
      <th>OPEN</th>
      <th>HIGH</th>
      <th>LOW</th>
      <th>CLOSE</th>
      <th>ADJ CLOSE</th>
      <th>VOLUME</th>
      <th>HIGH_tick</th>
      <th>LOW_tick</th>
      <th>RANGE_TICKS</th>
      <th>CLOSE-1</th>
      <th>OPEN/LAST-1</th>
      <th>OPEN+1</th>
      <th>OPEN/HIGH</th>
      <th>LOW/OPEN</th>
      <th>HIGH_tick_std</th>
      <th>LOW_tick_std</th>
      <th>RANGE_TICKS_std</th>
      <th>OPEN/HIGH&lt;hat&gt;</th>
      <th>LOW/OPEN&lt;hat&gt;</th>
      <th>HIGH_hat</th>
      <th>LOW_hat</th>
      <th>HIGH_diff</th>
      <th>LOW_diff</th>
      <th>HIGH_diff_tick</th>
      <th>LOW_diff_tick</th>
      <th>RANGE_TICKS_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-10-21</td>
      <td>329.6</td>
      <td>330.2</td>
      <td>324.8</td>
      <td>324.8</td>
      <td>324.8</td>
      <td>13947162</td>
      <td>3</td>
      <td>24</td>
      <td>27</td>
      <td>331.0</td>
      <td>0.995770</td>
      <td>1.000616</td>
      <td>0.998183</td>
      <td>0.985437</td>
      <td>0.03</td>
      <td>0.24</td>
      <td>0.27</td>
      <td>0.990460</td>
      <td>0.988908</td>
      <td>332.6</td>
      <td>326.0</td>
      <td>2.4</td>
      <td>1.2</td>
      <td>12</td>
      <td>6</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-10-22</td>
      <td>325.0</td>
      <td>327.8</td>
      <td>324.8</td>
      <td>327.6</td>
      <td>327.6</td>
      <td>10448427</td>
      <td>14</td>
      <td>1</td>
      <td>15</td>
      <td>324.8</td>
      <td>1.000616</td>
      <td>0.991453</td>
      <td>0.991458</td>
      <td>0.999385</td>
      <td>0.14</td>
      <td>0.01</td>
      <td>0.15</td>
      <td>0.990460</td>
      <td>0.988910</td>
      <td>328.0</td>
      <td>321.4</td>
      <td>0.2</td>
      <td>-3.4</td>
      <td>1</td>
      <td>-17</td>
      <td>33</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-10-23</td>
      <td>324.8</td>
      <td>325.8</td>
      <td>319.6</td>
      <td>320.0</td>
      <td>320.0</td>
      <td>19855257</td>
      <td>5</td>
      <td>26</td>
      <td>31</td>
      <td>327.6</td>
      <td>0.991453</td>
      <td>0.996875</td>
      <td>0.996931</td>
      <td>0.983990</td>
      <td>0.05</td>
      <td>0.26</td>
      <td>0.31</td>
      <td>0.990456</td>
      <td>0.988902</td>
      <td>327.8</td>
      <td>321.2</td>
      <td>2.0</td>
      <td>1.6</td>
      <td>10</td>
      <td>8</td>
      <td>33</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-10-24</td>
      <td>319.0</td>
      <td>320.6</td>
      <td>316.6</td>
      <td>319.0</td>
      <td>319.0</td>
      <td>18472498</td>
      <td>8</td>
      <td>12</td>
      <td>20</td>
      <td>320.0</td>
      <td>0.996875</td>
      <td>1.004389</td>
      <td>0.995009</td>
      <td>0.992476</td>
      <td>0.08</td>
      <td>0.12</td>
      <td>0.20</td>
      <td>0.990457</td>
      <td>0.988905</td>
      <td>322.0</td>
      <td>315.6</td>
      <td>1.4</td>
      <td>-1.0</td>
      <td>7</td>
      <td>-5</td>
      <td>32</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-10-25</td>
      <td>320.4</td>
      <td>320.4</td>
      <td>316.6</td>
      <td>316.6</td>
      <td>316.6</td>
      <td>15789881</td>
      <td>0</td>
      <td>19</td>
      <td>19</td>
      <td>319.0</td>
      <td>1.004389</td>
      <td>0.999368</td>
      <td>1.000000</td>
      <td>0.988140</td>
      <td>0.00</td>
      <td>0.19</td>
      <td>0.19</td>
      <td>0.990456</td>
      <td>0.988903</td>
      <td>323.4</td>
      <td>317.0</td>
      <td>3.0</td>
      <td>0.4</td>
      <td>15</td>
      <td>2</td>
      <td>32</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>56</th>
      <td>2020-01-13</td>
      <td>400.0</td>
      <td>406.4</td>
      <td>397.4</td>
      <td>406.4</td>
      <td>406.4</td>
      <td>27570261</td>
      <td>32</td>
      <td>13</td>
      <td>45</td>
      <td>398.6</td>
      <td>1.003512</td>
      <td>1.008858</td>
      <td>0.984252</td>
      <td>0.993500</td>
      <td>0.32</td>
      <td>0.13</td>
      <td>0.45</td>
      <td>0.990469</td>
      <td>0.988909</td>
      <td>403.8</td>
      <td>395.6</td>
      <td>-2.6</td>
      <td>-1.8</td>
      <td>-13</td>
      <td>-9</td>
      <td>41</td>
    </tr>
    <tr>
      <th>57</th>
      <td>2020-01-14</td>
      <td>410.0</td>
      <td>413.0</td>
      <td>396.6</td>
      <td>400.4</td>
      <td>400.4</td>
      <td>26827634</td>
      <td>15</td>
      <td>67</td>
      <td>82</td>
      <td>406.4</td>
      <td>1.008858</td>
      <td>0.992008</td>
      <td>0.992736</td>
      <td>0.967317</td>
      <td>0.15</td>
      <td>0.67</td>
      <td>0.82</td>
      <td>0.990469</td>
      <td>0.988913</td>
      <td>413.8</td>
      <td>405.6</td>
      <td>0.8</td>
      <td>9.0</td>
      <td>4</td>
      <td>45</td>
      <td>41</td>
    </tr>
    <tr>
      <th>58</th>
      <td>2020-01-15</td>
      <td>397.2</td>
      <td>403.0</td>
      <td>396.2</td>
      <td>398.8</td>
      <td>398.8</td>
      <td>15938138</td>
      <td>29</td>
      <td>5</td>
      <td>34</td>
      <td>400.4</td>
      <td>0.992008</td>
      <td>1.000502</td>
      <td>0.985608</td>
      <td>0.997482</td>
      <td>0.29</td>
      <td>0.05</td>
      <td>0.34</td>
      <td>0.990467</td>
      <td>0.988908</td>
      <td>401.0</td>
      <td>392.8</td>
      <td>-2.0</td>
      <td>-3.4</td>
      <td>-10</td>
      <td>-17</td>
      <td>41</td>
    </tr>
    <tr>
      <th>59</th>
      <td>2020-01-16</td>
      <td>399.0</td>
      <td>403.0</td>
      <td>396.4</td>
      <td>400.0</td>
      <td>400.0</td>
      <td>13770626</td>
      <td>20</td>
      <td>13</td>
      <td>33</td>
      <td>398.8</td>
      <td>1.000502</td>
      <td>1.000000</td>
      <td>0.990074</td>
      <td>0.993484</td>
      <td>0.20</td>
      <td>0.13</td>
      <td>0.33</td>
      <td>0.990469</td>
      <td>0.988905</td>
      <td>402.8</td>
      <td>394.6</td>
      <td>-0.2</td>
      <td>-1.8</td>
      <td>-1</td>
      <td>-9</td>
      <td>41</td>
    </tr>
    <tr>
      <th>60</th>
      <td>2020-01-17</td>
      <td>400.0</td>
      <td>400.6</td>
      <td>396.0</td>
      <td>399.0</td>
      <td>399.0</td>
      <td>13670846</td>
      <td>3</td>
      <td>20</td>
      <td>23</td>
      <td>400.0</td>
      <td>1.000000</td>
      <td>1.015038</td>
      <td>0.998502</td>
      <td>0.990000</td>
      <td>0.03</td>
      <td>0.20</td>
      <td>0.23</td>
      <td>0.990483</td>
      <td>0.988921</td>
      <td>403.8</td>
      <td>395.6</td>
      <td>3.2</td>
      <td>-0.4</td>
      <td>16</td>
      <td>-2</td>
      <td>41</td>
    </tr>
  </tbody>
</table>
<p>61 rows × 27 columns</p>
</div>



## The Results Are Not Good Although the Training Seems Good

# How to improve







## Data

To improve the effectiveness of prediction, we need to add more data:
1. Instead of a single name, we need a big pool of names to train the model. The single name provides very limited data, which is very limited to train a model. Also, we will use the model to predict not only a single name but also for other names. And it’s better to have the data of the market index.

2. Instead of real HIGH and LOW values, we may need a more practical ones. The HIGH and LOW are extreme data, which is not stable and could be impacted by 1 price spike. Even we have predicted the extreme data correctly, which may be not feasible to fill an order. So instead of to predicate the HIGH and LOW, we may predicate the 95% confidence range based on the VWAP. Which could be more realistic.

3. Instead of a daily volume, we use an array of volume at price for the day. People believe that more volume means more evidence. So the extreme data (high/ low) can be predicted with more confidence.  

4. More reliable data, such as VWAP of AM, PM.

## Model

1. We may try other non-machine learning models, such as Fractionally cointegrated vector autoregressive model.

2. We may try other machine learning models, even auto-AI package.

3. We may tune hyperparameters.


# Conclusion

This repository was digested from a big project I did. In that project more feature engineering and extra model have been done and the results are good.

In this repsitory I didn't tune the hyperparameters yet. This is just for demo, but aim for a practical usage.
