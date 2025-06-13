Basic Data Information:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 743 entries, 0 to 742
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   Hour    743 non-null    float64
 1   Hits    735 non-null    float64
dtypes: float64(2)
memory usage: 11.7 KB
Dataset contains 743 samples and 2 features.
First 10 rows of data:
   Hour    Hits
0   1.0  2272.0
1   2.0     NaN
2   3.0  1386.0
3   4.0  1365.0
4   5.0  1488.0
5   6.0  1337.0
6   7.0  1883.0
7   8.0  2283.0
8   9.0  1335.0
9  10.0  1025.0
Found 8 missing values, accounting for 1.08% of the data.
2025-06-13 16:29:59.944 python[76795:11158813] +[IMKClient subclass]: chose IMKClient_Modern
2025-06-13 16:29:59.945 python[76795:11158813] +[IMKInputSession subclass]: chose IMKInputSession_Modern
2025-06-13 16:30:02.216 python[76795:11158813] error messaging the mach port for IMKCFRunLoopWakeUpReliable
Shape after filtering: (735,)
1st-degree polynomial error: 657.13
1st-Degree Polynomial Evaluation Metrics:
  MSE: 431822.81, RMSE: 657.13, MAPE: 28.42%
2th-Degree Polynomial Evaluation Metrics:
  MSE: 244875.52, RMSE: 494.85, MAPE: 22.44%
----------------------------------------
3th-Degree Polynomial Evaluation Metrics:
  MSE: 189592.03, RMSE: 435.42, MAPE: 20.73%
----------------------------------------
5th-Degree Polynomial Evaluation Metrics:
  MSE: 169339.75, RMSE: 411.51, MAPE: 19.97%
----------------------------------------
10th-Degree Polynomial Evaluation Metrics:
  MSE: 165907.93, RMSE: 407.32, MAPE: 19.56%
----------------------------------------
/Users/muxiaohui/training-ai/practice/test.py:92: RankWarning: Polyfit may be poorly conditioned
  model = np.poly1d(np.polyfit(x, y, degree))
20th-Degree Polynomial Evaluation Metrics:
  MSE: 150377.87, RMSE: 387.79, MAPE: 18.34%
----------------------------------------

Model Performance Comparison:
Degree  MSE             RMSE            MAPE(%)
--------------------------------------------------
1       431822.81               657.13          28.42
2       244875.52               494.85          22.44
3       189592.03               435.42          20.73
5       169339.75               411.51          19.97
10      165907.93               407.32          19.56
20      150377.87               387.79          18.34
Total error of piecewise linear model: 822.23
1th-Degree Polynomial (Post-Inflection) Evaluation Metrics:
  MSE: 150639.06, RMSE: 388.12, MAPE: 9.76%
----------------------------------------
2th-Degree Polynomial (Post-Inflection) Evaluation Metrics:
  MSE: 134481.95, RMSE: 366.72, MAPE: 9.16%
----------------------------------------
3th-Degree Polynomial (Post-Inflection) Evaluation Metrics:
  MSE: 134465.66, RMSE: 366.70, MAPE: 9.16%
----------------------------------------
5th-Degree Polynomial (Post-Inflection) Evaluation Metrics:
  MSE: 129141.14, RMSE: 359.36, MAPE: 8.95%
----------------------------------------
/Users/muxiaohui/training-ai/practice/test.py:140: RankWarning: Polyfit may be poorly conditioned
  model = np.poly1d(np.polyfit(xb, yb, degree))
10th-Degree Polynomial (Post-Inflection) Evaluation Metrics:
  MSE: 128906.78, RMSE: 359.04, MAPE: 8.90%
----------------------------------------
/Users/muxiaohui/training-ai/practice/test.py:163: RankWarning: Polyfit may be poorly conditioned
  model = np.poly1d(np.polyfit(xtrain, ytrain, degree))

Best Model: 2th-degree polynomial
Best Test Error: 332.72
Final Model Error: 366.70
Estimated time to reach 100000 hits/hour: 8.65 weeks
Time remaining until capacity limit: 4.22 weeks