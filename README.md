# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION

### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
      import numpy as np
      import pandas as pd
      import matplotlib.pyplot as plt
      
      # Load dataset (update path if needed!)
      data = pd.read_csv(r"C:\Users\admin\Downloads\favorite_music_dataset.csv", parse_dates=["Listened_Date"])
      
      # --- Preprocess listening data ---
      # Aggregate count of songs listened by Month
      data['Month'] = data['Listened_Date'].dt.to_period('M')
      monthly_data = data.groupby('Month').size().reset_index(name='Songs_Listened')
      monthly_data['Month'] = monthly_data['Month'].dt.to_timestamp()
      
      # --- Prepare centered time index ---
      years_fractional = monthly_data['Month'].dt.year + (monthly_data['Month'].dt.month - 1) / 12
      years = years_fractional.tolist()
      listens = monthly_data['Songs_Listened'].tolist()
      
      # Convert to centered X values
      X = [y - years[len(years) // 2] for y in years]
      x2 = [x**2 for x in X]
      n = len(years)
      xy = [x*y for x, y in zip(X, listens)]
      x3 = [x**3 for x in X]
      x4 = [x**4 for x in X]
      x2y = [x2_i*y for x2_i, y in zip(x2, listens)]
      
      # --- Linear Trend ---
      b_linear = (n*sum(xy) - sum(X)*sum(listens)) / (n*sum(x2) - (sum(X))**2)
      a_linear = (sum(listens) - b_linear*sum(X)) / n
      linear_trend = [a_linear + b_linear*x for x in X]
      
      # --- Polynomial Trend (Degree 2) ---
      coeff_matrix = np.array([
          [n, sum(X), sum(x2)],
          [sum(X), sum(x2), sum(x3)],
          [sum(x2), sum(x3), sum(x4)]
      ])
      Y = np.array([sum(listens), sum(xy), sum(x2y)])
      a_poly, b_poly, c_poly = np.linalg.solve(coeff_matrix, Y)
      poly_trend = [a_poly + b_poly*X[i] + c_poly*x2[i] for i in range(n)]
      
      # --- Add trends to DataFrame ---
      monthly_data['Linear Trend'] = linear_trend
      monthly_data['Polynomial Trend'] = poly_trend
      
      # --- Plot 1: Linear Trend Estimation ---
      plt.figure(figsize=(12, 5))
      plt.plot(monthly_data['Month'], monthly_data['Songs_Listened'], marker='o', label='Actual Listens', color='blue')
      plt.plot(monthly_data['Month'], monthly_data['Linear Trend'], linestyle='--', label='Linear Trend', color='black')
      plt.xlabel('Month')
      plt.ylabel('Number of Songs Listened')
      plt.title('Monthly Music Listening with Linear Trend')
      plt.legend()
      plt.grid(True)
      plt.show()
      
      # --- Plot 2: Polynomial Trend Estimation ---
      plt.figure(figsize=(12, 5))
      plt.plot(monthly_data['Month'], monthly_data['Songs_Listened'], marker='o', label='Actual Listens', color='blue')
      plt.plot(monthly_data['Month'], monthly_data['Polynomial Trend'], linestyle='-', label='Polynomial Trend (Degree 2)', color='red')
      plt.xlabel('Month')
      plt.ylabel('Number of Songs Listened')
      plt.title('Monthly Music Listening with Polynomial Trend (Degree 2)')
      plt.legend()
      plt.grid(True)
      plt.show()



### OUTPUT


<img width="997" height="468" alt="download" src="https://github.com/user-attachments/assets/d81bca5b-30be-49ac-9de4-2be5e18959c9" />


<img width="997" height="468" alt="download" src="https://github.com/user-attachments/assets/c6958c1d-c70a-4447-96d2-fffc393f6de1" />


### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
