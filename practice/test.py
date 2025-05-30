import numpy as np  
import matplotlib.pyplot as plt  
from scipy.optimize import fsolve  
import pandas as pd  
from sklearn.metrics import mean_squared_error as sk_mse  

np.random.seed(42)

# Load data
data = np.genfromtxt("web_traffic.tsv", delimiter="\t")  
df = pd.DataFrame(data, columns=['Hour', 'Hits'])  
print("Basic Data Information:")  
df.info()  
rows, columns = data.shape  
print(f"Dataset contains {rows} samples and {columns} features.")  
print("First 10 rows of data:")  
print(df.head(10).to_string())  

x = data[:, 0]  
y = data[:, 1]  
nan_mask = np.isnan(y)  
nan_count = np.sum(nan_mask)  
print(f"Found {nan_count} missing values, accounting for {nan_count/len(y):.2%} of the data.")  

if nan_count > 0:  
    plt.figure(figsize=(12, 3))  
    plt.plot(nan_mask, 'r.', label='Missing Values')  
    plt.title('Missing Value Distribution')  
    plt.xlabel('Sample Index')  
    plt.legend()  
    plt.show()  

x = x[~nan_mask]  
y = y[~nan_mask]  
print(f"Shape after filtering: {x.shape}")  

def plot_web_traffic(x, y, models=None, title="Web Traffic Over the Last Month", xlim=None, show_confidence=False):  
    plt.figure(figsize=(12, 6))  
    plt.scatter(x, y, s=10, alpha=0.6, label="Raw Data")  
    plt.title(title)  
    plt.xlabel("Time (Weeks)")  
    plt.ylabel("Hits/Hour")  
    weeks = np.arange(0, 5)  
    plt.xticks([w*7*24 for w in weeks], [f"Week {w+1}" for w in weeks])  
    if xlim: plt.xlim(xlim)  
    if models:  
        colors = ['g', 'k', 'b', 'm', 'r', 'c', 'y']  
        linestyles = ['-', '-.', '--', ':', '-', '--', '-.']  
        mx = np.linspace(0, x[-1]*1.2 if not xlim else xlim[1], 1000)  
        for i, model in enumerate(models):  
            label = f"{getattr(model, 'order', f'Model {i+1}')}"  
            plt.plot(mx, model(mx), linestyle=linestyles[i], linewidth=2, c=colors[i], label=label)  
            if show_confidence and i == 0:  
                error = np.std(y - model(x))  
                plt.fill_between(mx, model(mx) - 2*error, model(mx) + 2*error, color=colors[i], alpha=0.2, label='95% Confidence Interval')  
    plt.legend(loc="upper left")  
    plt.grid(True)  
    plt.tight_layout()  
    plt.show()  

plot_web_traffic(x, y)

def error(f, x, y):  
    return np.sqrt(np.mean((f(x) - y) ** 2))  

def root_mean_squared_error(f, x, y):  
    return np.sqrt(np.mean((f(x) - y) ** 2))  

def mean_absolute_percentage_error(f, x, y):  
    return np.mean(np.abs((y - f(x)) / y)) * 100  

def print_model_metrics(model, x, y, model_name="Model"):  
    mse = np.mean((model(x) - y)**2)  
    rmse = root_mean_squared_error(model, x, y)  
    mape = mean_absolute_percentage_error(model, x, y)  
    print(f"{model_name} Evaluation Metrics:")  
    print(f"  MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")  
    return mse, rmse, mape  

degree = 1  
f1 = np.poly1d(np.polyfit(x, y, degree))  
error1 = error(f1, x, y)  
print(f"{degree}st-degree polynomial error: {error1:.2f}")  
mse1, rmse1, mape1 = print_model_metrics(f1, x, y, f"{degree}st-Degree Polynomial")  
plot_web_traffic(x, y, [f1], title=f"{degree}st-Degree Polynomial Fit")  

degrees = [2, 3, 5, 10, 20]  
models = [f1]  
errors = [error1]  
metrics = [(mse1, rmse1, mape1)]  
for degree in degrees:  
    model = np.poly1d(np.polyfit(x, y, degree))  
    models.append(model)  
    err = error(model, x, y)  
    errors.append(err)  
    m, r, p = print_model_metrics(model, x, y, f"{degree}th-Degree Polynomial")  
    metrics.append((m, r, p))  
    print("-" * 40)  

plot_web_traffic(x, y, models, title="Comparison of Polynomial Models by Degree")  

plt.figure(figsize=(10, 6))  
plt.plot([1] + degrees, errors, 'o-')  
plt.title('Error Comparison by Polynomial Degree')  
plt.xlabel('Polynomial Degree')  
plt.ylabel('Error Value')  
plt.grid(True)  
plt.xticks([1] + degrees)  
plt.show()  

print("\nModel Performance Comparison:")  
print("Degree\tMSE\t\tRMSE\t\tMAPE(%)")  
print("-" * 50)  
for i, degree in enumerate([1] + degrees):  
    m, r, p = metrics[i]  
    print(f"{degree}\t{m:.2f}\t\t{r:.2f}\t\t{p:.2f}")  

inflection = int(3.5 * 7 * 24)  
xa, ya = x[:inflection], y[:inflection]  
xb, yb = x[inflection:], y[inflection:]  

fa = np.poly1d(np.polyfit(xa, ya, 1))  
fb = np.poly1d(np.polyfit(xb, yb, 1))  
total_error = error(fa, xa, ya) + error(fb, xb, yb)  
print(f"Total error of piecewise linear model: {total_error:.2f}")  

def piecewise_model(x_val):  
    if isinstance(x_val, np.ndarray):  
        return np.where(x_val < inflection, fa(x_val), fb(x_val))  
    else:  
        return fa(x_val) if x_val < inflection else fb(x_val)  

plot_web_traffic(x, y, [piecewise_model], title="Piecewise Linear Model")  

degrees = [1, 2, 3, 5, 10]  
models_b = []  
errors_b = []  
metrics_b = []  
for degree in degrees:  
    model = np.poly1d(np.polyfit(xb, yb, degree))  
    models_b.append(model)  
    err = error(model, xb, yb)  
    errors_b.append(err)  
    m, r, p = print_model_metrics(model, xb, yb, f"{degree}th-Degree Polynomial (Post-Inflection)")  
    metrics_b.append((m, r, p))  
    print("-" * 40)  

plot_web_traffic(xb, yb, models_b, title="Polynomial Models on Post-Inflection Data")  

test_size = 7 * 24  
train_size = len(xb) - test_size  
xtrain, ytrain = xb[:train_size], yb[:train_size]  
xtest, ytest = xb[train_size:], yb[train_size:]  

degrees = [1, 2, 3, 5, 10]  
train_errors = []  
test_errors = []  
best_test_error = float('inf')  
best_model = None  
best_degree = 0  

for degree in degrees:  
    model = np.poly1d(np.polyfit(xtrain, ytrain, degree))  
    train_err = error(model, xtrain, ytrain)  
    test_err = error(model, xtest, ytest)  
    train_errors.append(train_err)  
    test_errors.append(test_err)  
    if test_err < best_test_error:  
        best_test_error = test_err  
        best_model = model  
        best_degree = degree  

plt.figure(figsize=(10, 6))  
plt.plot(degrees, train_errors, 'o-', label='Training Error')  
plt.plot(degrees, test_errors, 's-', label='Test Error')  
plt.title('Training vs. Test Error by Polynomial Degree')  
plt.xlabel('Polynomial Degree')  
plt.ylabel('Error Value')  
plt.legend()  
plt.grid(True)  
plt.show()  

print(f"\nBest Model: {best_degree}th-degree polynomial")  
print(f"Best Test Error: {best_test_error:.2f}")  

final_degree = 3  
final_model = np.poly1d(np.polyfit(xb, yb, final_degree))  
final_error = error(final_model, xb, yb)  
print(f"Final Model Error: {final_error:.2f}")  
plot_web_traffic(x, y, [final_model], xlim=(0, x[-1] + 2*7*24), show_confidence=True, title="Final Forecast Model (3rd-Degree)")  

target_capacity = 100000  
equation = lambda x: final_model(x) - target_capacity  
start_guess = x[-1]  
hours_reached = fsolve(equation, start_guess)[0]  
weeks_reached = hours_reached / (7 * 24)  
weeks_remaining = (hours_reached - x[-1]) / (7 * 24)  

print(f"Estimated time to reach {target_capacity} hits/hour: {weeks_reached:.2f} weeks")  
print(f"Time remaining until capacity limit: {weeks_remaining:.2f} weeks")  

residuals = yb - final_model(xb)  
std_dev = np.std(residuals)  
confidence_levels = [68, 95, 99.7]  
confidence_intervals = []  

for cl in confidence_levels:  
    z = 1  # for simplicity  
    margin_of_error = z * std_dev  
    upper_eq = lambda x: final_model(x) + margin_of_error - target_capacity  
    lower_eq = lambda x: final_model(x) - margin_of_error - target_capacity  
    upper_hours = fsolve(upper_eq, start_guess)[0]  
    lower_hours = fsolve(lower_eq, start_guess)[0]  
    confidence_intervals.append((lower_hours / (7*24), upper_hours / (7*24)))  

plt.figure(figsize=(10, 6))  
plt.axhline(y=target_capacity, color='r', linestyle='-', label='Capacity Limit')  
plt.axvline(x=weeks_reached, color='g', linestyle='-', label='Forecast Time')  
for i, (lw, uw) in enumerate(confidence_intervals):  
    plt.axvspan(lw, uw, alpha=0.2, label=f'{confidence_levels[i]}% Confidence Interval')  
plt.title('Confidence Intervals for Capacity Limit Forecast')  
plt.xlabel('Time (Weeks)')  
plt.ylabel('Hits/Hour')  
plt.grid(True)  
plt.legend()  
plt.tight_layout()  
plt.show()
