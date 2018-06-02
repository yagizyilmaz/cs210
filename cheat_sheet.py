#cheat sheet

dataset.groupby('Primary Type').ID.nunique()
ser = dataset.groupby('Primary Type').ID.nunique()
ser = ser.reset_index()
ser.columns = [c.replace('ID', 'Count') for c in ser.columns]
ser.sort_values(['Count'], ascending=[False], inplace=True)
x_label = ser["Primary Type"].values
y_val = ser["Count"].values
x_val = list(range(1, 34))
plt.figure(figsize=(8, 6))
plt.bar(x_val, y_val, align='center')
plt.xticks(x_val, x_label, rotation='vertical')
plt.show()