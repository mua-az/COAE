import pandas
import numpy
import matplotlib.pyplot as plt

data = pandas.DataFrame(
    {
        "column1" : numpy.random.rand(50),
        "column2" : numpy.random.rand(50) * 10
    }
)

print(data.head())

plt.scatter( data["column1"] , data["column2"])
plt.xlabel("Column 1")
plt.ylabel("Column 2")
plt.title("Scatter plot")
plt.show()