import pandas as pd
df=pd.read_csv(r"C:\Users\abhis\OneDrive\Desktop\project folder\Full_News_Impact_Sectoral_Movement_2014_2024.csv")
print(df)
df.drop_duplicates()
df.info()
df.describe()
df.head()
df.drop_duplicates(inplace=True)
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv()
