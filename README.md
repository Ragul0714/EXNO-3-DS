## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
<img width="420" height="490" alt="image" src="https://github.com/user-attachments/assets/3e533c9e-b1ad-4398-b782-1dd363623795" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="397" height="257" alt="image" src="https://github.com/user-attachments/assets/fad0085c-be46-43cf-95f8-2cbbb2fc7f41" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="487" height="492" alt="image" src="https://github.com/user-attachments/assets/d0a6d5d0-51b3-4237-b362-146329fd28bc" />

```
 le=LabelEncoder()
 dfc=df.copy()
 dfc['ord_2']=le.fit_transform(dfc['ord_2'])
 dfc
```
<img width="475" height="622" alt="image" src="https://github.com/user-attachments/assets/32c55091-f42a-4719-bd9f-48ebe73e1d27" />

```
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

ohe = OneHotEncoder(sparse_output=False)  # use sparse_output instead of sparse
df2 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]), 
                   columns=ohe.get_feature_names_out(["nom_0"]))
df2 = pd.concat([df2, enc], axis=1)
print(df2)
```
<img width="668" height="226" alt="image" src="https://github.com/user-attachments/assets/beedcfc8-293c-4481-b976-5804c5aa80b8" />

```
 pd.get_dummies(df2,columns=["nom_0"])
```
<img width="918" height="388" alt="image" src="https://github.com/user-attachments/assets/4f393c6a-d1ab-4040-98d1-4e729804a21e" />

```
pip install --upgrade category_encoders
```
```
 from category_encoders import BinaryEncoder
 df=pd.read_csv("data.csv")
```
<img width="550" height="385" alt="image" src="https://github.com/user-attachments/assets/f6fe5aba-ace1-40f8-ad80-c6d64a88cdc1" />

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
<img width="558" height="393" alt="image" src="https://github.com/user-attachments/assets/fa08529a-2093-4a12-862f-c294f79ee2b1" />

```
dfb=pd.concat([df,nd],axis=1)
dfb
```
<img width="768" height="397" alt="image" src="https://github.com/user-attachments/assets/560bd604-7162-44d7-9d30-f337db75e54a" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="618" height="387" alt="image" src="https://github.com/user-attachments/assets/1710f1f8-36cf-4d75-93f6-57c4ba98197d" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
<img width="661" height="465" alt="image" src="https://github.com/user-attachments/assets/f62edac9-77ef-4777-8415-5f6bfa20eaeb" />

```
 df.skew()
```
<img width="388" height="237" alt="image" src="https://github.com/user-attachments/assets/41cd761b-7f14-435f-8e8a-ff0d73d61d73" />

```
np.log(df["Highly Positive Skew"])
```
<img width="372" height="497" alt="image" src="https://github.com/user-attachments/assets/00cc2c6e-718b-4936-828a-49c4a38c42ec" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="363" height="488" alt="image" src="https://github.com/user-attachments/assets/1c516fc9-5c31-469b-a2c2-4b253202afad" />

```
np.sqrt(df["Highly Positive Skew"])
```

<img width="361" height="488" alt="image" src="https://github.com/user-attachments/assets/c2be7a0a-8499-4806-80a0-b9af554b8a20" />

```
 np.square(df["Highly Positive Skew"])
```

<img width="387" height="493" alt="image" src="https://github.com/user-attachments/assets/7738a118-ea06-4d90-8b2c-f233936ce50a" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

<img width="645" height="502" alt="image" src="https://github.com/user-attachments/assets/a5befa63-0cef-4285-b432-bc28f4123c36" />

```
df.skew()
```

<img width="411" height="257" alt="image" src="https://github.com/user-attachments/assets/6b60767f-b284-43f0-a336-53d08bb79457" />

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

<img width="501" height="298" alt="image" src="https://github.com/user-attachments/assets/a488bca9-db72-4209-b44b-7a45d4d628d3" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate  Negative Skew"]])
df
```
<img width="1540" height="486" alt="image" src="https://github.com/user-attachments/assets/d51a6283-57fc-4d76-b622-f9874838c846" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="672" height="480" alt="image" src="https://github.com/user-attachments/assets/7c951808-8e06-48de-a910-5452cb64039b" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
<img width="673" height="477" alt="image" src="https://github.com/user-attachments/assets/00ddd0df-b037-4c94-a0ab-c5b80dc53dc8" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="667" height="478" alt="image" src="https://github.com/user-attachments/assets/90c49ad2-69a2-42c3-acc9-6ca912faab1b" />


```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
<img width="667" height="487" alt="image" src="https://github.com/user-attachments/assets/1bcaacae-0abf-4859-baf8-90c5b84a9056" />


```
dt=pd.read_csv("titanic_dataset.csv")
dt
```
<img width="1285" height="458" alt="image" src="https://github.com/user-attachments/assets/f354ae7b-41dd-485e-b3ae-6d7860299cc0" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 dt["Age_1"]=qt.fit_transform(dt[["Age"]])
 sm.qqplot(dt['Age'],line='45') 
 plt.show()

```
<img width="666" height="487" alt="image" src="https://github.com/user-attachments/assets/d8ff9e07-7800-437f-94fc-18353f2ef4cf" />

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
<img width="653" height="478" alt="image" src="https://github.com/user-attachments/assets/c9fd4e6f-971e-47f4-b4a3-584ff7a631f8" />



# RESULT:
  Thus the given data, Feature Encoding, Transformation process and save the data to a file
  was performed successfully

       
