from sklearn import svm

import pandas as pd
df = pd.read_csv(csv_file)
saved_column = df.age #you can also use df['column_name']

print(saved_column)
# clf = svm.SVC()
# clf.fit(X,y)
