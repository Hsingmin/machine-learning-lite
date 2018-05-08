
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame(np.random.randn(4,5), columns=['A', 'B', 'C', 'D', 'E'])


df['ROW_SUM'] = df.apply(lambda x: x.sum(), axis=1)
df.loc['COL_SUM'] = df.apply(lambda x: x.sum())

print(df)

print('df[len(df)-1] = ')
print(df[2:3])

plt.bar([2,4,6,8,10], [8,6,2,5,6], label="Example", color='g')
plt.legend()
plt.xlabel('bar number')
plt.ylabel('bar height')

plt.title('Epic Graph')

plt.show()
























