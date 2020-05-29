import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
veriseti=pd.read_csv("Breast_cancer_data.csv")
print(veriseti.info())

# %%
y=veriseti.diagnosis.values
x_data=veriseti.drop(["diagnosis"],axis=1)

# %% normalization
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=1)

#%%
from sklearn.neural_network import MLPClassifier

mlpc_model=MLPClassifier()
mlpc_model.fit(x_train,y_train)

# %%
print("score:",mlpc_model.score(x_test,y_test))

# %% model tuning

from sklearn.model_selection import GridSearchCV

mlpc_params={"alpha":[1,0.1,0.01,0.005],
             "hidden_layer_sizes":[(10,10),(100,100,100),(3,5)]}
# mlpc=MLPClassifier() #ilk çalışan
mlpc=MLPClassifier(activation="logistic",solver="lbfgs") #2. çalışan
mlpc_cv_model=GridSearchCV(mlpc,mlpc_params,cv=10,n_jobs=-1,verbose=2).fit(x_train,y_train)

# %%
print(mlpc_cv_model.best_params_)
# %%

mlpc_tuned=MLPClassifier(alpha=0.1,hidden_layer_sizes=(3,5),activation="logistic",solver="lbfgs").fit(x_train,y_train)
# %%
print("score:",mlpc_tuned.score(x_test,y_test))
#%%
import sklearn.metrics as metrics
y_pred=mlpc_model.predict(x_test)

#%%
from sklearn.metrics import confusion_matrix, classification_report
hm=confusion_matrix(y_test, y_pred)
print(classification_report(y_test,y_pred))

#%%

from sklearn.metrics import roc_curve, auc
ypo,dpo, esikDeger=roc_curve(y_test, y_pred)
aucDegeri =auc(ypo,dpo)

plt.figure()
plt.plot(ypo,dpo, label='AUC %0.2f' % aucDegeri)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('yanlış pozitif oranı(YPO)')
plt.ylabel('dogru pozitif oranı(DPO)')
plt.title('roc egrisi')
plt.legend(loc="best")
plt.show()