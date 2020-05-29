import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense,Activation

veriseti=pd.read_csv("Breast_cancer_data.csv")
print(veriseti.info())
#veri setinde kayıp ya da sayısal olmayan verilerin olup olmadığının kontrolü için bu kod bloğu oluşturuldu.
import re
kayip_veriler=[]
sayisal_olmayan_veriler=[]

for oznitelik in veriseti:
    essiz_deger=veriseti[oznitelik].unique()
    print("'{}' özniteliğine ait (unique) veriler : {}".format(oznitelik, essiz_deger.size))
    if (essiz_deger.size>10):
        print("10 adet essizdeger listele")
        print(essiz_deger[0:10])
        print("\n---------\n")
        
        if(True in pd.isnull(essiz_deger)):
            s="{} özniteliğe ait kayıp veriler {}".format(oznitelik, pd.isnull(veriseti[oznitelik]).sum())
            kayip_veriler.append(s)
            
            for i in range (1,np.prod(essiz_deger.shape)):
                if (re.match('nan',str(essiz_deger[i]))):
                    break
                if not (re.search('(^\d+\.?\d*$)|(^\d*\.?\d+$)', str(essiz_deger[i]))):
                    sayisal_olmayan_veriler.append(oznitelik)
                    break
print("Kayıp veriye sahip öznitelikler:\n{}\n\n".format(kayip_veriler))
print("Sayısal olmayan veriye sahip öznitelikler:\n{}".format(sayisal_olmayan_veriler))
# %%
y=veriseti.iloc[:,5:6].values
x_data=veriseti.iloc[:,0:5].values

# %% öznitelik ölçeklendirme
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

# %% veri setinin eğitim ve  test kümelerine ayrılması

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=1) 

# %%keras
siniflandirici=Sequential()

#ilk gizli katman
siniflandirici.add(Dense(output_dim=16, init='uniform',activation='relu', input_dim=5))

#ikinci gizli katman 
siniflandirici.add(Dense(output_dim=32, init='uniform',activation='relu'))

#cikti katmanı
siniflandirici.add(Dense(output_dim=1, init='uniform',activation='sigmoid'))

#derleme işlemi
siniflandirici.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


egitim=siniflandirici.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test))

import sklearn.metrics as metrics
y_pred=siniflandirici.predict_classes(x_test)
# %%
plt.plot(egitim.history['loss'])
plt.plot(egitim.history['val_loss'])
plt.title('model loss')
plt.xlabel('epochs')
plt.ylabel('loss values')
plt.legend(['loss','val_loss'],loc='lower right')
plt.show()

# %%hata matrisi
from sklearn.metrics import confusion_matrix, classification_report
hm=confusion_matrix(y_test, y_pred)
print(classification_report(y_test,y_pred))

# %% roc ve auc

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

#%% ağırlıklar
for i in siniflandirici.layers:
    ilk_gizli_katman=siniflandirici.layers[0].get_weights()
    ikinci_gizli_katman=siniflandirici.layers[1].get_weights()
    cikti_katman=siniflandirici.layers[2].get_weights()
    
