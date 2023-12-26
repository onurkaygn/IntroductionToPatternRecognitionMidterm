#pip install pandas scikit-learn komutunu kullanarak projeye eksik paketler eklenebilir.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder



#İlk olarak Drive'a istek atarak yapmaya çalıştım fakat olmadı. Sonrasında bu şekilde klasik olarak okudum.
csvYolu = r"otu.csv"
veriSeti = pd.read_csv(csvYolu, dtype=str)


X = veriSeti.iloc[1:, :].T
y = veriSeti.iloc[:1, :].T.squeeze()
le = LabelEncoder()
y = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=4)

#İlk olarak rassal orman algoritmasını kullanmayı denedim fakat proje çalışmadı sonrasında SVC kullandım.
model = make_pipeline(StandardScaler(), SVC(kernel='linear', random_state=0, probability=True))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


Matrix = confusion_matrix(y_test, y_pred)
classificationReport = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
sensitivity = Matrix[0, 0] / (Matrix[0, 0] + Matrix[0, 1])
specificity = Matrix[1, 1] / (Matrix[1, 0] + Matrix[1, 1])
ROC_AUC = roc_auc_score(y_test, y_pred)

print("Matris:\n", Matrix)
print("Siniflandirma Raporu:\n", classificationReport)
print("Tutarlilik Değeri (0 ile 1 arasinda yuzde degeri ->):", accuracy, "\n")
print('Hassasiyet : ', sensitivity)
print('Özgüllük : ', specificity)
print('AUC Değeri : {:.4f}'.format(ROC_AUC))