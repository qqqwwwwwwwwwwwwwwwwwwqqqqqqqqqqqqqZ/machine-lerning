import pandas as pd
df = pd.read_csv("train.csv")
df.drop(["id","has_photo","has_mobile","followers_count","city","last_seen","bdate","education_status","occupation_name"],axis=1,inplace = True)
print(df.info())
def fill_sex(sex):
    if sex == 2:
        return 0 
    return 1
df["sex"] = df["sex"].apply(fill_sex)
print(df["sex"])
print(df["education_form"].value_counts())
df["education_form"].fillna("Full-time",inplace=True)
#створюємо категоріальні змінні для кожної форми навчання людини
df[list(pd.get_dummies(df["education_form"]).columns)] = pd.get_dummies(df["education_form"])
df.drop("education_form", axis=1, inplace=True)
print(df.info())
def listlangs(langs):
    return langs.split(";")
df["langs"] = df["langs"].apply(listlangs)
print(df["langs"])
df["namberoflangs"] = df["langs"].apply(len)
df.drop("langs", axis=1, inplace=True)
print(df["life_main"])
def lifemaintint(stutus):
    if stutus != "False":
        return int(stutus)
    else:return 0 
df["life_main"] = df["life_main"].apply(lifemaintint)
print(df["people_main"])
df["people_main"]=df["people_main"].apply(lifemaintint)
print(df.info())
print(df["occupation_type"].value_counts())
df["occupation_type"].fillna("university",inplace = True)
def fill_occupation(wer01):
    if wer01 == "university":return 0
    else: return 1
df["occupation_type"]=df["occupation_type"].apply(fill_occupation)
print(df["career_start"])
def career_int(career):
    if career != "False":
        return int(career)
    else: return 2024
df["career_start"]=df["career_start"].apply(career_int)
df["career_end"]=df["career_end"].apply(career_int)
def faind_dureihen(year):
    return 2024 - year 
df["career_dureihen"] = df["career_start"].apply(faind_dureihen)
df["not_work"] = df["career_end"].apply(faind_dureihen)
df.drop(["career_start","career_end"],axis=1,inplace=True)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#беремо дані, на основі яких ми будемо прогнозувати результат (все крім result)
x = df.drop('result', axis = 1)
#беремо дані, які будуть прогнозуватися (result)
y = df['result']

#ділимо дані на 2 набори: тренувальний та тестувальний
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

#Стандартизуємо дані(як тренувальні, так і тестувальні)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#створюємо математичну модель k-найближчих сусідів та "згодовуємо" їй тренувальні дані
our_model = KNeighborsClassifier(n_neighbors = 499)
our_model.fit(x_train, y_train)

#просимо модель спрогнозувати результати на основі тестових даних
y_pred = our_model.predict(x_test)
print('Відсоток правильно передбачених результатів:', accuracy_score(y_test, y_pred) * 100)