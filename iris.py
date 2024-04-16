import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Функция по удалению выбросов
def IQR(df, str):
    Q1, Q2, Q3 = df[str].quantile([0.25, 0.5, 0.75])
    IQR = Q3 - Q1
    for i in df[str].index:
        if df.loc[i, str] <= Q1 - 3 * IQR or df.loc[i, str] >= Q3 + 3 * IQR:
            print(i, train_df.loc[i, str])
            df = df.drop([i])

    df.dropna(inplace=True)

# Функция для нахождения мат. ожидания для заданного вида и параметра
def nu(df, species, parameter):
    nu = df[df['Species'] == species][parameter].mean()
    return nu


def likehood(df, nu_species, cov):
    x = 0
    for i in df.index:
        nu_value = []
        feature_value = df.loc[i]
        feature_value = feature_value[1:5]
        if df.loc[i, 'Species'] == 'Iris-setosa':
            nu_value = nu_species[0]
        elif df.loc[i, 'Species'] == 'Iris-versicolor':
            nu_value = nu_species[1]
        else:
            nu_value = nu_species[2]

        species = df['Species']

        Theta = len(df[df['Species'] == species]) / n
        feature_value = feature_value.astype(float)
        mean = feature_value.values
        x = Theta * np.random.multivariate_normal(mean, cov.values, size=150)
    return x


# Считывание данных из файла
df = pd.read_excel('Iris.xlsx')

# Сколько строк каждого вида должно попасть в обучающую выборку
cnt_setosa = len(df[df['Species'] == 'Iris-setosa']) * 0.8
cnt_versicolor = len(df[df['Species'] == 'Iris-versicolor']) * 0.8
cnt_virginica = len(df[df['Species'] == 'Iris-virginica']) * 0.8

# Создание обучающей выборки
train_df = pd.DataFrame(columns=df.columns)
test_df = pd.DataFrame(columns=df.columns)

for i in df.index:
    if df.loc[i, 'Species'] == 'Iris-setosa' and cnt_setosa > 0:
        cnt_setosa -= 1
        train_df.loc[len(train_df.index)] = df.iloc[i]
    elif df.loc[i, 'Species'] == 'Iris-versicolor' and cnt_versicolor > 0:
        cnt_versicolor -= 1
        train_df.loc[len(train_df.index)] = df.iloc[i]
    elif df.loc[i, 'Species'] == 'Iris-virginica' and cnt_virginica > 0:
        cnt_virginica -= 1
        train_df.loc[len(train_df.index)] = df.iloc[i]
    else:
        test_df.loc[len(test_df)] = df.iloc[i]

# Очистите выборки от выбросов по методу  IQR (Inter Quartile Range)
IQR(train_df, 'SepalLengthCm')
IQR(train_df, 'SepalWidthCm')
IQR(train_df, 'PetalLengthCm')
IQR(train_df, 'PetalWidthCm')

print(train_df.shape)

# Построение точечных графиков
sns.pairplot(train_df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']], hue='Species')
plt.show()

# Построение гистрограм
plt.figure(figsize=(20, 48))
plot_number = 0
for feature_name in ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']:
    for target_name in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']:
        plot_number += 1
        plt.subplot(4, 3, plot_number)
        plt.hist(train_df[train_df['Species'] == target_name][feature_name])
        plt.title(target_name)
        plt.xlabel('cm')
        plt.ylabel(feature_name[:-4])
plt.show()

# Оценивание параметров вероятности каждого вида
n = len(df)
Theta_setosa = len(df[df['Species'] == 'Iris-setosa']) / n
Theta_versicolor = len(df[df['Species'] == 'Iris-versicolor']) / n
Theta_virginica = len(df[df['Species'] == 'Iris-virginica']) / n
print(Theta_setosa, Theta_versicolor, Theta_virginica)

# Оценивание параметров мат. ожидания для каждого вида
nu_species = []
for feature_name in ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']:
    nu_value = []
    for target_name in ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']:
        nu_value.append(nu(train_df, target_name, feature_name))
    nu_species.append(nu_value)
print(nu_species)


# Выделелние из таблицы только этих четырех столбцов, чтобы найти ковариацию
feature_df = train_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
cov = feature_df.cov()
print(cov)

# Вычислите правдоподобие данных на обучающей и тестовой выборке. Вероятность одного экземпляра по этой модели
likehood_value_train = likehood(train_df, nu_species, cov)
likehood_value_test = likehood(test_df, nu_species, cov)
print(likehood_value_train, likehood_value_test)


#  Обучите градиентным спуском линейную модель вида:
fit_output = stats.linregress(train_df['PetalLengthCm'].values.tolist(), train_df['PetalWidthCm'].values.tolist())
print(fit_output.slope, fit_output.intercept)

#  Рисуем график с точками и линией регрессии:
plt.plot(train_df[['PetalLengthCm']], train_df[['PetalWidthCm']],'o', label='Data')
plt.plot(train_df[['PetalLengthCm']], fit_output.intercept + fit_output.slope*train_df[['PetalLengthCm']], 'r', linewidth=3, label='Linear regression line')
plt.ylabel('petal width (cm)')
plt.xlabel('petal length (cm)')
plt.legend()
plt.show()