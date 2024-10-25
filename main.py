import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import plotly.express as px


# Функция для загрузки данных
def load_data():
    # Пример загрузки данных из CSV файла
    data = pd.read_csv('data.csv')
    return data


# Функция для предобработки данных
def preprocess_data(data):
    # Пример предобработки: заполнение пропущенных значений
    data.fillna(data.mean(), inplace=True)
    return data


# Функция для обучения модели
def train_model(data):
    # Пример разделения данных на обучающую и тестовую выборки
    X = data.drop('отток', axis=1)
    y = data['отток']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Пример обучения модели логистической регрессии
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Пример оценки модели
    y_pred = model.predict(X_test)
    st.write(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    st.write(classification_report(y_test, y_pred))

    return model


# Функция для визуализации данных
def visualize_data(data):
    # Пример визуализации данных с использованием Plotly
    fig = px.scatter(data, x='объем_перевозок', y='денежные_средства', color='отток',
                     title='Объем перевозок vs Денежные средства')
    st.plotly_chart(fig)


# Основная функция приложения
def main():
    st.title('Прогнозирование оттока клиентов ОАО «РЖД»')

    # Загрузка данных
    data = load_data()

    # Предобработка данных
    data = preprocess_data(data)

    # Визуализация данных
    visualize_data(data)

    # Обучение модели
    model = train_model(data)

    # Пример прогнозирования оттока для нового клиента
    st.subheader('Прогнозирование оттока для нового клиента')
    new_client = {}
    for feature in data.columns[:-1]:
        new_client[feature] = st.number_input(f'Введите значение для {feature}')

    if st.button('Сделать прогноз'):
        new_client_df = pd.DataFrame([new_client])
        prediction = model.predict(new_client_df)
        if prediction[0] == 1:
            st.error('Клиент, скорее всего, уйдет')
        else:
            st.success('Клиент, скорее всего, останется')


if __name__ == '__main__':
    main()
