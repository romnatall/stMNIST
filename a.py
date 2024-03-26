import streamlit as st
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from M import *
from streamlit_drawable_canvas import st_canvas


# Загрузка модели PyTorch
@st.cache_resource
def load_model():
    model = torch.load('Adam.pt')
    model.eval()
    return model

model = load_model()
# Функция для обработки нарисованной цифры и предсказания с помощью модели
def predict_digit(image):
    # Преобразование изображения в тензор
    image_tensor = torch.tensor(np.array(image))
    image_tensor = image_tensor.view(1, 1, 28, 28).float() / 255.0  # нормализация и изменение размерности
    #st.write(image_tensor)
    # Предсказание с помощью модели
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output).item()
        probabilities = torch.softmax(output, dim=1).squeeze().numpy()
    
    return predicted_class, probabilities

# Отображение интерфейса Streamlit
st.title("Распознавание рукописных цифр")

# Холст для рисования
canvas = st_canvas(key='canvas', stroke_width=40, stroke_color="rgb(255, 255, 255)", background_color='black', height = 280, width = 280)

# Кнопка для предсказания цифры
if st.button('Распознать'):
    # Получение нарисованного изображения
    digit_image = np.array(canvas.image_data)
    digit_image = Image.fromarray(digit_image[:, :, 0])  # удаление альфа-канала (если есть)
    digit_image = digit_image.resize((28, 28))  # изменение размерности
    # Предсказание цифры
    predicted_class, probabilities = predict_digit(digit_image)
    
    # Отображение результатов
    st.write(f"Предсказанная цифра: {predicted_class}")
    

    digits = list(range(10))
    fig = plt.figure(figsize=(10, 5))
    plt.bar(digits, probabilities)
    plt.xlabel('Цифра')
    plt.ylabel('Вероятность')
    plt.xticks(digits)
    st.pyplot( fig )





