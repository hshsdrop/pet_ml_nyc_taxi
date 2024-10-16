# [Manhattan Taxi Trip](https://front-ny-taxi.streamlit.app/)
<img src = "https://raw.githubusercontent.com/hshsdrop/pet_ml_nyc_taxi/main/readme_img/example.png" width = "800"  align = "center" />

Веб-приложение для предсказания продолжительности поездки на такси в Манхэттене и построения оптимального маршрута на карте.

В этом пет-проекте используется модель CatBoost для предсказания продолжительности поездки, FastAPI - для бэкенд-сервиса и Streamlit - для фронтенд-сервиса. Docker-compose оркестрирует два сервиса и позволяет им взаимодействовать друг с другом.

### Организация репозитория 

    ├── backend              <- репозиторий бэкенд-сервиса.
    │
    ├── frontend             <- репозиторий фронтенд-сервиса.
    │
    ├── readme_img           <- изображения для README.
    │
    ├── .gitignore           <- файл .gitignore.
    │                     
    ├── docker-compose.yml   <- файл docker-compose.
    │
    └── README.md            <- Описание пет-проекта.

### Локальная установка
Чтобы запустить приложение локально (требуется Docker и docker-compose), скопируйте репозиторий и выполните:

    docker-compose build
    docker-compose up

Streamlit UI - http://localhost:8501

Документация FastAPI - http://localhost:8000/docs 

### Реализация в вебе
- [Страница](https://front-ny-taxi.streamlit.app/) с интерфейсом проекта развернута на Streamlit.share. 
- Бэкенд(FastAPI) развернут на платформе platform.sh[(документация)](https://main-bvxea6i-k4o37wg5i6vqs.de-2.platformsh.site/docs).

### О проекте
#### 1. Подготовка данных
Датасет взят из [соревнования](https://www.kaggle.com/c/nyc-taxi-trip-duration).

##### 1.1 Очистка данных
Данные были очищены от несущественных поездок, которые начинались или заканчивались за пределами Манхэттена. Поездки на такси были отфильтрованы по разделительной линии, которая была построена вдоль 13 мостов, ведущих в Манхэттен. 

<p align="center">
  <img src="https://raw.githubusercontent.com/hshsdrop/pet_ml_nyc_taxi/main/readme_img/start_end_1.png" width = "650" />
  <img src="https://raw.githubusercontent.com/hshsdrop/pet_ml_nyc_taxi/main/readme_img/start_end_2.png" width = "650" />
  <img src="https://raw.githubusercontent.com/hshsdrop/pet_ml_nyc_taxi/main/readme_img/start_end_3.png" width = "650" />
  <img src="https://raw.githubusercontent.com/hshsdrop/pet_ml_nyc_taxi/main/readme_img/clustering.png" width = "450" />
</p>

Район Нью-Йорка Манхэттен был выбран из-за небольшой поисковой области, т.к. быстрее ищется оптимальный маршрут (у сервиса есть ограничения по производительности).

##### 1.2 Предобработка признаков

Признаки: 

    - manh_length                  <- Оценка манхэттенского расстояния между началом и концом поездки.
    - route                        <- Маршрут поездки основан на сформированных кластерах. 
    - временные признаки           <- час, день недели, высокий трафик (двоичный), и т.д.
    - признаки на основе координат <- широта/долгота.

#### 2. Обучение и предсказание
##### 2.1 Обучение модели
Для обучения использовалась модель CatBoost.

Функция потерь MSLE, т.к. штрафует за недостаточный прогноз больше, чем за преувеличенный.  

Для поиска гиперпараметров использовалась библиотека Optuna.

##### 2.2 Оценка качества модели

| Данные | MSLE | MAE         | $R^2$|
| ---    | ---  | ---         | ---  |
| Train  | 0.08 | 2.36 минуты | 0.83 |
| Test   | 0.13 | 2.48 минуты | 0.74 |

#### 3. Схема работы приложения

<p align="center">
  <img src="https://raw.githubusercontent.com/hshsdrop/pet_ml_nyc_taxi/main/readme_img/hiw_main.png" width = "650" />
</p>