import pickle
from flask import Flask, request, render_template_string
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Загрузка модели
with open('best_model_gb_r_lab.pkl', 'rb') as f:
    model = pickle.load(f)

# Загрузка маппинга категорий
with open('category_mapping.pkl', 'rb') as f:
    category_mapping = pickle.load(f)

# Создание LabelEncoders для каждой категории
label_encoders = {}
for feature, classes in category_mapping.items():
    le = LabelEncoder()
    le.classes_ = np.array(classes)
    label_encoders[feature] = le

def safe_float(value, default=0.0):
    try:
        return float(value)
    except ValueError:
        return default

def safe_int(value, default=0):
    try:
        return int(value)
    except ValueError:
        return default

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_price = None
    form_data = {}

    if request.method == 'POST':
        try:
            # Получение данных из формы
            form_data['year'] = safe_int(request.form['year'])
            form_data['make'] = request.form['make']
            form_data['model'] = request.form['model']
            form_data['condition'] = request.form['condition']
            form_data['consumer_rating'] = safe_float(request.form['consumer_rating'], default=4.0)  # Значение по умолчанию
            form_data['consumer_reviews'] = safe_int(request.form['consumer_reviews'])
            form_data['state'] = request.form['state']
            form_data['deal_type'] = request.form['deal_type']
            form_data['comfort_rating'] = safe_float(request.form['comfort_rating'], default=4.0)  # Значение по умолчанию
            form_data['interior_design_rating'] = safe_float(request.form['interior_design_rating'], default=4.0)  # Значение по умолчанию
            form_data['performance_rating'] = safe_float(request.form['performance_rating'], default=4.0)  # Значение по умолчанию
            form_data['value_for_money_rating'] = safe_float(request.form['value_for_money_rating'], default=4.0)  # Значение по умолчанию
            form_data['exterior_styling_rating'] = safe_float(request.form['exterior_styling_rating'], default=4.0)  # Значение по умолчанию
            form_data['reliability_rating'] = safe_float(request.form['reliability_rating'], default=4.0)  # Значение по умолчанию
            form_data['exterior_color'] = request.form['exterior_color']
            form_data['interior_color'] = request.form['interior_color']
            form_data['drivetrain'] = request.form['drivetrain']
            form_data['min_mpg'] = safe_float(request.form['min_mpg'])
            form_data['max_mpg'] = safe_float(request.form['max_mpg'])
            form_data['fuel_type'] = request.form['fuel_type']
            form_data['transmission'] = request.form['transmission']
            form_data['engine'] = request.form['engine']
            form_data['mileage'] = safe_float(request.form['mileage'])
            form_data['age'] = safe_int(request.form['age'])
            form_data['price'] = safe_float(request.form['price'])
            form_data['seller_type'] = request.form['seller_type']
            form_data['seller_rating'] = safe_float(request.form['seller_rating'], default=4.0)  # Значение по умолчанию
            form_data['seller_reviews'] = safe_int(request.form['seller_reviews'])
            form_data['engine_hp'] = safe_float(request.form['engine_hp'])

            print("Form Data Received:", form_data)  # Отладочный вывод

            # Подготовка данных для модели
            input_data = []
            for feature in ['make', 'model', 'condition', 'state', 'deal_type', 'exterior_color', 'interior_color',
                            'drivetrain', 'fuel_type', 'transmission', 'engine', 'seller_type']:
                le = label_encoders.get(feature)
                if le and form_data[feature] in le.classes_:
                    input_data.append(le.transform([form_data[feature]])[0])
                else:
                    input_data.append(-1)

            input_data.extend([
                form_data['year'], form_data['consumer_rating'],
                form_data['consumer_reviews'], form_data['comfort_rating'],
                form_data['interior_design_rating'], form_data['performance_rating'],
                form_data['value_for_money_rating'], form_data['exterior_styling_rating'],
                form_data['reliability_rating'], form_data['min_mpg'], form_data['max_mpg'],
                form_data['mileage'], form_data['age'], form_data['price'],
                form_data['seller_rating'], form_data['seller_reviews'],
                form_data['engine_hp']
            ])

            input_data = np.array([input_data])
            print("Input Data for Prediction:", input_data)  # Отладочный вывод

            # Прогнозирование цены
            predicted_price = model.predict(input_data)[0]
            print("Predicted Price:", predicted_price)  # Отладочный вывод

        except Exception as e:
            return f"Error occurred: {e}"

    return render_template_string('''
    <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f4f4f4; }
                header { background-color: #333; padding: 20px; text-align: center; color: white; }
                .container { width: 60%; margin: auto; padding: 20px; background: white; box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); }
                form { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                label { display: block; font-weight: bold; margin-bottom: 5px; }
                input[type="text"] { padding: 10px; margin-bottom: 10px; width: 100%; }
                input[type="submit"] { padding: 10px 20px; background-color: #007bff; color: white; border: none; cursor: pointer; grid-column: span 2; }
                input[type="submit"]:hover { background-color: #0056b3; }
                h1, h2 { text-align: center; }
                .result { background-color: #e0e0e0; padding: 15px; text-align: center; font-size: 1.5em; margin-top: 20px; }
            </style>
        </head>
        <body>
            <header>
                <h1>Прогнозирование Цены Автомобиля</h1>
            </header>
            <div class="container">
                <form method="post">
                    <label>Year (Год выпуска): <input type="text" name="year"></label>
                    <label>Make (Производитель): <input type="text" name="make"></label>
                    <label>Model (Модель): <input type="text" name="model"></label>
                    <label>Condition (Состояние): <input type="text" name="condition"></label>
                    <label>Consumer Rating (Рейтинг потребителей): <input type="text" name="consumer_rating" placeholder="4.0 по умолчанию"></label>
                    <label>Consumer Reviews (Отзывы потребителей): <input type="text" name="consumer_reviews"></label>
                    <label>State (Штат): <input type="text" name="state"></label>
                    <label>Deal Type (Тип сделки): <input type="text" name="deal_type"></label>
                    <label>Comfort Rating (Рейтинг комфорта): <input type="text" name="comfort_rating" placeholder="4.0 по умолчанию"></label>
                    <label>Interior Design Rating (Рейтинг внутреннего дизайна): <input type="text" name="interior_design_rating" placeholder="4.0 по умолчанию"></label>
                    <label>Performance Rating (Рейтинг производительности): <input type="text" name="performance_rating" placeholder="4.0 по умолчанию"></label>
                    <label>Value For Money Rating (Соотношение цены и качества): <input type="text" name="value_for_money_rating" placeholder="4.0 по умолчанию"></label>
                    <label>Exterior Styling Rating (Рейтинг внешнего стиля): <input type="text" name="exterior_styling_rating" placeholder="4.0 по умолчанию"></label>
                    <label>Reliability Rating (Рейтинг надежности): <input type="text" name="reliability_rating" placeholder="4.0 по умолчанию"></label>
                    <label>Exterior Color (Цвет кузова): <input type="text" name="exterior_color"></label>
                    <label>Interior Color (Цвет интерьера): <input type="text" name="interior_color"></label>
                    <label>Drivetrain (Привод): <input type="text" name="drivetrain"></label>
                    <label>Min MPG (Минимальный расход топлива): <input type="text" name="min_mpg"></label>
                    <label>Max MPG (Максимальный расход топлива): <input type="text" name="max_mpg"></label>
                    <label>Fuel Type (Тип топлива): <input type="text" name="fuel_type"></label>
                    <label>Transmission (Трансмиссия): <input type="text" name="transmission"></label>
                    <label>Engine (Двигатель): <input type="text" name="engine"></label>
                    <label>Mileage (Пробег): <input type="text" name="mileage"></label>
                    <label>Age (Возраст автомобиля): <input type="text" name="age"></label>
                    <label>Price (Цена): <input type="text" name="price"></label>
                    <label>Seller Type (Тип продавца): <input type="text" name="seller_type"></label>
                    <label>Seller Rating (Рейтинг продавца): <input type="text" name="seller_rating" placeholder="4.0 по умолчанию"></label>
                    <label>Seller Reviews (Отзывы продавца): <input type="text" name="seller_reviews"></label>
                    <label>Engine HP (Мощность двигателя в л.с.): <input type="text" name="engine_hp"></label>
                    <input type="submit" value="Прогнозировать цену">
                </form>

                {% if predicted_price %}
                <div class="result">
                    Предсказанная цена: ${{ predicted_price }}
                </div>
                {% endif %}
            </div>
        </body>
    </html>
    ''', predicted_price=predicted_price)
    

if __name__ == '__main__':
    app.run(debug=True)