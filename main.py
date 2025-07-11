from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
import joblib
import numpy as np

app = FastAPI()

# Загрузка модели и масштабировщика
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# 🧾 HTML-главная с формой
@app.get("/", response_class=HTMLResponse)
def form_page():
    html_content = """
    <html>
        <head>
            <title>Проверка экономического риска</title>
            <style>
                body { font-family: Arial; padding: 40px; background: #f4f4f4; }
                .box { background: white; padding: 20px; border-radius: 10px; width: 400px; margin: auto; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
                input, button { width: 100%; padding: 10px; margin-top: 10px; }
                h2 { color: #2c3e50; }
            </style>
        </head>
        <body>
            <div class="box">
                <h2>🔍 Проверка риска</h2>
                <form action="/predict_form" method="post">
                    <input name="profit_margin" placeholder="Рентабельность продаж" required />
                    <input name="debt_ratio" placeholder="Коэффициент задолженности" required />
                    <input name="liquidity_ratio" placeholder="Коэффициент ликвидности" required />
                    <input name="return_on_assets" placeholder="Рентабельность активов" required />
                    <button type="submit">Проверить</button>
                </form>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# 🎯 Обработка формы и предсказание
@app.post("/predict_form", response_class=HTMLResponse)
def predict_from_form(
    profit_margin: float = Form(...),
    debt_ratio: float = Form(...),
    liquidity_ratio: float = Form(...),
    return_on_assets: float = Form(...)
):
    X = np.array([[profit_margin, debt_ratio, liquidity_ratio, return_on_assets]])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    result = "⚠️ Есть риск!" if prediction == 1 else "✅ Риска нет"

    html_result = f"""
    <html>
        <head>
            <title>Результат</title>
            <style>
                body {{ font-family: Arial; padding: 40px; background: #f4f4f4; text-align: center; }}
                .result {{ background: white; padding: 30px; display: inline-block; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h2 {{ color: {'red' if prediction == 1 else 'green'}; }}
            </style>
        </head>
        <body>
            <div class="result">
                <h2>{result}</h2>
                <a href="/">Назад</a>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_result)