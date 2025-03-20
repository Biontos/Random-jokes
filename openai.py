import json
import random
from flask import Flask, render_template, jsonify, request
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import numpy as np

# Загружаем модель GPT-2 для генерации шуток
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
model_gpt2 = TFGPT2LMHeadModel.from_pretrained("gpt2")

# Словарь для случайных шуток
jokes_parts = {
    "начало": [
        "Когда жизнь даёт тебе лимоны,",
        "Я настолько ленив, что даже",
        "Мой Wi-Fi работает быстрее,",
        "Кто рано встаёт,",
        "Лучше не работать, чем работать,",
        "Деньги не приносят счастья, но зато",
        "Если бы у меня была возможность, я бы",
        "Что-то пошло не так, и теперь я",
        "Я всегда думал, что если бы был супергероем,",
        "Любовь — это когда",
        "Вся моя жизнь — это",
        "Если жизнь тебе ставит палки в колеса,",
        "Иногда я просто",
        "Если бы я был машиной времени,",
        "Когда наступает понедельник,",
        "Я всегда верил, что когда-нибудь",
        "Если бы я был компьютером,",
        "Если бы я знал, что буду в таком состоянии,",
        "Если бы я был поваром,",
        "Лучше поздно, чем никогда, но если ты опоздал на работу,",
        "Иногда хочется просто забежать в кафе и заказать",
        "Когда ты на совещании, и тебе говорят: 'Ты тут зачем?'"
    ],
    "середина": [
        "сделай лимонад.",
        "мои мысли выходят погулять без меня.",
        "Хотел бы я так же...",
        "тому потом весь день хочется спать!",
        "и завтра снова ничего не сделаю.",
        "не могу понять, что тут не так...",
        "спокойно живу, потому что мне нечего терять!",
        "вроде бы всё, но что-то не хватает...",
        "был бы счастливым, если бы не мои друзья.",
        "я бы с радостью, но только если бы мне платили.",
        "ни о чём не думаю, потому что у меня всегда нет времени.",
        "и теперь ищу, где это пошло не так.",
        "наверное, я просто напишу заявку в поддержку.",
        "просто отправляю свою жизнь в перезагрузку.",
        "и подумаю, может быть, я не тот человек.",
        "теперь мне пора делать перерыв.",
        "вроде бы всё понятно, но мне всё равно скучно.",
        "пришёл в такой момент, что даже не знаю, что делать.",
        "но потом снова вернусь к своему делу.",
        "вроде и не страшно, но в голове тревога."
    ]
}

def generate_random_joke():
    start = random.choice(jokes_parts["начало"])
    middle = random.choice(jokes_parts["середина"])
    return start + " " + middle

# Подготовка данных для обучения модели генерации шуток
tokenizer = Tokenizer()
tokenizer.fit_on_texts(["Когда жизнь даёт тебе лимоны, сделай лимонад."])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for joke in ["Когда жизнь даёт тебе лимоны, сделай лимонад."]:
    token_list = tokenizer.texts_to_sequences([joke])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = keras.utils.to_categorical(y, num_classes=total_words)

# Модель для генерации шуток
model = keras.Sequential([
    keras.layers.Embedding(total_words, 64, input_length=max_sequence_length - 1),
    keras.layers.LSTM(100, return_sequences=True),
    keras.layers.LSTM(100),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, verbose=1)

def generate_joke(seed_text, next_words=10):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

app = Flask(__name__)

# Загрузка сохранённых шуток
def load_saved_jokes():
    try:
        with open("saved_jokes.json", "r", encoding="utf-8") as f:
            data = f.read().strip()
            if not data:
                return []
            return json.loads(data)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []

# Сохранение шуток
def save_saved_jokes(jokes):
    with open("saved_jokes.json", "w", encoding="utf-8") as f:
        json.dump(jokes, f, ensure_ascii=False, indent=4)

saved_jokes = load_saved_jokes()

@app.route('/')
def index():
    return render_template('index.html', jokes=saved_jokes)

@app.route('/generate_joke')
def api_generate_joke():
    joke = generate_joke("Когда жизнь")
    return jsonify({'joke': joke})

@app.route('/generate_random_joke')
def api_generate_random_joke():
    joke = generate_random_joke()
    return jsonify({'joke': joke})

@app.route('/generate_smart_joke')
def api_generate_smart_joke():
    seed_text = "Когда жизнь"
    joke = generate_joke(seed_text)
    return jsonify({'joke': joke})

@app.route('/remember_joke', methods=['POST'])
def remember_joke():
    joke = request.form.get('joke')
    if joke:
        saved_jokes.append(joke)
        save_saved_jokes(saved_jokes)
    return jsonify({'message': 'Joke saved!', 'saved_jokes': saved_jokes})

if __name__ == '__main__':
    app.run(debug=True)
