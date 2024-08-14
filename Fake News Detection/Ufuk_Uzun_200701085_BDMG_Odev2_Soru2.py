import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Concatenate, Input, Reshape
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def check_data_set(data_set_name):
    if os.path.exists(data_set_name):
        print(f"{data_set_name} veri seti bulundu.")
        data_path = os.path.join(os.getcwd(), data_set_name)
        return data_path
    else:
        print(f"{data_set_name} veri seti bulunamadı.")
        exit(0)


def prepare_classification_data(data):
    X = data["text"]
    y = data["label"]

    vectorizer = CountVectorizer(stop_words="english", max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)

    return X_vectorized, y, vectorizer


def train_and_evaluate_classification_model(X_train, y_train, X_test, y_test):
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Sınıflandırma Modeli Performansı:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


def prepare_transformer_data(data, max_sequence_len=100):
    X = data["text"]
    y = data["label"]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = tokenizer.texts_to_sequences(X)
    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len)

    return input_sequences, y, tokenizer, total_words


def create_transformer_model(total_words, max_sequence_len):
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len))

    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_transformer_model(model, input_sequences, y_transformer, epochs=10, batch_size=64):
    model.fit(input_sequences, y_transformer, epochs=epochs, batch_size=batch_size)


def main():
    data_set_name = "FakeArchive"
    data_directory = check_data_set(data_set_name)

    # Veri setlerini yükleme ve birleştirme
    fake_news = pd.read_csv(f"{data_directory}\\fake.csv")
    true_news = pd.read_csv(f"{data_directory}\\true.csv")
    fake_news["label"] = 1
    true_news["label"] = 0
    combined_data = pd.concat([fake_news, true_news], ignore_index=True).sample(frac=1).reset_index(drop=True)

    # Eğitim ve test setlerine bölme
    X_train, X_test, y_train, y_test = train_test_split(combined_data["text"], combined_data["label"], test_size=0.2,
                                                        random_state=42)

    # Sınıflandırma Modeli
    X_train_vectorized, y_train, vectorizer = prepare_classification_data(
        pd.DataFrame({"text": X_train, "label": y_train}))
    X_test_vectorized, y_test, _ = prepare_classification_data(pd.DataFrame({"text": X_test, "label": y_test}))
    train_and_evaluate_classification_model(X_train_vectorized, y_train, X_test_vectorized, y_test)

    # Transformer Modeli
    transformer_data = pd.concat([fake_news.sample(frac=0.1, random_state=42), true_news], ignore_index=True).sample(
        frac=1).reset_index(drop=True)
    input_sequences, y_transformer, tokenizer, total_words = prepare_transformer_data(transformer_data)
    model_transformer = create_transformer_model(total_words, max_sequence_len=input_sequences.shape[1])
    train_transformer_model(model_transformer, input_sequences, y_transformer)

if __name__ == "__main__":
    main()
