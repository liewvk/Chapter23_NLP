import string
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def clean_text(text):
    text = text.lower()
    text = text.strip()

    for punctuation in string.punctuation:
        text = text.replace(punctuation, "")

    return text


def main():
    data_file = Path("data") / "feedback.csv"
    output_folder = Path("outputs")
    output_file = output_folder / "sentiment_results.csv"

    output_folder.mkdir(exist_ok=True)

    df = pd.read_csv(data_file)

    print("Original Feedback Data")
    print("----------------------")
    print(df)

    print()
    print("Sentiment Counts")
    print("----------------")
    print(df["Sentiment"].value_counts())

    df["CleanText"] = df["Text"].apply(clean_text)

    print()
    print("Cleaned Text")
    print("------------")
    print(df[["Text", "CleanText"]])

    X = df["CleanText"]
    y = df["Sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    vectorizer = CountVectorizer()

    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train_vectorized, y_train)

    predictions = model.predict(X_test_vectorized)

    accuracy = accuracy_score(y_test, predictions)

    results = pd.DataFrame({
        "OriginalText": X_test,
        "ActualSentiment": y_test,
        "PredictedSentiment": predictions
    })

    print()
    print("Prediction Results")
    print("------------------")
    print(results)

    print()
    print("Model Accuracy")
    print("--------------")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Accuracy Percentage: {accuracy * 100:.2f}%")

    print()
    print("Classification Report")
    print("---------------------")
    print(classification_report(y_test, predictions))

    print()
    print("Vocabulary")
    print("----------")
    print(vectorizer.get_feature_names_out())

    new_feedback = [
        "This lesson is very useful",
        "The instructions are too confusing",
        "I love the clear explanation",
        "The project is hard to follow"
    ]

    new_feedback_cleaned = [
        clean_text(text)
        for text in new_feedback
    ]

    new_feedback_vectorized = vectorizer.transform(new_feedback_cleaned)

    new_predictions = model.predict(new_feedback_vectorized)

    print()
    print("New Feedback Predictions")
    print("------------------------")

    for text, prediction in zip(new_feedback, new_predictions):
        print(f"Text: {text}")
        print(f"Predicted Sentiment: {prediction}")
        print()

    results.to_csv(output_file, index=False)

    print(f"Prediction results saved to: {output_file}")


main()
