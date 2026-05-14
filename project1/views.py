from django.shortcuts import render, redirect
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from django.conf import settings
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def index(request):
    return render(request, "project1/index.html")


def classification(request):
    context = {}

    upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, "current_classification_dataset.csv")

    if request.method == "POST":

        if "csv_file" not in request.FILES:
            context["error"] = "Please choose a CSV file first."
            return render(request, "project1/classification.html", context)

        csv_file = request.FILES["csv_file"]

        with open(file_path, "wb+") as destination:
            for chunk in csv_file.chunks():
                destination.write(chunk)

        return redirect("project1:classification_analyze")

    return render(request, "project1/classification.html", context)


def classification_analyze(request):
    context = {}

    upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    file_path = os.path.join(upload_dir, "current_classification_dataset.csv")

    if not os.path.exists(file_path):
        context["error"] = "Please upload a CSV file first."
        return render(request, "project1/classification.html", context)

    df = pd.read_csv(file_path)

    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    columns = df.columns.tolist()

    if len(columns) < 3:
        context["error"] = "CSV must contain at least two feature columns and one target column."
        return render(request, "project1/classification.html", context)

    feature_columns = columns[:-1]
    target_column = columns[-1]

    x_col = request.POST.get("x_col") or feature_columns[0]
    y_col = request.POST.get("y_col") or feature_columns[1]

    context["columns"] = feature_columns
    context["target_column"] = target_column
    context["selected_x"] = x_col
    context["selected_y"] = y_col
    context["tables"] = df.head().to_html(classes="data-table")

    # Scatter plot
    plt.figure()
    plt.scatter(df[x_col], df[y_col])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{x_col} vs {y_col}")

    scatter_filename = f"classification_scatter_{int(time.time())}.png"
    scatter_path = os.path.join(settings.MEDIA_ROOT, scatter_filename)
    plt.savefig(scatter_path)
    plt.close()

    context["scatter_plot_url"] = settings.MEDIA_URL + scatter_filename

    # Bar chart
    class_counts = df[target_column].value_counts()
    colors = ["#9DCFE6", "#F19393", "#A8E6A3", "#FFD27F", "#C8A2FF"]

    plt.figure()
    class_counts.plot(kind="bar", color=colors)
    plt.xlabel("Class")
    plt.ylabel("Number of examples")
    plt.title(f"Class Distribution: {target_column}")
    plt.xticks(rotation=0)

    bar_filename = f"class_distribution_{int(time.time())}.png"
    bar_path = os.path.join(settings.MEDIA_ROOT, bar_filename)
    plt.savefig(bar_path)
    plt.close()

    context["bar_plot_url"] = settings.MEDIA_URL + bar_filename

    return render(request, "project1/classification_analyze.html", context)


def classification_train(request):
    context = {}

    upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    file_path = os.path.join(upload_dir, "current_classification_dataset.csv")

    if not os.path.exists(file_path):
        context["error"] = "Please upload a CSV file first."
        return render(request, "project1/classification.html", context)

    df = pd.read_csv(file_path)

    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    columns = df.columns.tolist()

    if len(columns) < 3:
        context["error"] = "CSV must contain at least two feature columns and one target column."
        return render(request, "project1/classification.html", context)

    target_column = columns[-1]

    selected_model = request.POST.get("model", "decision_tree")
    test_size = float(request.POST.get("test_size", 0.2))

    context["selected_model"] = selected_model
    context["selected_test_size"] = str(test_size)
    context["target_column"] = target_column

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )

    if selected_model == "decision_tree":
        model = DecisionTreeClassifier()
        model_name = "Decision Tree Classifier"

    elif selected_model == "random_forest":
        model = RandomForestClassifier()
        model_name = "Random Forest Classifier"

    elif selected_model == "knn":
        model = KNeighborsClassifier()
        model_name = "K-Nearest Neighbors Classifier"

    else:
        model = DecisionTreeClassifier()
        model_name = "Decision Tree Classifier"

    model.fit(X_train, y_train)
    
    model_dir = os.path.join(settings.MEDIA_ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "classification_model.pkl")
    metadata_path = os.path.join(model_dir, "classification_metadata.pkl")

    joblib.dump(model, model_path)

    metadata = {
        "feature_columns": df.columns[:-1].tolist(),
        "target_column": df.columns[-1],
        "model_name": model_name,
    }

    joblib.dump(metadata, metadata_path)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    context["model_name"] = model_name
    context["train_accuracy"] = round(train_accuracy * 100, 2)
    context["test_accuracy"] = round(test_accuracy * 100, 2)
    context["train_size"] = round((1 - test_size) * 100)
    context["test_size"] = round(test_size * 100)

    # Accuracy bar chart
    plt.figure()

    accuracy_names = ["Training Accuracy", "Testing Accuracy"]
    accuracy_values = [
        train_accuracy * 100,
        test_accuracy * 100
    ]

    plt.bar(
        accuracy_names,
        accuracy_values,
        color=["#9DCFE6", "#F19393"]
    )

    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.title("Training vs Testing Accuracy")

    accuracy_filename = f"accuracy_chart_{int(time.time())}.png"
    accuracy_path = os.path.join(settings.MEDIA_ROOT, accuracy_filename)

    plt.savefig(accuracy_path)
    plt.close()

    context["accuracy_chart_url"] = settings.MEDIA_URL + accuracy_filename

    return render(request, "project1/classification_train.html", context)

def classification_test(request):
    context = {}

    model_dir = os.path.join(settings.MEDIA_ROOT, "models")
    model_path = os.path.join(model_dir, "classification_model.pkl")
    metadata_path = os.path.join(model_dir, "classification_metadata.pkl")

    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        context["error"] = "Please train a model before testing."
        return render(request, "project1/classification_train.html", context)

    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)

    feature_columns = metadata["feature_columns"]
    target_column = metadata["target_column"]
    model_name = metadata["model_name"]

    context["feature_columns"] = feature_columns
    context["target_column"] = target_column
    context["model_name"] = model_name

    if request.method == "POST":
        input_values = []

        try:
            for feature in feature_columns:
                value = float(request.POST.get(feature))
                input_values.append(value)

            prediction = model.predict([input_values])[0]

            context["prediction"] = prediction
            context["input_values"] = zip(feature_columns, input_values)

        except ValueError:
            context["error"] = "Please enter valid numeric values for all fields."

    return render(request, "project1/classification_test.html", context)