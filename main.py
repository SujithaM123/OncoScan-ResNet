import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
from PIL import Image
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import os


# Define your ResNet model
def create_resnet_model(num_classes=1000):
    model = models.resnet18(weights="IMAGENET1K_V1")
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Identity()  # Change the final layer to Identity to get features
    return model


# Define image transformations
def transform_image(image):
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image


# Load dataset
def load_dataset(data_dir, batch_size=32):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader, len(dataset.classes)


# Extract features
def extract_features(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # Set to evaluation mode
    features = []
    labels = []

    with torch.no_grad():
        for images, lbls in data_loader:
            images = images.to(device)
            output = model(images)
            features.append(output.cpu().numpy())
            labels.append(lbls.numpy())

    return np.concatenate(features), np.concatenate(labels)


# Train SVM classifier
def train_svm(features, labels):
    clf = svm.SVC(kernel="linear", probability=True)
    clf.fit(features, labels)
    return clf


# Evaluate SVM model
def evaluate_svm(clf, features, labels):
    predictions = clf.predict(features)
    accuracy = accuracy_score(labels, predictions)
    print(f"SVM Accuracy: {accuracy * 100:.2f}%")


# Predict melanoma probability
def predict_melanoma_probability(clf, model, image):
    single_image = transform_image(image)
    single_image = single_image.to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    with torch.no_grad():
        feature = model(single_image).cpu().numpy()
        probabilities = clf.predict_proba(feature.reshape(1, -1))[0]

    return probabilities


# Plot pie chart for the results
def plot_pie_chart(probabilities):
    labels = ["No Melanoma", "Melanoma"]
    colors = ["green", "red"]
    plt.figure(figsize=(5, 5))
    plt.pie(
        probabilities, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
    )
    plt.title("Prediction Probability Distribution")
    st.pyplot(plt)
#  pdf is generated here

def generate_pdf_report(image, prediction_result, confidence, probabilities):
    # Create instance of FPDF class
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(255, 0, 0)  # Red color for the title
    pdf.cell(200, 10, txt="Melanoma Prediction Report", ln=True, align="C")
    pdf.ln(10)

    # Add Image with border
    image_path = "temp_image.jpg"
    image.save(image_path)
    pdf.set_line_width(1)
    pdf.image(image_path, x=50, y=45, w=100)
    pdf.ln(120)  # Adjust the vertical spacing after the image

    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(0, 0, 0)  # Black color for text
    pdf.cell(200, 10, txt="Prediction Result", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, txt=f"Prediction: {prediction_result}")
    pdf.multi_cell(0, 10, txt=f"Confidence: {confidence:.2f}%")
    pdf.multi_cell(0, 10, txt=f"Probability of No Melanoma: {probabilities[0]:.2f}")
    pdf.multi_cell(0, 10, txt=f"Probability of Melanoma: {probabilities[1]:.2f}")
    pdf.ln(10)

    pdf.set_line_width(1)
    pdf.set_draw_color(255, 182, 193)  # Light pink border for content
    pdf.rect(10, 20, 190, 260)  # Draw border around the content area

    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(200, 10, txt="Detailed Explanation", ln=True)
    pdf.multi_cell(
        0,
        10,
        txt="The model uses ResNet-18 features, extracted from the image, to predict the presence of melanoma.",
    )
    pdf.multi_cell(
        0,
        10,
        txt="It analyzes the probability of melanoma presence, providing the confidence level.",
    )
    pdf.ln(10)

    pdf.set_y(-15)
    pdf.set_font("Arial", "I", 8)
    pdf.cell(0, 10, f"Page {pdf.page_no()}", 0, 0, "C")

    pdf_output_path = "Melanoma_Prediction_Report.pdf"
    pdf.output(pdf_output_path)
    return pdf_output_path


def login_page():
    st.title("Login Page")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if email == "admin@gmail.com" and password == "admin@1234":
            st.session_state.logged_in = True
            st.session_state.page = "Home"
            st.success("Login successful!")
        else:
            st.error("Invalid email or password")


import streamlit as st


def home_page():
    st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")

    # Header section
    st.title("Welcome to the Melanoma Detection Scanning App!")
    st.markdown("Click the button below to start the scanning process.")

    # Create a container for better layout control
    with st.container():
        col1, col2 = st.columns(
            [2, 1]
        )  # Adjust the column width as per your preference

        # Add a welcoming image (optional)
        with col1:
            st.image(
                "https://imgs.search.brave.com/FZcmK7LDqjNEEwpNRFlAEzqpVJCg68sBv3meR91cRmY/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly93d3cu/Y25ldC5jb20vYS9p/bWcvcmVzaXplL2Ix/YmE4YTI2NTBmYTlk/YjliNWQ3MTI5NGY2/OGZhMDAzYzkyZWI0/NzMvaHViLzIwMjEv/MDUvMTcvNTMxZjZj/NTEtMzVlNC00MTRm/LTg4YjUtNjBjMDRi/NDAxNDU5L2Rlcm0t/aGVyby1pbWFnZS0y/LnBuZz9hdXRvPXdl/YnAmZml0PWNyb3Am/aGVpZ2h0PTY3NSZ3/aWR0aD0xMjAw",
                caption="Scan to Begin",
                use_column_width=True,
            )

        # Add button with better UI and style
        with col2:
            st.markdown(
                """
                <style>
                .stButton>button {
                    background-color: #4CAF50;
                    color: white;
                    font-size: 18px;
                    padding: 15px 32px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    border-radius: 8px;
                    cursor: pointer;
                    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                }
                .stButton>button:hover {
                    background-color: #45a049;
                }
                </style>
            """,
                unsafe_allow_html=True,
            )
            if st.button("Click Here to Scan"):
                st.session_state.page = "Detection"


def detection_page():
    st.title("Melanoma Detection Page")
    st.markdown("### Upload an image of skin for melanoma prediction.")
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Display spinner while processing
        with st.spinner("Processing... Please wait."):
            train_data_dir = (
                r"D:\main\Desktop\Ml\Maligant-ML\data\melanoma_cancer_dataset\train"
            )
            test_data_dir = (
                r"D:\main\Desktop\Ml\Maligant-ML\data\melanoma_cancer_dataset\test"
            )

            train_loader, num_classes = load_dataset(train_data_dir)
            test_loader, _ = load_dataset(test_data_dir)

            model = create_resnet_model(num_classes=num_classes)
            train_features, train_labels = extract_features(model, train_loader)
            svm_classifier = train_svm(train_features, train_labels)
            test_features, test_labels = extract_features(model, test_loader)
            evaluate_svm(svm_classifier, test_features, test_labels)

            probabilities = predict_melanoma_probability(svm_classifier, model, image)
            result = (
                "Melanoma" if probabilities[1] > probabilities[0] else "No Melanoma"
            )
            result_color = "red" if result == "Melanoma" else "green"

        # Display results after processing
        st.markdown(f"## Prediction: {result}")
        st.markdown(
            f"<h3 style='color:{result_color};'>{result} Detected</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(f"### Confidence: {max(probabilities) * 100:.2f}%")

        plot_pie_chart(probabilities)
        pdf_path = generate_pdf_report(
            image, result, max(probabilities) * 100, probabilities
        )
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download Prediction Report",
                data=f,
                file_name="Melanoma_Prediction_Report.pdf",
                mime="application/pdf",
            )


def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "page" not in st.session_state:
        st.session_state.page = "Login"

    if st.session_state.logged_in:
        if st.session_state.page == "Home":
            home_page()
        elif st.session_state.page == "Detection":
            detection_page()
    else:
        login_page()


if __name__ == "__main__":
    main()
