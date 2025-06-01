import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms, models
from transformers import pipeline
import streamlit as st

# SAD KEYWORDS that will decrease the score if found
SAD_KEYWORDS = [
    "sad", "upset", "depressed", "down", "unhappy", "cry", "tears", "gloom", "sorrow", "mourn",
    "grief", "heartbroken", "blue", "miserable", "melancholy"
]

@st.cache_data
def get_image_and_caption(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    image_tag = soup.find('meta', property='og:image')
    caption_tag = soup.find('meta', property='og:description')

    image_url = image_tag['content'] if image_tag else None
    caption = caption_tag['content'] if caption_tag else None

    return image_url, caption

@st.cache_resource
def load_caption_pipeline():
    return pipeline("sentiment-analysis")

@st.cache_resource
def load_image_model():
    model = models.resnet18(pretrained=True)
    model.eval()
    return model

def analyze_caption_sentiment(caption_text):
    sentiment_pipeline = load_caption_pipeline()
    result = sentiment_pipeline(caption_text)
    label = result[0]['label']
    score = result[0]['score']

    # Check for sad keywords and decrease score
    caption_lower = caption_text.lower()
    sad_count = sum(1 for word in SAD_KEYWORDS if word in caption_lower)

    if sad_count > 0:
        penalty = 0.10 * sad_count
        score = max(0.0, score - penalty)

    # Determine label based on adjusted score
    if score > 0.60:
        label = "POSITIVE"
    elif score < 0.40:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return label, score

def analyze_image_sentiment(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert('RGB')

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    model = load_image_model()

    with torch.no_grad():
        output = model(input_tensor)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)

    score = top_prob.item()

    # Determine label based on score
    if score > 0.60:
        label = "POSITIVE"
    elif score < 0.40:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return label, score

def get_sentiment_quote(sentiment):
    if sentiment == "POSITIVE":
        return "💫 'Living the life!'"
    elif sentiment == "NEGATIVE":
        return "🙂 'Chin up! A smile can change everything.'"
    else:
        return "🌿 'Steady as she goes!'"

# Streamlit App
st.title("📸 Social Media Post Sentiment Analysis")

url = st.text_input("🔗 Enter the social media post URL:")

if url:
    image_url, caption = get_image_and_caption(url)

    if not image_url or not caption:
        st.error("❌ Could not extract image or caption from the URL.")
    else:
        st.image(image_url, caption="Post Image", use_column_width=True)
        st.write("**Caption:**", caption)

        caption_sentiment, caption_score = analyze_caption_sentiment(caption)
        image_sentiment, image_score = analyze_image_sentiment(image_url)

        st.subheader("📊 Sentiment Analysis Results")
        st.write(f"**Caption Sentiment:** {caption_sentiment} (score: {caption_score:.2f})")
        st.write(f"**Image Sentiment:** {image_sentiment} (score: {image_score:.2f})")

        # Final sentiment logic based on average score
        combined_score = (caption_score + image_score) / 2

        if combined_score > 0.60:
            final_sentiment = "POSITIVE"
        elif combined_score < 0.40:
            final_sentiment = "NEGATIVE"
        else:
            final_sentiment = "NEUTRAL"

        quote = get_sentiment_quote(final_sentiment)
        st.success(f"**🟩 Final Sentiment: {final_sentiment}**\n\n*{quote}*")
