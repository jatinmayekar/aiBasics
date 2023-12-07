import pyscript

@pyscript
def create_wordcloud(text):
    # Configure D3 word cloud
    word_cloud = WordCloud()
    word_cloud.words(text)
    word_cloud.generate()

    # Get image data
    image_data = word_cloud.to_svg()

    # Convert to data URL and pass to JavaScript
    data_url = "data:image/svg+xml;base64," + b64encode(image_data.encode()).decode()
    py2js.call_js("saveImage", data_url)