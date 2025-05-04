from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

def simulate_emotion_prediction():
    base64_image = "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"
    driver.execute_script("""
        fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ image: arguments[0] })
        });
    """, base64_image)

try:
    # Visit your deployed domain
    driver.get("https://emotioninsight.in/dashboard/")
    wait = WebDriverWait(driver, 10)

    # Check navbar
    navbar_text = wait.until(EC.presence_of_element_located((By.XPATH, "//span[text()='Emotion Insights']")))
    assert navbar_text is not None
    print("✅ Navbar loaded")

    # Check video feed
    video = driver.find_element(By.ID, "video-feed")
    assert video is not None
    print("✅ Video feed found")

    # Check emotion header
    emotion_header = driver.find_element(By.XPATH, "//h5[text()='Current Emotion']")
    assert emotion_header is not None
    print("✅ Emotion box loaded")

    # Simulate emotion prediction to trigger movie recommendations
    simulate_emotion_prediction()

    # Wait for recommendation header to appear after prediction
    rec_header = wait.until(EC.presence_of_element_located((By.ID, "recommendation-header")))
    assert rec_header is not None
    print("✅ Recommendation header found")

    # Check if movie poster appears
    poster_img = wait.until(EC.presence_of_element_located((By.ID, "movie-poster")))
    assert poster_img is not None
    print("✅ Movie poster image loaded")

    print("🎉 All UI elements loaded successfully.")

finally:
    driver.quit()