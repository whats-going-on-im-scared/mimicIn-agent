import os, re, subprocess, pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService

def get_default_chrome_options():
    o = webdriver.ChromeOptions()
    o.add_argument("--no-sandbox")
    o.add_argument("--disable-gpu")               # quiet Skia/Viz errors
    o.add_argument("--disable-dev-shm-usage")     # more stable in CI
    # o.add_argument("--headless=new")            # if you run headless, also set a window size:
    # o.add_argument("--window-size=1280,800")
    o.add_argument("--disable-features=VizDisplayCompositor")
    return o

def test_basic_options():
    driver = webdriver.Chrome(options=get_default_chrome_options())
    driver.quit()

def test_args():
    options = get_default_chrome_options()
    options.add_argument("--start-maximized")
    # Send chromedriver logs to file to keep stdout clean:
    service = ChromeService(log_output="chromedriver.log")
    driver = webdriver.Chrome(service=service, options=options)
    try:
        driver.get("https://www.selenium.dev")
        assert "Selenium" in driver.title
    finally:
        driver.quit()

def test_get_browser_logs():
    driver = webdriver.Chrome(options=get_default_chrome_options())
    try:
        driver.get("https://www.selenium.dev/selenium/web/bidi/logEntryAdded.html")
        driver.find_element(By.ID, "consoleError").click()
        logs = driver.get_log("browser")
        assert any("I am console error" in log["message"] for log in logs)
    finally:
        driver.quit()
