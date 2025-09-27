from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import os

def screenshot_page(url: str, out_path: str = "page.png", wait_ms: int = 800) -> str:
    """
    Take a full-page screenshot of a publicly accessible URL you have permission to capture.
    Returns the saved file path.
    """
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1280,2000")  # initial viewport

    driver = webdriver.Chrome(ChromeDriverManager().install(), options=opts)
    try:
        driver.get(url)
        time.sleep(wait_ms / 1000.0)

        # Try to compute full height and use Chrome's full-page capture
        total_height = driver.execute_script(
            "return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight, "
            "document.body.offsetHeight, document.documentElement.offsetHeight, "
            "document.body.clientHeight, document.documentElement.clientHeight);"
        )
        driver.set_window_size(1280, max(1200, total_height))
        time.sleep(0.2)
        driver.save_screenshot(out_path)
        return os.path.abspath(out_path)
    finally:
        driver.quit()
