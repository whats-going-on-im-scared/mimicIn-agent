#!/usr/bin/env python3
"""
LinkedIn Profile Screenshot Scraper with LLM Integration

Takes screenshots of LinkedIn profiles and uses Gemini 2.0 to extract information.

Usage:
  python linkedin_screenshot_scraper.py "https://www.linkedin.com/in/USERNAME/" \
      --email you@example.com --password 'secret' --gemini-api-key YOUR_API_KEY \
      --headless --output profile.json

If Selenium Manager can't fetch ChromeDriver (blocked network), download a matching
chromedriver.exe and pass --chromedriver "C:\\path\\to\\chromedriver.exe"
"""

import argparse
import base64
import json
import os
import random
import sys
import time
from pathlib import Path
from io import BytesIO

import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# --------- small helpers ---------
def jitter(a=0.5, b=1.2):
    """Random delay to appear more human-like"""
    time.sleep(random.uniform(a, b))


def wait_for(driver, locator, timeout=25):
    """Wait for element to be present"""
    return WebDriverWait(driver, timeout).until(EC.presence_of_element_located(locator))


# --------- driver setup ----------
def init_driver(headless: bool, chromedriver_path: str | None):
    """Initialize Chrome driver with optimized settings"""
    opts = ChromeOptions()

    # Stability/noise reduction flags
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-features=VizDisplayCompositor")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option('useAutomationExtension', False)

    # Set window size for consistent screenshots
    opts.add_argument("--window-size=1920,1080")

    if headless:
        opts.add_argument("--headless=new")

    if chromedriver_path:
        service = ChromeService(executable_path=chromedriver_path, log_output="chromedriver.log")
        driver = webdriver.Chrome(service=service, options=opts)
    else:
        # Uses Selenium Manager (will try to download the right driver)
        driver = webdriver.Chrome(options=opts)

    driver.set_page_load_timeout(45)

    # Set viewport size
    driver.execute_script("window.scrollTo(0, 0);")

    return driver


# --------- login functionality -------
def login(driver, email: str, password: str):
    """Login to LinkedIn"""
    print("Logging in to LinkedIn...")
    driver.get("https://www.linkedin.com/login")

    wait_for(driver, (By.ID, "username"))
    driver.find_element(By.ID, "username").clear()
    driver.find_element(By.ID, "username").send_keys(email)
    driver.find_element(By.ID, "password").clear()
    driver.find_element(By.ID, "password").send_keys(password)
    driver.find_element(By.ID, "password").submit()

    # Accept any of these redirects as success
    WebDriverWait(driver, 30).until(
        lambda d: any(s in d.current_url for s in ("/feed", "/in/", "/checkpoint", "login-submit"))
    )
    jitter()
    print("Login successful!")


# --------- screenshot functionality -------
def take_full_page_screenshot(driver, output_path: str = None):
    """
    Take a full page screenshot using Chrome DevTools Protocol
    Returns: bytes of the screenshot
    """
    # Get dimensions
    total_height = driver.execute_script("return document.body.scrollHeight")
    viewport_height = driver.execute_script("return window.innerHeight")

    screenshots = []

    # Scroll to top
    driver.execute_script("window.scrollTo(0, 0);")
    jitter(0.5, 1.0)

    # Calculate number of screenshots needed
    num_screenshots = (total_height // viewport_height) + 1

    for i in range(num_screenshots):
        # Take screenshot
        screenshot = driver.get_screenshot_as_png()
        screenshots.append(Image.open(BytesIO(screenshot)))

        # Scroll down
        scroll_position = min((i + 1) * viewport_height, total_height - viewport_height)
        driver.execute_script(f"window.scrollTo(0, {scroll_position});")
        jitter(0.3, 0.6)

    # Stitch screenshots together
    total_width = screenshots[0].width
    stitched = Image.new('RGB', (total_width, total_height))

    y_offset = 0
    for img in screenshots:
        stitched.paste(img, (0, y_offset))
        y_offset += img.height

    # Save if output path provided
    if output_path:
        stitched.save(output_path)
        print(f"Screenshot saved to {output_path}")

    # Convert to bytes
    img_byte_arr = BytesIO()
    stitched.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return img_byte_arr.getvalue()


def take_section_screenshots(driver):
    """
    Take strategic screenshots of important sections
    Returns: dict with section names and screenshot bytes
    """
    screenshots = {}

    # 1. Profile header screenshot
    driver.execute_script("window.scrollTo(0, 0);")
    jitter(0.5, 1.0)
    screenshots['header'] = driver.get_screenshot_as_png()

    # 2. Experience section screenshot
    try:
        exp_section = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.XPATH, "//section[.//h2//span[contains(text(),'Experience')]]")
            )
        )
        driver.execute_script("arguments[0].scrollIntoView({block:'start'});", exp_section)
        jitter(0.5, 1.0)
        screenshots['experience'] = driver.get_screenshot_as_png()
    except Exception as e:
        print(f"Could not find Experience section: {e}")

    # 3. Education section screenshot (if exists)
    try:
        edu_section = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located(
                (By.XPATH, "//section[.//h2//span[contains(text(),'Education')]]")
            )
        )
        driver.execute_script("arguments[0].scrollIntoView({block:'start'});", edu_section)
        jitter(0.5, 1.0)
        screenshots['education'] = driver.get_screenshot_as_png()
    except Exception:
        print("Education section not found or not accessible")

    # 4. Skills section screenshot (if exists)
    try:
        skills_section = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located(
                (By.XPATH, "//section[.//h2//span[contains(text(),'Skills')]]")
            )
        )
        driver.execute_script("arguments[0].scrollIntoView({block:'start'});", skills_section)
        jitter(0.5, 1.0)
        screenshots['skills'] = driver.get_screenshot_as_png()
    except Exception:
        print("Skills section not found or not accessible")

    return screenshots


# --------- Gemini integration -------
def setup_gemini(api_key: str):
    """Initialize Gemini API"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    return model


def analyze_with_gemini(model, screenshots: dict, profile_url: str):
    """
    Send screenshots to Gemini for analysis
    """
    print("Analyzing screenshots with Gemini 2.0...")

    # Prepare images for Gemini
    images = []
    for section_name, screenshot_bytes in screenshots.items():
        img = Image.open(BytesIO(screenshot_bytes))
        images.append(img)

    # Craft the prompt
    prompt = """
    Analyze these LinkedIn profile screenshots and extract the following information in JSON format:

    {
        "name": "Full name of the person",
        "position": "Current job title/position (from most recent experience)",
        "company": "Current company name (from most recent experience)",
        "location": "Location/city",
        "summary": "Brief professional summary if visible",
        "experience": [
            {
                "title": "Job title",
                "company": "Company name",
                "duration": "Time period",
                "description": "Brief description if available"
            }
        ],
        "education": [
            {
                "degree": "Degree name",
                "institution": "School/University name",
                "year": "Graduation year or period"
            }
        ],
        "skills": ["List of skills if visible"],
        "additional_info": "Any other relevant information"
    }

    Please provide accurate information based only on what's visible in the screenshots.
    If a field is not visible or unclear, use null for that field.
    Focus on extracting the most recent/current position and company for the main position and company fields.
    """

    try:
        # Send to Gemini
        response = model.generate_content([prompt] + images)

        # Parse the response
        response_text = response.text

        # Try to extract JSON from the response
        # Gemini might return JSON wrapped in markdown code blocks
        if "```json" in response_text:
            json_start = response_text.index("```json") + 7
            json_end = response_text.index("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.index("```") + 3
            json_end = response_text.index("```", json_start)
            json_str = response_text[json_start:json_end].strip()
        else:
            # Assume the entire response is JSON
            json_str = response_text.strip()

        # Parse JSON
        profile_data = json.loads(json_str)
        profile_data['url'] = profile_url

        return profile_data

    except json.JSONDecodeError as e:
        print(f"Error parsing Gemini response as JSON: {e}")
        print(f"Raw response: {response_text[:500]}...")
        # Return a basic structure with the raw response
        return {
            "error": "Failed to parse Gemini response",
            "raw_response": response_text,
            "url": profile_url
        }
    except Exception as e:
        print(f"Error communicating with Gemini: {e}")
        return {
            "error": str(e),
            "url": profile_url
        }


# --------- main scraping function -------
def scrape_profile(driver, profile_url: str, gemini_model, save_screenshots: bool = False):
    """
    Navigate to profile, take screenshots, and analyze with Gemini
    """
    print(f"Navigating to profile: {profile_url}")
    driver.get(profile_url)

    # Wait for main content to load
    wait_for(driver, (By.TAG_NAME, "main"))
    jitter(2, 3)  # Give extra time for content to fully render

    # Take screenshots
    print("Taking screenshots...")
    screenshots = take_section_screenshots(driver)

    # Optionally save screenshots locally
    if save_screenshots:
        timestamp = int(time.time())
        for section_name, screenshot_bytes in screenshots.items():
            filename = f"linkedin_{section_name}_{timestamp}.png"
            with open(filename, 'wb') as f:
                f.write(screenshot_bytes)
            print(f"Saved {filename}")

    # Analyze with Gemini
    profile_data = analyze_with_gemini(gemini_model, screenshots, profile_url)

    return profile_data


# ---------- CLI -------------------
def parse_args():
    p = argparse.ArgumentParser(description="LinkedIn profile screenshot scraper with Gemini 2.0")
    p.add_argument("profile_url", help="Full LinkedIn profile URL, e.g. https://www.linkedin.com/in/USERNAME/")
    p.add_argument("--email", help="LinkedIn email (fallback to .env EMAIL)")
    p.add_argument("--password", help="LinkedIn password (fallback to .env PASSWORD)")
    p.add_argument("--gemini-api-key", help="Gemini API key (fallback to .env GOOGLE_API_KEY)")
    p.add_argument("--headless", action="store_true", help="Run Chrome in headless mode")
    p.add_argument("--output", "-o", help="Write JSON to this file")
    p.add_argument("--save-screenshots", action="store_true", help="Save screenshots locally")
    p.add_argument("--chromedriver", help="Path to chromedriver.exe if Selenium Manager is blocked")
    return p.parse_args()

def main():
    load_dotenv()

    args = parse_args()
    email = args.email or os.getenv("EMAIL")
    password = args.password or os.getenv("PASSWORD")
    # fix here
    google_api_key = args.gemini_api_key or os.getenv("GOOGLE_API_KEY")

    if not email or not password:
        print("ERROR: Missing credentials. Provide --email/--password or set EMAIL and PASSWORD in .env",
              file=sys.stderr)
        sys.exit(2)

    if not google_api_key:
        print("ERROR: Missing Gemini API key. Provide --gemini-api-key or set GOOGLE_API_KEY in .env",
              file=sys.stderr)
        sys.exit(2)

    driver = None
    try:
        # Initialize Gemini
        gemini_model = setup_gemini(google_api_key)

        # Initialize driver
        driver = init_driver(headless=args.headless, chromedriver_path=args.chromedriver)

        # Login to LinkedIn
        login(driver, email, password)

        # Scrape profile
        data = scrape_profile(driver, args.profile_url, gemini_model, args.save_screenshots)

        # Output results
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(data, ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


if __name__ == "__main__":
    main()