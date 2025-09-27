#!/usr/bin/env python3
"""
LinkedIn Profile Scraper (Chrome/Selenium)

Extracts: name, position (from first Experience item), company (from first Experience item), location

Usage:
  python linkedin_profile_scrape_chrome.py "https://www.linkedin.com/in/USERNAME/" \
      --email you@example.com --password 'secret' --headless --output profile.json

If Selenium Manager can't fetch ChromeDriver (blocked network), download a matching
chromedriver.exe and pass --chromedriver "C:\\path\\to\\chromedriver.exe"
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# --------- small helpers ---------
def jitter(a=0.5, b=1.2):
    time.sleep(random.uniform(a, b))


def wait_for(driver, locator, timeout=25):
    return WebDriverWait(driver, timeout).until(EC.presence_of_element_located(locator))


def safe_text(driver, by, value, context=None, timeout=8):
    base = context or driver
    try:
        elem = WebDriverWait(base, timeout).until(EC.presence_of_element_located((by, value)))
        txt = (elem.text or "").strip()
        return txt or None
    except Exception:
        return None


def clean(s, max_len=300):
    if not s:
        return None
    s = " ".join(s.split())
    return s[:max_len]


def looks_like_title(txt: str) -> bool:
    """Heuristic to avoid misclassifying a title as company."""
    title_words = (
        "Recruiter", "Engineer", "Manager", "Intern", "Specialist",
        "Director", "Lead", "Sr", "Senior", "Staff", "Associate",
        "Scientist", "Designer", "Consultant", "Analyst", "Architect",
    )
    t = txt.lower()
    return any(w.lower() in t for w in title_words)


def bad_company_marker(txt: str) -> bool:
    for m in ("United States", "Remote", "Hybrid", "On-site", "On site"):
        if m in txt:
            return True
    return False


# --------- driver setup ----------
def init_driver(headless: bool, chromedriver_path: str | None):
    opts = ChromeOptions()
    # stability/noise reduction flags
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-features=VizDisplayCompositor")

    if headless:
        opts.add_argument("--headless=new")
        # use explicit window-size with headless
        opts.add_argument("--window-size=1280,800")

    if chromedriver_path:
        service = ChromeService(executable_path=chromedriver_path, log_output="chromedriver.log")
        driver = webdriver.Chrome(service=service, options=opts)
    else:
        # Uses Selenium Manager (will try to download the right driver)
        driver = webdriver.Chrome(options=opts)

    driver.set_page_load_timeout(45)
    return driver


# --------- login + scraping -------
def login(driver, email: str, password: str):
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


# --- name/location from top card (do NOT use headline for position) ---
def extract_name_location(driver):
    # Name
    name = safe_text(
        driver, By.XPATH,
        "//h1[contains(@class,'text-heading-xlarge') or contains(@class,'text-heading-large')]"
    ) or safe_text(
        driver, By.XPATH,
        "//main//section[.//h1]//h1"
    )

    # Location: primary + fallback
    location = safe_text(
        driver, By.XPATH,
        "//span[contains(@class,'text-body-small') and contains(@class,'inline') and contains(@class,'break-words')]"
    ) or safe_text(
        driver, By.XPATH,
        "//div[contains(@class,'pv-text-details__left-panel')]//span[contains(@class,'text-body-small')]"
    )

    return clean(name), clean(location)

def scroll_to_experience(driver, tries=3):
    """
    Robustly scrolls the Experience section into view.
    Tries: find the <section> whose H2 contains 'Experience', then scrollIntoView.
    Falls back to incremental page scrolls if needed.
    """
    for attempt in range(tries):
        try:
            sec = WebDriverWait(driver, 8).until(
                EC.presence_of_element_located(
                    (By.XPATH, "//section[.//h2//span[contains(.,'Experience')]]")
                )
            )
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", sec)
            jitter(0.6, 1.0)
            # Wait for first item to render
            WebDriverWait(sec, 8).until(EC.presence_of_element_located((By.XPATH, ".//li[1]")))
            return sec
        except Exception:
            # gentle incremental scroll fallback
            driver.execute_script("window.scrollBy(0, Math.floor(window.innerHeight*0.6));")
            jitter(0.6, 1.0)
    return None

def extract_position_and_company_from_experience(driver):
    """
    1) Scroll to Experience (authoritative source).
    2) Detect two patterns:
       A) Single-role card: lines[0]=role, next non-meta is company.
       B) Company-group card: lines[0]=company, next title-like non-meta is role.
    """
    exp_section = scroll_to_experience(driver)
    if exp_section is None:
        return None, None

    try:
        first_li = WebDriverWait(exp_section, 10).until(
            EC.presence_of_element_located((By.XPATH, ".//li[1]"))
        )
    except Exception:
        return None, None

    # Pull visible text and normalize
    lines = [ln.strip() for ln in (first_li.text or "").splitlines() if ln.strip()]
    if not lines:
        return None, None

    def is_meta(line: str) -> bool:
        # Dates/durations/Present or location / work-mode / misc badges
        l = line.lower()
        if l.startswith(("helped me get this job", "currently hiring", "soon to be hiring", "filled")):
            return True
        has_digit = any(c.isdigit() for c in line)
        if has_digit and ("-" in line or "–" in line or "present" in l or "mos" in l or "yr" in l or "yrs" in l):
            return True
        if any(m in line for m in ("United States", "Remote", "Hybrid", "On-site", "On site")):
            return True
        # “Internship · 4 mos” style summary is meta
        if " · " in line and any(tok in l for tok in ("internship", "contract", "full-time", "part-time", "apprentice", "co-op")):
            return True
        return False

    def looks_like_title(txt: str) -> bool:
        title_words = (
            "Recruiter","Engineer","Manager","Intern","Specialist","Director","Lead","Sr","Senior","Staff",
            "Associate","Scientist","Designer","Consultant","Analyst","Architect","Owner","Founder","Executive",
            "Officer","Developer","Administrator","Coordinator","Assistant","Engineer I","Engineer II"
        )
        t = txt.lower()
        return any(w.lower() in t for w in title_words)

    # ---------- Detect layout ----------
    # Heuristic: if first line does NOT look like a title and is not meta,
    # treat it as a Company header (Layout B). Otherwise, Layout A.
    first = lines[0]
    layout_grouped_company = (not looks_like_title(first)) and (not is_meta(first))

    role = None
    company = None

    if layout_grouped_company:
        # Layout B: lines[0] is company; find the first title-like non-meta line after it
        company = first.split("·", 1)[0].strip()
        for ln in lines[1:]:
            if is_meta(ln):
                continue
            # Skip obvious section labels like "Experience" (defensive)
            if ln.lower() == "experience":
                continue
            if looks_like_title(ln):
                role = ln
                break
        # Fallback: if we never found a title-like line, treat the next non-meta as role
        if not role:
            for ln in lines[1:]:
                if not is_meta(ln):
                    role = ln
                    break
    else:
        # Layout A: first line is role; find next non-meta as company (and not same as role)
        role = first
        for ln in lines[1:]:
            if is_meta(ln):
                continue
            cand = ln.split("·", 1)[0].strip()
            if cand and cand.lower() != role.lower():
                company = cand
                break

        # Extra guard: if company accidentally looks like a title (rare), swap
        if company and looks_like_title(company) and not looks_like_title(role):
            role, company = company, role

    role = clean(role)
    company = clean(company) if company else None
    return role, company

def scrape_profile(driver, profile_url: str):
    driver.get(profile_url)
    wait_for(driver, (By.TAG_NAME, "main"))
    jitter()

    # Top card: name + location only
    name, location = extract_name_location(driver)

    # Experience: position + company (authoritative)
    position, company = extract_position_and_company_from_experience(driver)

    return {
        "name": name,
        "position": position,
        "company": company,
        "location": location,
        "url": profile_url,
    }


# ---------- CLI -------------------
def parse_args():
    p = argparse.ArgumentParser(description="LinkedIn profile scraper (Chrome)")
    p.add_argument("profile_url", help="Full LinkedIn profile URL, e.g. https://www.linkedin.com/in/USERNAME/")
    p.add_argument("--email", help="LinkedIn email (fallback to .env EMAIL)")
    p.add_argument("--password", help="LinkedIn password (fallback to .env PASSWORD)")
    p.add_argument("--headless", action="store_true", help="Run Chrome in headless mode")
    p.add_argument("--output", "-o", help="Write JSON to this file")
    p.add_argument("--chromedriver", help="Path to chromedriver.exe if Selenium Manager is blocked")
    return p.parse_args()


def main():
    load_dotenv()

    args = parse_args()
    email = args.email or os.getenv("EMAIL")
    password = args.password or os.getenv("PASSWORD")

    if not email or not password:
        print("ERROR: Missing credentials. Provide --email/--password or set EMAIL and PASSWORD in .env", file=sys.stderr)
        sys.exit(2)

    driver = None
    try:
        driver = init_driver(headless=args.headless, chromedriver_path=args.chromedriver)
        login(driver, email, password)
        data = scrape_profile(driver, args.profile_url)

        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            print(json.dumps(data, ensure_ascii=False, indent=2))
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


if __name__ == "__main__":
    main()
