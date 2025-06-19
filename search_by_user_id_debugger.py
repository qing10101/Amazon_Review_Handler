import time
import sys
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


def build_url(user_id):
    """Constructs the URL for the user's review page."""
    return f"https://www.amazon.com/gp/profile/amzn1.account.{user_id}/ref=cm_cr_arp_d_pdp?ie=UTF8&type=review"


def scrape_user_reviews_diagnostic(user_id):
    """
    Attempts to scrape reviews and, upon failure, saves the page source for manual inspection.
    """
    reviews = []
    current_url = build_url(user_id)

    options = webdriver.ChromeOptions()
    # Keep the browser visible for this diagnostic run
    options.add_argument('start-maximized')
    options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    print(f"Navigating to: {current_url}")
    driver.get(current_url)

    page_num = 1
    review_selector = "div[data-hook='profile-review']"

    try:
        # Step 1: Wait for the main review container to be present.
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, review_selector))
        )
        print("Success: Found what appears to be a review container.")

    except Exception:
        # Step 2: If the container isn't found, save the HTML and exit.
        print("\n" + "=" * 50)
        print("DIAGNOSTIC MODE: Could not find the review container on the main page.")

        page_source = driver.page_source
        with open("amazon_page_source.html", "w", encoding="utf-8") as f:
            f.write(page_source)

        print("ACTION REQUIRED: An HTML file named 'amazon_page_source.html' has been saved.")
        print("Please inspect this file to find the correct class names or data-hooks for the reviews.")
        print("It's also possible the content is in an <iframe>.")
        print("=" * 50 + "\n")

        driver.quit()
        return []

    # If the script gets here, the container was found, so now we parse.
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    review_blocks = soup.select(review_selector)

    if not review_blocks:
        print("Warning: The review container was located, but parsing found no individual reviews.")
        driver.quit()
        return []

    print(f"Found {len(review_blocks)} review blocks on page {page_num}.")
    for block in review_blocks:
        try:
            review_data = {
                'product_title': block.select_one('a.a-link-normal').get_text(strip=True),
                'product_link': 'https://www.amazon.com' + block.select_one('a.a-link-normal')['href'],
                'rating': block.select_one('i[data-hook="review-star-rating"]').get_text(strip=True),
                'review_title': block.select_one('a[data-hook="review-title"]').get_text(strip=True),
                'review_date': block.select_one('span[data-hook="review-date"]').get_text(strip=True),
                'review_body': block.select_one('span[data-hook="review-body"]').get_text(strip=True)
            }
            reviews.append(review_data)
        except AttributeError:
            # This error means one of the inner selectors (like 'review-title') is wrong.
            print("Warning: Could not fully parse a review block. The inner structure may have changed.",
                  file=sys.stderr)
            continue

    driver.quit()
    return reviews


def print_reviews(reviews_list):
    # This function is unchanged
    if not reviews_list:
        print("\nNo reviews were ultimately found or scraped.")
        return
    print(f"\n--- Found {len(reviews_list)} reviews ---\n")
    for i, review in enumerate(reviews_list, 1):
        print(f"--- Review {i} ---")
        print(f"Product: {review['product_title']}")
        print(f"Rating: {review['rating']}")
        print(f"Date: {review['review_date']}")
        print(f"Title: {review['review_title']}")
        print(f"Review: {review['review_body']}")
        print(f"Product Link: {review['product_link']}")
        print("-" * 20 + "\n")


if __name__ == "__main__":
    # This section is unchanged
    print("Amazon Review Finder (Diagnostic Edition)")
    print("=" * 35)
    user_id_input = input("Please enter the Amazon User ID: ").strip()
    if not user_id_input:
        print("No User ID entered. Exiting.")
    else:
        scraped_reviews = scrape_user_reviews_diagnostic(user_id_input)
        print_reviews(scraped_reviews)