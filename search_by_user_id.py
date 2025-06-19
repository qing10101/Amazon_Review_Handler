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


def scrape_user_reviews_truly_final(user_id):
    """
    Scrapes user reviews with the final, correct logic to extract the star
    rating from the element's class name.
    """
    reviews = []
    current_url = build_url(user_id)

    options = webdriver.ChromeOptions()
    # Comment out '--headless' to watch the script run.
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('start-maximized')
    options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    print(f"Navigating to: {current_url}")
    driver.get(current_url)

    parent_container_selector = "div#reviewTabContentContainer"
    review_card_selector = "div.review-card-container"

    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, parent_container_selector))
        )
        print("Success: Located the parent review container.")
    except Exception:
        print(f"Error: Timed out waiting for the container '{parent_container_selector}'.")
        driver.save_screenshot('final_error_screenshot.png')
        print("Saved a screenshot to 'final_error_screenshot.png'.")
        driver.quit()
        return []

    soup = BeautifulSoup(driver.page_source, 'html.parser')

    review_container = soup.select_one(parent_container_selector)
    if not review_container:
        print("Error: Could not parse the parent container. Exiting.")
        driver.quit()
        return []

    review_cards = review_container.select(review_card_selector)
    print(f"Found {len(review_cards)} review cards to process.")

    for card in review_cards:
        try:
            detail_container = card.select_one("div.review-detail-container")
            if not detail_container:
                continue

            # === RATING EXTRACTION: THE FINAL FIX ===
            rating_text = "Not found"
            # Find the <i> tag that has a class containing "a-star-"
            rating_tag = detail_container.select_one('i[class*="a-star-"]')
            if rating_tag:
                # The 'class' attribute is a list of strings, e.g., ['a-icon', 'a-icon-star', 'a-star-5', 'review-stars']
                class_list = rating_tag.get('class', [])
                # Find the specific class that tells us the rating
                for cls in class_list:
                    if cls.startswith('a-star-'):
                        # e.g., 'a-star-5' -> split by '-' -> ['a', 'star', '5'] -> take last part '5'
                        rating_value = cls.split('-')[-1]
                        rating_text = f"{rating_value} out of 5 stars"
                        break  # Exit the loop once we've found our rating

            review_data = {
                'review_title': detail_container.select_one("span.review-title").get_text(strip=True),
                'review_body': detail_container.select_one("span.review-description").get_text(strip=True),
                'rating': rating_text,
                'product_image_alt': card.select_one("img")['alt'],
                'review_link': 'https://www.amazon.com' + card.select_one("a.a-link-normal")['href']
            }
            reviews.append(review_data)
        except (AttributeError, TypeError):
            continue

    driver.quit()
    return reviews


def print_reviews(reviews_list):
    """Formats and prints the scraped review data."""
    if not reviews_list:
        print("\nScraping complete, but no valid reviews were found.")
        return

    print(f"\n--- Success! Scraped {len(reviews_list)} reviews ---\n")
    for i, review in enumerate(reviews_list, 1):
        print(f"--- Review {i} ---")
        print(f"Review Title: {review['review_title']}")
        print(f"Product: {review['product_image_alt']}")
        print(f"Rating: {review['rating']}")
        print(f"Snippet: {review['review_body']}")
        print(f"Link to Full Review: {review['review_link']}")
        print("-" * 20 + "\n")


if __name__ == "__main__":
    print("Amazon Review Finder (Truly Final Version)")
    print("=" * 45)
    user_id_input = input("Please enter the Amazon User ID: ").strip()
    if not user_id_input:
        print("No User ID entered. Exiting.")
    else:
        scraped_reviews = scrape_user_reviews_truly_final(user_id_input)
        print_reviews(scraped_reviews)