import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin, urlparse
import time
import json

def fetch_page(url, timeout=10, headers=None):
    """
    Fetch HTML content from a URL
    
    Args:
        url (str): The URL to scrape
        timeout (int): Request timeout in seconds
        headers (dict): Optional custom headers
    
    Returns:
        BeautifulSoup object or None if failed
    """
    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None


def extract_text(soup, selector=None):
    """
    Extract all text from a page or specific elements
    
    Args:
        soup (BeautifulSoup): Parsed HTML
        selector (str): CSS selector for specific elements
    
    Returns:
        str or list: Extracted text
    """
    if selector:
        elements = soup.select(selector)
        return [elem.get_text(strip=True) for elem in elements]
    return soup.get_text(strip=True)


def extract_links(soup, base_url=None, filter_external=False):
    """
    Extract all links from a page
    
    Args:
        soup (BeautifulSoup): Parsed HTML
        base_url (str): Base URL for converting relative links
        filter_external (bool): Only return internal links
    
    Returns:
        list: List of URLs
    """
    links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if base_url:
            href = urljoin(base_url, href)
        
        if filter_external and base_url:
            if urlparse(href).netloc != urlparse(base_url).netloc:
                continue
        
        links.append(href)
    
    return list(set(links))  # Remove duplicates


def extract_images(soup, base_url=None):
    """
    Extract all image URLs from a page
    
    Args:
        soup (BeautifulSoup): Parsed HTML
        base_url (str): Base URL for converting relative links
    
    Returns:
        list: List of image URLs with alt text
    """
    images = []
    for img in soup.find_all('img'):
        src = img.get('src', '')
        if base_url and src:
            src = urljoin(base_url, src)
        
        images.append({
            'src': src,
            'alt': img.get('alt', ''),
            'title': img.get('title', '')
        })
    
    return images


def extract_table(soup, table_index=0):
    """
    Extract data from HTML tables
    
    Args:
        soup (BeautifulSoup): Parsed HTML
        table_index (int): Index of table to extract (0 for first table)
    
    Returns:
        pandas DataFrame or None
    """
    tables = soup.find_all('table')
    
    if not tables or table_index >= len(tables):
        print(f"Table at index {table_index} not found")
        return None
    
    table = tables[table_index]
    
    # Extract headers
    headers = []
    header_row = table.find('thead')
    if header_row:
        headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
    else:
        first_row = table.find('tr')
        if first_row:
            headers = [th.get_text(strip=True) for th in first_row.find_all(['th', 'td'])]
    
    # Extract rows
    rows = []
    for tr in table.find_all('tr')[1:]:  # Skip header row
        cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
        if cells:
            rows.append(cells)
    
    if not headers:
        headers = [f"Column_{i}" for i in range(len(rows[0]))]
    
    return pd.DataFrame(rows, columns=headers)


def extract_custom_data(soup, selectors):
    """
    Extract custom data using CSS selectors
    
    Args:
        soup (BeautifulSoup): Parsed HTML
        selectors (dict): Dictionary of {field_name: css_selector}
    
    Returns:
        list: List of dictionaries with extracted data
    """
    results = []
    
    # Find the maximum number of elements for any selector
    max_elements = 0
    selector_results = {}
    
    for field, selector in selectors.items():
        elements = soup.select(selector)
        selector_results[field] = elements
        max_elements = max(max_elements, len(elements))
    
    # Build result list
    for i in range(max_elements):
        item = {}
        for field, elements in selector_results.items():
            if i < len(elements):
                item[field] = elements[i].get_text(strip=True)
            else:
                item[field] = None
        results.append(item)
    
    return results


def scrape_multiple_pages(urls, delay=1, **kwargs):
    """
    Scrape multiple URLs with delay between requests
    
    Args:
        urls (list): List of URLs to scrape
        delay (int): Delay in seconds between requests
        **kwargs: Additional arguments to pass to fetch_page
    
    Returns:
        list: List of BeautifulSoup objects
    """
    soups = []
    for i, url in enumerate(urls):
        print(f"Scraping {i+1}/{len(urls)}: {url}")
        soup = fetch_page(url, **kwargs)
        if soup:
            soups.append(soup)
        
        if i < len(urls) - 1:  # Don't delay after last request
            time.sleep(delay)
    
    return soups


def save_to_json(data, filename):
    """Save scraped data to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Data saved to {filename}")


def save_to_csv(data, filename):
    """Save scraped data to CSV file"""
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Data saved to {filename}")


# Example usage
if __name__ == "__main__":
    # Example 1: Basic scraping
    url = "https://en.wikipedia.org/wiki/One_Piece"
    soup = fetch_page(url)
    
    if soup:
        # Extract all text
        text = extract_text(soup)
        print("Page text:", text[:200])
        
        # Extract all links
        links = extract_links(soup, base_url=url)
        print(f"\nFound {len(links)} links")
        
        # Extract images
        images = extract_images(soup, base_url=url)
        print(f"Found {len(images)} images")
    
    # Example 2: Extract specific elements
    selectors = {
        'title': 'h1',
        'description': 'p.description',
        'price': 'span.price'
    }
    # data = extract_custom_data(soup, selectors)
    # save_to_json(data, 'scraped_data.json')
    
    # Example 3: Extract table
    # table_df = extract_table(soup, table_index=0)
    # if table_df is not None:
    #     table_df.to_csv('table_data.csv', index=False)
    
    # Example 4: Scrape multiple pages
    # urls = ['https://example.com/page1', 'https://example.com/page2']
    # soups = scrape_multiple_pages(urls, delay=2)