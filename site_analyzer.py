#!/usr/bin/env python3
"""
Site Crawler – depth-1 crawler that extracts detailed site information and saves to JSON.

Usage:
    python site_analyzer.py https://example.com
"""

import os
import sys
import json
import time
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import platform
from collections import defaultdict
from tenacity import retry, stop_after_attempt, wait_exponential

class TechnicalAnalyzer:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; SiteAnalyzer/1.0)'
        })
        self.broken_links = defaultdict(list)
        self.redirect_chains = {}
        self.sitemap_issues = []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def check_url(self, url, is_internal=True):
        """Check URL status with retries and redirect following."""
        try:
            start_time = time.time()
            response = self.session.head(url, allow_redirects=True, timeout=10)
            latency = time.time() - start_time

            # Record redirect chain
            if len(response.history) > 0:
                chain = [r.url for r in response.history] + [response.url]
                self.redirect_chains[url] = {
                    'chain': chain,
                    'count': len(response.history),
                    'latency': latency,
                    'final_status': response.status_code
                }

            # Record broken links
            if response.status_code >= 400:
                self.broken_links[response.status_code].append({
                    'url': url,
                    'is_internal': is_internal,
                    'redirect_chain': self.redirect_chains.get(url, None)
                })

            return response
        except requests.RequestException as e:
            self.broken_links['error'].append({
                'url': url,
                'is_internal': is_internal,
                'error': str(e)
            })
            return None

    def validate_sitemap(self, domain, visited_sitemaps=None):
        """Validate XML sitemap and its URLs."""
        if visited_sitemaps is None:
            visited_sitemaps = set()
            
        sitemap_url = f"https://{domain}/sitemap.xml"
        if sitemap_url in visited_sitemaps:
            return
            
        visited_sitemaps.add(sitemap_url)
        
        try:
            response = self.session.get(sitemap_url, timeout=10)
            if response.status_code != 200:
                self.sitemap_issues.append(f"Sitemap not found at {sitemap_url}")
                return

            try:
                root = ET.fromstring(response.content)
                namespace = {'ns': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}

                # Check if it's a sitemap index
                if 'sitemapindex' in root.tag:
                    for sitemap in root.findall('.//ns:loc', namespace):
                        sitemap_domain = urlparse(sitemap.text).netloc
                        if sitemap_domain and sitemap_domain not in visited_sitemaps:
                            self.validate_sitemap(sitemap_domain, visited_sitemaps)
                else:
                    # Validate URLs in sitemap
                    for url in root.findall('.//ns:url', namespace):
                        loc = url.find('ns:loc', namespace)
                        if loc is not None:
                            url_to_check = loc.text
                            response = self.check_url(url_to_check)
                            if response and response.status_code not in [200, 301]:
                                self.sitemap_issues.append(f"Invalid status {response.status_code} for sitemap URL: {url_to_check}")

                        # Validate lastmod format if present
                        lastmod = url.find('ns:lastmod', namespace)
                        if lastmod is not None:
                            try:
                                datetime.fromisoformat(lastmod.text.replace('Z', '+00:00'))
                            except ValueError:
                                self.sitemap_issues.append(f"Invalid lastmod format in sitemap: {lastmod.text}")

                        # Validate priority if present
                        priority = url.find('ns:priority', namespace)
                        if priority is not None:
                            try:
                                prio = float(priority.text)
                                if not 0 <= prio <= 1:
                                    self.sitemap_issues.append(f"Invalid priority value in sitemap: {priority.text}")
                            except ValueError:
                                self.sitemap_issues.append(f"Invalid priority format in sitemap: {priority.text}")

            except ET.ParseError as e:
                self.sitemap_issues.append(f"Invalid XML in sitemap: {str(e)}")

        except requests.RequestException as e:
            self.sitemap_issues.append(f"Error fetching sitemap: {str(e)}")
        except Exception as e:
            self.sitemap_issues.append(f"Unexpected error processing sitemap: {str(e)}")

    def get_results(self):
        """Get technical analysis results."""
        return {
            'broken_links': dict(self.broken_links),
            'redirect_chains': self.redirect_chains,
            'sitemap_issues': self.sitemap_issues
        }

class SiteCrawler:
    def __init__(self):
        self.setup_driver()
        self.robots_txt_content = None
        self.crawl_issues = defaultdict(int)
        self.canonical_issues = defaultdict(int)
        self.technical_analyzer = TechnicalAnalyzer()

    def setup_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-popup-blocking")
        
        # Handle M1/M2 Macs
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            chrome_options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        
        try:
            # Try multiple possible ChromeDriver locations
            possible_paths = [
                "/opt/homebrew/bin/chromedriver",
                "/usr/local/bin/chromedriver",
                "./chromedriver"
            ]
            
            service = None
            for path in possible_paths:
                if os.path.exists(path):
                    service = Service(path)
                    break
            
            if service is None:
                print("ChromeDriver not found in common locations. Please install ChromeDriver and ensure it's in your PATH.")
                print("You can install it using: brew install chromedriver")
                raise Exception("ChromeDriver not found")
                
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.set_page_load_timeout(30)  # Set initial timeout
            print("ChromeDriver initialized successfully")
        except Exception as e:
            print(f"Error setting up Chrome driver: {str(e)}")
            print("Please make sure Chrome and ChromeDriver are installed and up to date.")
            print("For M1/M2 Macs, you can install ChromeDriver using: brew install chromedriver")
            raise

    def get_robots_txt(self, domain):
        """Fetch and parse robots.txt content."""
        if self.robots_txt_content is None:
            try:
                robots_url = f"{domain}/robots.txt"
                response = requests.get(robots_url, timeout=10)
                if response.status_code == 200:
                    self.robots_txt_content = response.text
                else:
                    self.robots_txt_content = ""
            except:
                self.robots_txt_content = ""
        return self.robots_txt_content

    def is_allowed_by_robots(self, url):
        """Check if URL is allowed by robots.txt."""
        if not self.robots_txt_content:
            return True  # If no robots.txt, assume allowed
        
        # Simple check for Disallow directives
        for line in self.robots_txt_content.split('\n'):
            if line.lower().startswith('disallow:'):
                path = line.split(':', 1)[1].strip()
                if path and url.endswith(path):
                    return False
        return True

    def get_page_content(self, url: str) -> str | None:
        """Get page content with timeout."""
        try:
            print(f"Fetching content for {url}...")
            self.driver.set_page_load_timeout(30)  # Increased timeout to 30 seconds
            self.driver.get(url)
            
            # Wait for content to load with explicit timeout
            try:
                WebDriverWait(self.driver, 30).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                print(f"Body element found for {url}")
            except Exception as e:
                print(f"Timeout waiting for body element on {url}: {str(e)}")
                return None
                
            # Give extra time for dynamic content
            time.sleep(2)
            print(f"Successfully loaded {url}")
            return self.driver.page_source
            
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return None

    def extract_metadata(self, soup: BeautifulSoup, url: str) -> dict:
        """Extract detailed metadata from the page."""
        metadata = {
            "title_tag": "",
            "meta_description": "",
            "h_tags": defaultdict(list),
            "images": []
        }

        # Get title
        if soup.title:
            metadata["title_tag"] = soup.title.string.strip()
        else:
            self.crawl_issues["urls_missing_title_tag"] += 1

        # Get meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            metadata["meta_description"] = meta_desc.get('content').strip()
        else:
            self.crawl_issues["urls_missing_meta_description"] += 1

        # Get heading tags
        for i in range(1, 7):
            for h in soup.find_all(f'h{i}'):
                metadata["h_tags"][f"h{i}"].append(h.get_text(strip=True))
        
        if not metadata["h_tags"]["h1"]:
            self.crawl_issues["urls_missing_h1"] += 1

        # Get images
        for img in soup.find_all('img'):
            img_data = {
                "src": img.get('src', ''),
                "alt_text": img.get('alt', '')
            }
            if not img_data["alt_text"]:
                self.crawl_issues["images_missing_alt_text"] += 1
            metadata["images"].append(img_data)

        return metadata

    def check_indexability(self, soup: BeautifulSoup, url: str) -> dict:
        """Check if page is indexable and get canonical URL."""
        indexability = {
            "robots_txt_allowed": self.is_allowed_by_robots(url),
            "meta_robots": "index,follow",  # Default
            "canonical": url,
            "canonical_self_referencing": True,
            "noindex_reason": None,
            "canonical_issues": []
        }

        # Check meta robots
        meta_robots = soup.find('meta', attrs={'name': 'robots'})
        if meta_robots and meta_robots.get('content'):
            indexability["meta_robots"] = meta_robots.get('content').lower()
            if 'noindex' in indexability["meta_robots"]:
                indexability["noindex_reason"] = "meta_robots_noindex"
                self.crawl_issues["non_indexable_urls"] += 1

        # Check canonical
        canonical = soup.find('link', attrs={'rel': 'canonical'})
        if canonical and canonical.get('href'):
            canonical_url = canonical.get('href')
            indexability["canonical"] = canonical_url
            
            # Check if canonical is relative
            if not canonical_url.startswith(('http://', 'https://')):
                indexability["canonical_issues"].append("relative_url")
                self.canonical_issues["relative_url"] += 1
            
            # Check if canonical points to different domain
            canonical_domain = urlparse(canonical_url).netloc
            current_domain = urlparse(url).netloc
            if canonical_domain and canonical_domain != current_domain:
                indexability["canonical_issues"].append("points_to_different_domain")
                self.canonical_issues["points_to_different_domain"] += 1
            
            # Check if canonical is self-referencing
            if canonical_url != url:
                indexability["canonical_self_referencing"] = False
                if not indexability["canonical_issues"]:  # Only add if no other issues
                    indexability["canonical_issues"].append("points_to_different_url")
                    self.canonical_issues["points_to_different_url"] += 1
        else:
            indexability["canonical_issues"].append("missing_canonical")
            self.canonical_issues["missing_canonical"] += 1

        return indexability

    def crawl_site(self, url: str, depth: int = 0, visited: set = None) -> dict:
        """Analyze page and its linked pages up to specified depth."""
        if visited is None:
            visited = set()
        
        if url in visited or depth > 1:  # Limit to depth 1
            return None
        
        visited.add(url)
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        base_url = f"{parsed_url.scheme}://{domain}"
        
        # Initialize results structure
        results = {
            "crawl_timestamp": datetime.now().isoformat(),
            "url_count": 1,
            "domain": base_url,
            "analyzed_url": url,
            "crawl_issues_summary": {},
            "canonical_issues_summary": {},
            "linked_pages": []
        }

        print(f"\nAnalyzing page: {url}")

        try:
            # Get robots.txt
            self.get_robots_txt(base_url)

            # Start sitemap validation
            self.technical_analyzer.validate_sitemap(domain)

            # Get page content
            content = self.get_page_content(url)
            if not content:
                print(f"No content received for {url}, skipping...")
                return results
                
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract metadata and check indexability
            metadata = self.extract_metadata(soup, url)
            indexability = self.check_indexability(soup, url)

            # Process all links on the page
            links = soup.find_all('a', href=True)
            print(f"Found {len(links)} links on {url}")
            
            internal_links = set()
            for link in links:
                href = link['href']
                try:
                    absolute_url = urljoin(url, href)
                    parsed_url = urlparse(absolute_url)
                    
                    # Skip invalid URLs and fragments
                    if not parsed_url.scheme or parsed_url.scheme.startswith(('mailto', 'tel', 'javascript')):
                        continue
                        
                    # Check if it's an internal or external link
                    is_internal = parsed_url.netloc == domain or parsed_url.netloc == f"www.{domain}"
                    
                    if is_internal and absolute_url not in visited:
                        internal_links.add(absolute_url)
                    
                    # Check URL status for all links
                    print(f"Checking URL: {absolute_url} (internal: {is_internal})")
                    self.technical_analyzer.check_url(absolute_url, is_internal)
                    
                except Exception as e:
                    print(f"Error processing link {href}: {str(e)}")

            # Add page info to results
            results["page_info"] = {
                "url": url,
                "indexability": indexability,
                "metadata": metadata
            }
            
            # Add crawl issues summary
            results["crawl_issues_summary"] = dict(self.crawl_issues)
            results["canonical_issues_summary"] = dict(self.canonical_issues)
            
            # Add technical analysis results
            results["technical_analysis"] = self.technical_analyzer.get_results()
            
            # Crawl linked pages if at depth 0
            if depth == 0:
                for linked_url in internal_links:
                    try:
                        linked_results = self.crawl_site(linked_url, depth + 1, visited)
                        if linked_results:
                            results["linked_pages"].append(linked_results)
                    except Exception as e:
                        print(f"Error crawling linked page {linked_url}: {str(e)}")
                        continue
            
            print(f"\nAnalysis complete for {url}")
            return results
            
        except Exception as e:
            print(f"Error analyzing page {url}: {str(e)}")
            return results

    def close(self):
        """Close the browser."""
        if hasattr(self, 'driver'):
            self.driver.quit()

def main():
    if len(sys.argv) != 2:
        print("Usage: python site_analyzer.py <url>")
        sys.exit(1)
            
    url = sys.argv[1]
    crawler = SiteCrawler()
    
    try:
        results = crawler.crawl_site(url)
        
        # Save main results to JSON file named after the domain
        domain = urlparse(url).netloc.replace(':', '_')
        
        # Save technical analysis
        tech_filename = f"{domain}-technical-discovery.json"
        with open(tech_filename, 'w', encoding='utf-8') as f:
            json.dump(results["technical_analysis"], f, indent=2, ensure_ascii=False)
        print(f"Technical analysis saved to {tech_filename}")
        
        # Save crawl and canonical issues
        issues_filename = f"{domain}-issues.json"
        issues_data = {
            "crawl_issues_summary": results["crawl_issues_summary"],
            "canonical_issues_summary": results["canonical_issues_summary"]
        }
        with open(issues_filename, 'w', encoding='utf-8') as f:
            json.dump(issues_data, f, indent=2, ensure_ascii=False)
        print(f"Issues summary saved to {issues_filename}")
        
        # Save page info
        page_info_filename = f"{domain}-page-info.json"
        with open(page_info_filename, 'w', encoding='utf-8') as f:
            json.dump(results["page_info"], f, indent=2, ensure_ascii=False)
        print(f"Page info saved to {page_info_filename}")
        
    finally:
        crawler.close()

if __name__ == "__main__":
    main()