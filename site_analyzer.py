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
        self.redirect_loops = 0  # Add counter for redirect loops

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
                
                # Check for redirect loops
                has_loop = len(set(chain)) < len(chain)
                if has_loop:
                    self.redirect_loops += 1
                
                self.redirect_chains[url] = {
                    'chain': chain,
                    'count': len(response.history),
                    'latency': latency,
                    'final_status': response.status_code,
                    'has_loop': has_loop
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
            'sitemap_issues': self.sitemap_issues,
            'redirect_loops': self.redirect_loops  # Add redirect loops to results
        }

class SiteCrawler:
    def __init__(self):
        self.driver = None
        self.setup_driver()
        
        # Initialize tracking dictionaries
        self.title_tracking = {}
        self.meta_desc_tracking = {}
        self.content_tracking = {}
        self.crawl_issues = defaultdict(int)
        self.canonical_issues = defaultdict(int)
        self.internal_links = defaultdict(set)
        self.inbound_links = defaultdict(set)
        self.page_depths = {}
        self.orphan_pages = set()
        
        # Initialize structured data tracking
        self.structured_data = {
            "schema_types": {
                "Organization": [],
                "LocalBusiness": [],
                "MedicalBusiness": [],
                "HealthAndBeautyBusiness": [],
                "Service": [],
                "Article": [],
                "BlogPosting": [],
                "WebPage": [],
                "FAQPage": [],
                "BreadcrumbList": [],
                "other": []
            },
            "implementation_methods": {
                "json_ld": {
                    "count": 0,
                    "valid": [],
                    "invalid": [],
                    "errors": []
                },
                "microdata": {
                    "count": 0,
                    "valid": [],
                    "invalid": [],
                    "errors": []
                },
                "rdfa": {
                    "count": 0,
                    "valid": [],
                    "invalid": [],
                    "errors": []
                }
            },
            "page_coverage": {
                "total_pages": 0,
                "pages_with_schema": 0,
                "pages_without_schema": []
            }
        }
        
        self.robots_txt_content = None
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
            
            # Add explicit timeout for page load
            try:
                self.driver.get(url)
            except Exception as e:
                print(f"Timeout or error loading page {url}: {str(e)}")
                return None
            
            # Wait for content to load with explicit timeout
            try:
                WebDriverWait(self.driver, 30).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                print(f"Body element found for {url}")
            except Exception as e:
                print(f"Timeout waiting for body element on {url}: {str(e)}")
                return None
                
            # Give extra time for dynamic content but with a shorter timeout
            try:
                time.sleep(2)
                print(f"Successfully loaded {url}")
                return self.driver.page_source
            except Exception as e:
                print(f"Error getting page source for {url}: {str(e)}")
                return None
            
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return None

    def extract_metadata(self, soup: BeautifulSoup, url: str) -> dict:
        """Extract detailed metadata from the page."""
        metadata = {
            "title_tag": "",
            "meta_description": "",
            "h_tags": defaultdict(list),
            "images": [],
            "structured_data": self.analyze_structured_data(soup, url)
        }

        # Get title
        if soup.title:
            title = soup.title.string.strip()
            metadata["title_tag"] = title
            # Track duplicate titles
            if title in self.title_tracking:
                self.crawl_issues["duplicate_titles"] += 1
                self.title_tracking[title].append(url)
            else:
                self.title_tracking[title] = [url]
        else:
            self.crawl_issues["urls_missing_title_tag"] += 1

        # Get meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            desc = meta_desc.get('content').strip()
            metadata["meta_description"] = desc
            # Track duplicate meta descriptions
            if desc in self.meta_desc_tracking:
                self.crawl_issues["duplicate_meta_descriptions"] += 1
                self.meta_desc_tracking[desc].append(url)
            else:
                self.meta_desc_tracking[desc] = [url]
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

        # Track duplicate content
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.find('div', id='content')
        if main_content:
            content_text = main_content.get_text(strip=True)
            # Use a simple hash of the content for comparison
            content_hash = hash(content_text)
            if content_hash in self.content_tracking:
                self.crawl_issues["duplicate_content"] += 1
                self.content_tracking[content_hash].append(url)
            else:
                self.content_tracking[content_hash] = [url]

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

    def analyze_internal_links(self, url: str, depth: int = 0, visited: set = None):
        """Analyze internal linking structure and update metrics."""
        if visited is None:
            visited = set()
            self.page_depths[url] = 0  # Homepage is at depth 0
        
        if url in visited:
            return
        
        visited.add(url)
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        try:
            content = self.get_page_content(url)
            if not content:
                return
                
            soup = BeautifulSoup(content, 'html.parser')
            links = soup.find_all('a', href=True)
            
            for link in links:
                href = link['href']
                try:
                    absolute_url = urljoin(url, href)
                    parsed_link = urlparse(absolute_url)
                    
                    # Skip invalid URLs and fragments
                    if not parsed_link.scheme or parsed_link.scheme.startswith(('mailto', 'tel', 'javascript')):
                        continue
                    
                    # Check if it's an internal link
                    is_internal = parsed_link.netloc == domain or parsed_link.netloc == f"www.{domain}"
                    
                    if is_internal:
                        # Add to internal links tracking
                        self.internal_links[url].add(absolute_url)
                        self.inbound_links[absolute_url].add(url)
                        
                        # Update page depth if this is a new page
                        if absolute_url not in self.page_depths:
                            self.page_depths[absolute_url] = depth + 1
                        
                        # Recursively analyze the linked page
                        if absolute_url not in visited:
                            self.analyze_internal_links(absolute_url, depth + 1, visited)
                            
                except Exception as e:
                    print(f"Error processing link {href}: {str(e)}")
                    
        except Exception as e:
            print(f"Error analyzing internal links for {url}: {str(e)}")

    def identify_orphan_pages(self):
        """Identify pages with no inbound links."""
        all_pages = set(self.internal_links.keys())
        for page in all_pages:
            if not self.inbound_links[page]:
                self.orphan_pages.add(page)

    def get_page_linking_metrics(self, url: str) -> dict:
        """Get linking metrics for a specific page."""
        return {
            "outbound_links": len(self.internal_links[url]),
            "inbound_links": len(self.inbound_links[url]),
            "depth": self.page_depths.get(url, -1)
        }

    def get_linking_metrics(self):
        """Get comprehensive internal linking metrics."""
        self.identify_orphan_pages()
        
        metrics = {
            "orphan_pages": list(self.orphan_pages),
            "depth_distribution": defaultdict(int),
            "total_pages": len(self.internal_links),
            "total_internal_links": sum(len(links) for links in self.internal_links.values())
        }
        
        # Calculate depth distribution
        for url in self.internal_links:
            depth = self.page_depths.get(url, -1)
            if depth >= 0:
                metrics["depth_distribution"][depth] += 1
        
        return metrics

    def crawl_site(self, url: str, depth: int = 0, visited: set = None) -> dict:
        """Analyze page and its linked pages up to specified depth."""
        if visited is None:
            visited = set()
        
        if url in visited:  # Remove depth limit check
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

            # Analyze internal linking structure
            self.analyze_internal_links(url)

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
                "metadata": metadata,
                "linking_metrics": self.get_page_linking_metrics(url)
            }
            
            # Add crawl issues summary with default values for all metrics
            results["crawl_issues_summary"] = {
                "urls_missing_title_tag": self.crawl_issues.get("urls_missing_title_tag", 0),
                "urls_missing_meta_description": self.crawl_issues.get("urls_missing_meta_description", 0),
                "urls_missing_h1": self.crawl_issues.get("urls_missing_h1", 0),
                "images_missing_alt_text": self.crawl_issues.get("images_missing_alt_text", 0),
                "duplicate_titles": self.crawl_issues.get("duplicate_titles", 0),
                "duplicate_meta_descriptions": self.crawl_issues.get("duplicate_meta_descriptions", 0),
                "duplicate_content": self.crawl_issues.get("duplicate_content", 0),
                "redirect_loops": self.technical_analyzer.redirect_loops
            }
            results["canonical_issues_summary"] = dict(self.canonical_issues)
            
            # Add technical analysis results
            technical_results = self.technical_analyzer.get_results()
            linking_metrics = self.get_linking_metrics()
            technical_results["orphan_pages"] = linking_metrics["orphan_pages"]
            technical_results["depth_distribution"] = dict(linking_metrics["depth_distribution"])
            results["technical_analysis"] = technical_results
            
            # Add internal linking summary to issues
            results["internal_linking_summary"] = {
                "total_pages": linking_metrics["total_pages"],
                "total_internal_links": linking_metrics["total_internal_links"],
                "orphan_pages_count": len(linking_metrics["orphan_pages"])
            }
            
            # Crawl linked pages if at depth 0, but limit to first 10 internal links
            if depth == 0:
                for linked_url in internal_links:  # Remove [:10] limit
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

    def analyze_structured_data(self, soup: BeautifulSoup, url: str) -> dict:
        """Analyze structured data on a page."""
        structured_data = {
            "schema_types": [],
            "implementation_method": None,
            "validation_status": "valid",
            "schema_content": None,
            "errors": []
        }
        
        # Check for JSON-LD
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        if json_ld_scripts:
            structured_data["implementation_method"] = "json_ld"
            self.structured_data["implementation_methods"]["json_ld"]["count"] += 1
            
            for script in json_ld_scripts:
                try:
                    schema = json.loads(script.string)
                    schema_type = schema.get("@type", "")
                    
                    # Handle array of schemas
                    if isinstance(schema, list):
                        for item in schema:
                            item_type = item.get("@type", "")
                            if item_type in self.structured_data["schema_types"]:
                                self.structured_data["schema_types"][item_type].append(url)
                            else:
                                self.structured_data["schema_types"]["other"].append(url)
                            structured_data["schema_types"].append(item_type)
                    else:
                        if schema_type in self.structured_data["schema_types"]:
                            self.structured_data["schema_types"][schema_type].append(url)
                        else:
                            self.structured_data["schema_types"]["other"].append(url)
                        structured_data["schema_types"].append(schema_type)
                    
                    structured_data["schema_content"] = schema
                    self.structured_data["implementation_methods"]["json_ld"]["valid"].append(url)
                    
                except json.JSONDecodeError as e:
                    structured_data["validation_status"] = "invalid"
                    structured_data["errors"].append(f"Invalid JSON-LD: {str(e)}")
                    self.structured_data["implementation_methods"]["json_ld"]["invalid"].append(url)
                    self.structured_data["implementation_methods"]["json_ld"]["errors"].append(str(e))
        
        # Check for Microdata
        if not structured_data["implementation_method"]:
            microdata = soup.find_all(attrs={"itemtype": True})
            if microdata:
                structured_data["implementation_method"] = "microdata"
                self.structured_data["implementation_methods"]["microdata"]["count"] += 1
                
                for item in microdata:
                    try:
                        schema_type = item.get("itemtype", "").split("/")[-1]
                        if schema_type in self.structured_data["schema_types"]:
                            self.structured_data["schema_types"][schema_type].append(url)
                        else:
                            self.structured_data["schema_types"]["other"].append(url)
                        structured_data["schema_types"].append(schema_type)
                        self.structured_data["implementation_methods"]["microdata"]["valid"].append(url)
                    except Exception as e:
                        structured_data["validation_status"] = "invalid"
                        structured_data["errors"].append(f"Invalid Microdata: {str(e)}")
                        self.structured_data["implementation_methods"]["microdata"]["invalid"].append(url)
                        self.structured_data["implementation_methods"]["microdata"]["errors"].append(str(e))
        
        # Check for RDFa
        if not structured_data["implementation_method"]:
            rdfa = soup.find_all(attrs={"vocab": True})
            if rdfa:
                structured_data["implementation_method"] = "rdfa"
                self.structured_data["implementation_methods"]["rdfa"]["count"] += 1
                
                for item in rdfa:
                    try:
                        schema_type = item.get("vocab", "").split("/")[-1]
                        if schema_type in self.structured_data["schema_types"]:
                            self.structured_data["schema_types"][schema_type].append(url)
                        else:
                            self.structured_data["schema_types"]["other"].append(url)
                        structured_data["schema_types"].append(schema_type)
                        self.structured_data["implementation_methods"]["rdfa"]["valid"].append(url)
                    except Exception as e:
                        structured_data["validation_status"] = "invalid"
                        structured_data["errors"].append(f"Invalid RDFa: {str(e)}")
                        self.structured_data["implementation_methods"]["rdfa"]["invalid"].append(url)
                        self.structured_data["implementation_methods"]["rdfa"]["errors"].append(str(e))
        
        # Update page coverage
        self.structured_data["page_coverage"]["total_pages"] += 1
        if structured_data["schema_types"]:
            self.structured_data["page_coverage"]["pages_with_schema"] += 1
        else:
            self.structured_data["page_coverage"]["pages_without_schema"].append(url)
        
        return structured_data

    def generate_output(self):
        """Generate the three output files with the analysis results."""
        # Technical Discovery Document
        technical_discovery = {
            "broken_links": self.technical_analyzer.broken_links,
            "redirect_chains": self.technical_analyzer.redirect_chains,
            "sitemap_issues": self.technical_analyzer.sitemap_issues,
            "orphan_pages": self.orphan_pages,
            "depth_distribution": self.get_linking_metrics()["depth_distribution"],
            "structured_data": {
                "schema_types": self.structured_data["schema_types"],
                "implementation_methods": self.structured_data["implementation_methods"],
                "page_coverage": self.structured_data["page_coverage"]
            }
        }
        
        with open(f"{self.domain}-technical-discovery.json", "w") as f:
            json.dump(technical_discovery, f, indent=2)

        # Generate issues file
        issues = {
            "crawl_issues": {
                "broken_links": self.technical_analyzer.broken_links,
                "redirect_chains": self.technical_analyzer.redirect_chains,
                "sitemap_issues": self.technical_analyzer.sitemap_issues,
                "orphan_pages": self.orphan_pages
            },
            "canonical_issues": {
                "missing_canonicals": self.canonical_issues["missing_canonical"],
                "invalid_canonicals": self.canonical_issues["invalid_canonical"],
                "self_referencing": self.canonical_issues["self_referencing"]
            },
            "internal_linking": {
                "total_pages": len(self.page_info),
                "total_links": sum(len(info["internal_links"]) for info in self.page_info.values())
            },
            "structured_data_issues": {
                "pages_without_schema": self.structured_data["page_coverage"]["pages_without_schema"],
                "invalid_implementations": {
                    "json_ld": len(self.structured_data["implementation_methods"]["json_ld"]["invalid"]),
                    "microdata": len(self.structured_data["implementation_methods"]["microdata"]["invalid"]),
                    "rdfa": len(self.structured_data["implementation_methods"]["rdfa"]["invalid"])
                }
            }
        }
        
        with open(f"{self.domain}-issues.json", "w") as f:
            json.dump(issues, f, indent=2)

        # Page Info Document
        page_info = {}
        for url, data in self.page_data.items():
            page_info[url] = {
                "indexability": data["indexability"],
                "metadata": data["metadata"],
                "linking_metrics": {
                    "outbound_links": len(data["outbound_links"]),
                    "inbound_links": len(self.inbound_links.get(url, [])),
                    "depth": self.page_depths.get(url, 0)
                },
                "structured_data": data["metadata"]["structured_data"]
            }
        
        with open(f"{self.domain}-page-info.json", "w") as f:
            json.dump(page_info, f, indent=2)

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
            "canonical_issues_summary": results["canonical_issues_summary"],
            "internal_linking_summary": results["internal_linking_summary"]
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