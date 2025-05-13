#!/usr/bin/env python3
"""
Site Analyzer – depth-1 crawler that extracts a company name, a 1-2 sentence
description, an engaging tagline, and a precise industry niche from a URL.
Results are dumped to a timestamped .txt file and printed to stdout.

Usage:
    python site_analyzer.py https://example.com
"""

import os
import re
import sys
import json
import time
import platform
import textwrap
from datetime import datetime
from collections import Counter
from urllib.parse import urljoin, urlparse
from tenacity import retry, stop_after_attempt, wait_exponential

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SiteAnalyzer:
    def __init__(self):
        self.setup_driver()
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.max_retries = 3
        self.retry_delay = 1  # seconds

    def setup_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Handle M1/M2 Macs
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            chrome_options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        
        try:
            service = Service("/opt/homebrew/bin/chromedriver")
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
        except Exception as e:
            print(f"Error setting up Chrome driver: {str(e)}")
            print("Please make sure Chrome is installed and up to date.")
            raise

    def get_page_content(self, url: str) -> str | None:
        try:
            self.driver.get(url)
            # Wait for content to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            # Give extra time for dynamic content
            time.sleep(2)
            return self.driver.page_source
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return None

    # ---------- Name --------------------------------------------------- #
    def extract_company_name(self, soup: BeautifulSoup, url: str) -> str:
        # Try to find company name from title
        title = soup.title.string if soup.title else ""
        if title:
            # Remove common suffixes and clean up
            name = re.sub(r'\s*[-|]\s*.*$', '', title)
            name = re.sub(r'\s*-\s*.*$', '', name)
            return name.strip()
            
        # Fallback to domain name
        domain = urlparse(url).netloc
        return domain.replace('www.', '').split('.')[0].title()

    # ---------- Description ------------------------------------------- #
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _call_gpt(self, messages: list, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """Make a GPT API call with retry logic."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API error: {str(e)}")
            raise

    def extract_company_description(self, soup: BeautifulSoup) -> str:
        # Try multiple meta tags in order of preference
        meta_tags = [
            ('meta', {'name': 'description'}),
            ('meta', {'property': 'og:description'}),
            ('meta', {'name': 'twitter:description'}),
            ('meta', {'property': 'og:title'}),
        ]
        
        # Collect all relevant text for GPT analysis
        text_content = []
        
        # Get meta descriptions
        for tag, attrs in meta_tags:
            meta = soup.find(tag, attrs=attrs)
            if meta and meta.get('content'):
                text_content.append(meta['content'])
        
        # Get main content
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'main|content|container'))
        if not main_content:
            main_content = soup.find('body')
        
        if main_content:
            # Get all text content
            text = main_content.get_text(separator=' ', strip=True)
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            # Split into sentences
            sentences = re.split(r'(?<=[.!?]) +', text)
            # Filter out very short sentences and navigation-like text
            sentences = [s for s in sentences if len(s.split()) > 3 and not any(x in s.lower() for x in ['menu', 'navigation', 'footer', 'copyright'])]
            text_content.extend(sentences[:5])  # Take first 5 meaningful sentences
        
        # Combine all text for GPT analysis
        combined_text = ' '.join(text_content)
        
        # Use GPT to generate a concise description
        try:
            return self._call_gpt([
                {"role": "system", "content": "You are a helpful assistant that creates concise, accurate company descriptions."},
                {"role": "user", "content": f"Based on this website content, create a 1-2 sentence description of what this company does. Be specific and factual:\n\n{combined_text}"}
            ])
        except Exception as e:
            print(f"Error generating description with GPT: {str(e)}")
            return "Description not found"

    # ---------- Tagline ------------------------------------------------ #
    def generate_tagline(self, description: str, company_name: str) -> str:
        try:
            return self._call_gpt([
                {"role": "system", "content": "You are a creative copywriter that creates engaging company taglines."},
                {"role": "user", "content": f"Create an engaging, memorable tagline for {company_name} based on this description. The tagline should be a complete statement that captures the company's mission or unique value proposition:\n\n{description}"}
            ], max_tokens=50, temperature=0.8)
        except Exception as e:
            print(f"Error generating tagline with GPT: {str(e)}")
            return f"{company_name} - Excellence delivered"

    # ---------- Niche -------------------------------------------------- #
    def determine_industry_niche(self, soup: BeautifulSoup, description: str) -> str:
        # Common industry keywords and their niches
        industry_keywords = {
            'restaurant': 'Food Service',
            'hotel': 'Hospitality',
            'law': 'Legal Services',
            'medical': 'Healthcare',
            'dental': 'Healthcare',
            'real estate': 'Real Estate',
            'construction': 'Construction',
            'retail': 'Retail',
            'technology': 'Technology',
            'education': 'Education',
            'fitness': 'Health & Fitness',
            'beauty': 'Beauty & Wellness',
            'automotive': 'Automotive',
            'financial': 'Financial Services',
            'insurance': 'Insurance',
            'manufacturing': 'Manufacturing',
            'consulting': 'Business Consulting',
            'marketing': 'Marketing & Advertising',
            'design': 'Design & Creative',
            'software': 'Software Development'
        }
        
        text = description.lower()
        for keyword, niche in industry_keywords.items():
            if keyword in text:
                return niche
                
        return "General Business"

    # ---------- Lightweight crawler ----------------------------------- #
    def crawl_site(self, url: str, depth: int = 1) -> str:
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(url, 0)]
        collected: list[str] = []

        while queue:
            current_url, lvl = queue.pop(0)
            if current_url in visited or lvl > depth:
                continue
            visited.add(current_url)

            html = self.get_page_content(current_url)
            if not html:
                continue

            soup = BeautifulSoup(html, "html.parser")
            main = (
                soup.find("main")
                or soup.find("article")
                or soup.find("div", class_=re.compile(r"(main|content|container)", re.I))
            )
            if main:
                collected.append(main.get_text(" ", strip=True))

            if lvl < depth:
                for a in soup.find_all("a", href=True):
                    nxt = urljoin(current_url, a["href"])
                    if urlparse(nxt).netloc == urlparse(url).netloc:
                        queue.append((nxt, lvl + 1))

        return " ".join(collected)

    # ------------------------------------------------------------------ #
    #  End-to-end analysis routine
    # ------------------------------------------------------------------ #
    def analyze_social_media(self, soup: BeautifulSoup) -> dict:
        """Analyze social media presence and links."""
        social_platforms = {
            'facebook': ['facebook.com', 'fb.com'],
            'twitter': ['twitter.com', 'x.com'],
            'linkedin': ['linkedin.com'],
            'instagram': ['instagram.com'],
            'youtube': ['youtube.com', 'youtu.be'],
            'tiktok': ['tiktok.com'],
            'pinterest': ['pinterest.com']
        }
        
        social_links = {}
        
        # Find all links
        for link in soup.find_all('a', href=True):
            href = link['href'].lower()
            for platform, domains in social_platforms.items():
                if any(domain in href for domain in domains):
                    social_links[platform] = href
                    break
        
        return social_links

    def analyze_tech_stack(self, soup: BeautifulSoup) -> dict:
        """Analyze the technology stack used by the website."""
        tech_stack = {
            'frameworks': [],
            'analytics': [],
            'cms': [],
            'hosting': []
        }
        
        # Check for common frameworks
        scripts = soup.find_all('script', src=True)
        for script in scripts:
            src = script['src'].lower()
            if 'react' in src:
                tech_stack['frameworks'].append('React')
            elif 'angular' in src:
                tech_stack['frameworks'].append('Angular')
            elif 'vue' in src:
                tech_stack['frameworks'].append('Vue.js')
            elif 'jquery' in src:
                tech_stack['frameworks'].append('jQuery')
        
        # Check for analytics
        if soup.find('script', src=re.compile(r'google-analytics|gtag')):
            tech_stack['analytics'].append('Google Analytics')
        if soup.find('script', src=re.compile(r'hotjar')):
            tech_stack['analytics'].append('Hotjar')
        
        # Check for CMS
        if soup.find('meta', {'name': 'generator', 'content': re.compile(r'wordpress', re.I)}):
            tech_stack['cms'].append('WordPress')
        elif soup.find('meta', {'name': 'generator', 'content': re.compile(r'shopify', re.I)}):
            tech_stack['cms'].append('Shopify')
        elif soup.find('meta', {'name': 'generator', 'content': re.compile(r'wix', re.I)}):
            tech_stack['cms'].append('Wix')
        
        return tech_stack

    def analyze_site(self, url: str) -> dict:
        """Analyze a website and return structured data."""
        try:
            # Load the page
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Get the page source and create BeautifulSoup object
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Extract basic information
            company_name = self.extract_company_name(soup, url)
            description = self.extract_company_description(soup)
            tagline = self.generate_tagline(description, company_name)
            industry_niche = self.determine_industry_niche(soup, description)
            
            # Analyze additional aspects
            social_media = self.analyze_social_media(soup)
            tech_stack = self.analyze_tech_stack(soup)
            
            return {
                'url': url,
                'company_name': company_name,
                'description': description,
                'tagline': tagline,
                'industry_niche': industry_niche,
                'social_media': social_media,
                'tech_stack': tech_stack,
                'analysis_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error analyzing site: {str(e)}")
            return None

    def close(self):
        self.driver.quit()


# ====================================================================== #
#  CLI
# ====================================================================== #
def main():
    url = None
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Enter the website URL to analyze: ")
    # Ensure URL starts with http or https
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    analyzer = SiteAnalyzer()
    try:
        results = analyzer.analyze_site(url)
        if results:
            print("\n=== Website Analysis Results ===\n")
            print(f"Company Name: {results['company_name']}")
            print(f"\nTagline: {results['tagline']}")
            print(f"\nDescription: {results['description']}")
            print(f"\nIndustry Niche: {results['industry_niche']}")
            
            if results['social_media']:
                print("\nSocial Media Presence:")
                for platform, link in results['social_media'].items():
                    print(f"- {platform.title()}: {link}")
            
            if any(results['tech_stack'].values()):
                print("\nTechnology Stack:")
                for category, technologies in results['tech_stack'].items():
                    if technologies:
                        print(f"- {category.title()}: {', '.join(technologies)}")
            
            print(f"\nAnalysis completed at: {results['analysis_date']}")
            
            # Save results to JSON file
            domain = urlparse(url).netloc.replace(':', '_')
            filename = f"site_analysis_{domain}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {filename}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        analyzer.close()


if __name__ == "__main__":
    main()