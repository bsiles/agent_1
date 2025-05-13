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
from datetime import datetime, timedelta
from collections import Counter
from typing import Optional
from urllib.parse import urljoin, urlparse, urlunparse, parse_qs, urlencode
from tenacity import retry, stop_after_attempt, wait_exponential
from pathlib import Path
import hashlib
import io
import base64
from PIL import Image
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
import xml.etree.ElementTree as ET
import webcolors
from colorthief import ColorThief
import numpy as np
import utils.colors

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
        # Setup cache directory
        self.cache_dir = Path('cache')
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_expiry = timedelta(days=7)  # Cache for 7 days
        
        # Cache JavaScript snippets
        self.js = {
            'extract_css_vars': """
                function extractCSSVars() {
                    const vars = {};
                    for (let sheet of document.styleSheets) {
                        try {
                            for (let rule of sheet.cssRules) {
                                if (rule.type === CSSRule.STYLE_RULE) {
                                    for (let prop of rule.style) {
                                        if (prop.startsWith('--color-') && 
                                            (prop.includes('primary') || 
                                             prop.includes('secondary') || 
                                             prop.includes('accent') || 
                                             prop.includes('brand'))) {
                                            vars[prop] = rule.style.getPropertyValue(prop).trim();
                                        }
                                    }
                                }
                            }
                        } catch (e) {
                            // Skip cross-origin stylesheets
                        }
                    }
                    return vars;
                }
                return extractCSSVars();
            """,
            'dom_color_histogram': """
                function getComputedColors() {
                    const colors = {};
                    const elements = Array.from(document.querySelectorAll('*')).slice(0, 2000);
                    const props = ['color', 'backgroundColor', 
                                 'borderTopColor', 'borderRightColor',
                                 'borderBottomColor', 'borderLeftColor'];
                    
                    for (let el of elements) {
                        const style = window.getComputedStyle(el);
                        for (let prop of props) {
                            const color = style[prop];
                            if (color && color !== 'transparent' && 
                                color !== 'rgba(0, 0, 0, 0)') {
                                colors[color] = (colors[color] || 0) + 1;
                            }
                        }
                    }
                    return colors;
                }
                return getComputedColors();
            """
        }

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
        """Get page content with timeout."""
        try:
            self.driver.set_page_load_timeout(10)  # 10 second timeout
            self.driver.get(url)
            # Wait for content to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            # Give extra time for dynamic content
            time.sleep(1)  # Reduced from 2 to 1 second
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
                {"role": "user", "content": f"Create an engaging, memorable tagline for a company with the following description. Do NOT include the company name in the tagline. The tagline should be a complete statement that captures the company's mission or unique value proposition:\n\n{description}"}
            ], max_tokens=50, temperature=0.8)
        except Exception as e:
            print(f"Error generating tagline with GPT: {str(e)}")
            return "Excellence delivered"

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

    # ---------- Personality -------------------------------------------- #
    def extract_company_personality(self, description: str, company_name: str) -> str:
        """Use GPT to extract a 1-3 word description of the company's personality or tone."""
        try:
            prompt = (
                f"Based on the following company name and description, provide a 1-3 word description "
                f"of the company's personality or tone (e.g., 'playful and bold', 'professional', 'friendly', 'innovative', 'luxurious', etc.). "
                f"Only output the 1-3 word phrase, nothing else.\n\n"
                f"Company Name: {company_name}\nDescription: {description}"
            )
            return self._call_gpt([
                {"role": "system", "content": "You are an expert brand strategist."},
                {"role": "user", "content": prompt}
            ], max_tokens=10, temperature=0.7)
        except Exception as e:
            print(f"Error generating company personality with GPT: {str(e)}")
            return "Not found"

    # ---------- Lightweight crawler ----------------------------------- #
    def normalize_url(self, url: str) -> str:
        """Normalize a URL: lowercase scheme/host, ensure scheme, remove trailing slash, fragments, duplicate slashes."""
        parsed = urlparse(url)
        scheme = parsed.scheme.lower() if parsed.scheme else 'https'
        netloc = parsed.netloc.lower()
        path = re.sub(r'/+', '/', parsed.path)
        if path != '/' and path.endswith('/'):
            path = path.rstrip('/')
        normalized = urlunparse((scheme, netloc, path, '', '', ''))
        return normalized

    def crawl_site(self, url: str, depth: int = 1) -> dict:
        """Crawl site focusing on contact/about pages, with a maximum page limit. Normalize all URLs."""
        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(self.normalize_url(url), 0)]
        all_html: list[str] = []
        all_text: list[str] = []
        urls_visited: list[str] = []
        max_pages = 5  # Limit total pages to crawl
        
        print("\nCrawling pages...")
        
        while queue and len(visited) < max_pages:
            current_url, lvl = queue.pop(0)
            current_url = self.normalize_url(current_url)
            if current_url in visited or lvl > depth:
                continue
                
            visited.add(current_url)
            urls_visited.append(current_url)
            print(f"\rProgress: {len(visited)}/{max_pages} pages", end="", flush=True)

            html = self.get_page_content(current_url)
            if not html:
                continue
            all_html.append(html)

            soup = BeautifulSoup(html, "html.parser")
            main = (
                soup.find("main")
                or soup.find("article")
                or soup.find("div", class_=re.compile(r"(main|content|container)", re.I))
            )
            if main:
                all_text.append(main.get_text(" ", strip=True))
            else:
                all_text.append(soup.get_text(" ", strip=True))

            # Only follow links if we haven't hit the page limit
            if lvl < depth and len(visited) < max_pages:
                # Prioritize contact/about pages
                contact_links = []
                other_links = []
                
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if not href.startswith(('http://', 'https://', '/')):
                        continue
                        
                    nxt = urljoin(current_url, href)
                    nxt = self.normalize_url(nxt)
                    if urlparse(nxt).netloc != urlparse(url).netloc:
                        continue
                        
                    # Check if it's a contact/about page
                    if any(x in nxt.lower() for x in ['contact', 'about', 'info']):
                        contact_links.append(nxt)
                    else:
                        other_links.append(nxt)
                
                # Add contact links first, then other links
                queue.extend((link, lvl + 1) for link in contact_links)
                queue.extend((link, lvl + 1) for link in other_links)

        print("\nCrawl complete!")
        return {
            'html': '\n'.join(all_html),
            'text': '\n'.join(all_text),
            'urls_visited': urls_visited
        }

    # ------------------------------------------------------------------ #
    #  End-to-end analysis routine
    # ------------------------------------------------------------------ #
    def analyze_social_media(self, soup: BeautifulSoup) -> dict:
        """Analyze social media presence and links, normalizing URLs."""
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
                    social_links[platform] = self.normalize_url(href)
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

    def extract_key_services(self, description: str, company_name: str, main_content: str = "") -> list:
        """Use GPT to extract 3-6 core services offered by the company, each with a short description."""
        try:
            prompt = (
                f"Based on the following company name, description, and website content, list 3-6 core services offered by the company. "
                f"For each service, provide a short (1-2 sentence) description written as if the company wrote it themselves, playing up the value of the product. "
                f"Return as a JSON array of objects with 'service' and 'description' fields.\n\n"
                f"Company Name: {company_name}\nDescription: {description}\nWebsite Content: {main_content[:1000]}"
            )
            response = self._call_gpt([
                {"role": "system", "content": "You are a business analyst who summarizes company services."},
                {"role": "user", "content": prompt}
            ], max_tokens=350, temperature=0.7)
            # Try to parse the response as JSON
            import json as _json
            try:
                return _json.loads(response)
            except Exception:
                # If not valid JSON, return as a string in a list
                return [response]
        except Exception as e:
            print(f"Error extracting key services with GPT: {str(e)}")
            return []

    def extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract and clean main content from the page."""
        main_content = ""
        # Try to find main content area
        main = (
            soup.find("main") or 
            soup.find("article") or 
            soup.find("div", class_=re.compile(r"(main|content|container|hero|banner)", re.I))
        )
        
        if main:
            # Get all text content
            text = main.get_text(separator=' ', strip=True)
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            # Split into sentences
            sentences = re.split(r'(?<=[.!?]) +', text)
            # Filter out very short sentences and navigation-like text
            sentences = [s for s in sentences if len(s.split()) > 3 and not any(x in s.lower() for x in ['menu', 'navigation', 'footer', 'copyright', 'privacy policy', 'terms of service'])]
            main_content = ' '.join(sentences[:10])  # Take first 10 meaningful sentences
        
        return main_content

    def extract_css_vars(self) -> dict:
        """Extract CSS color variables from stylesheets."""
        try:
            vars = self.driver.execute_script(self.js['extract_css_vars'])
            return {
                name: utils.colors.hex_from_rgb(*utils.colors.rgb_from_css(value))
                for name, value in vars.items()
                if utils.colors.rgb_from_css(value)
            }
        except Exception as e:
            print(f"Error extracting CSS variables: {str(e)}")
            return {}

    def dom_color_histogram(self) -> Counter:
        """Get color frequency from computed styles."""
        try:
            colors = self.driver.execute_script(self.js['dom_color_histogram'])
            counter = Counter()
            for color, count in colors.items():
                rgb = utils.colors.rgb_from_css(color)
                if rgb:
                    hex_color = utils.colors.hex_from_rgb(*rgb)
                    counter[hex_color] += count
            return counter
        except Exception as e:
            print(f"Error getting DOM colors: {str(e)}")
            return Counter()

    def pixel_palette(self) -> list:
        """Extract dominant colors from page screenshot using K-means."""
        try:
            from sklearn.cluster import KMeans
            from PIL import Image
            import io
            
            # Get screenshot
            png = self.driver.get_screenshot_as_png()
            img = Image.open(io.BytesIO(png))
            
            # Convert to RGB array
            img_array = np.array(img)
            pixels = img_array.reshape(-1, 3)
            
            # Run K-means
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(pixels)
            
            # Get cluster centers and sizes
            centers = kmeans.cluster_centers_.astype(int)
            sizes = np.bincount(kmeans.labels_)
            
            # Convert to hex and filter neutrals
            colors = []
            for center, size in zip(centers, sizes):
                # Ensure center is a 1D array of ints
                center = [int(x.item()) if hasattr(x, 'item') else int(x) for x in center]
                hex_color = utils.colors.hex_from_rgb(*center)
                if not utils.colors.is_neutral(hex_color):
                    colors.append((hex_color, int(size.item()) if hasattr(size, 'item') else int(size)))
            
            # Sort by cluster size
            colors.sort(key=lambda x: x[1], reverse=True)
            return [color for color, _ in colors]
        except Exception as e:
            import traceback
            print(f"Error extracting pixel palette: {str(e)}\n{traceback.format_exc()}")
            return []

    def extract_theme_color(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract theme-color meta tag."""
        theme_color = soup.find('meta', attrs={'name': 'theme-color'})
        if theme_color and theme_color.get('content'):
            color = theme_color['content']
            rgb = utils.colors.rgb_from_css(color)
            if rgb:
                hex_color = utils.colors.hex_from_rgb(*rgb)
                if not utils.colors.is_neutral(hex_color):
                    return hex_color
        return None

    def select_primary_secondary(self, candidates: dict) -> dict:
        """Select primary and secondary colors from candidates."""
        result = {
            'primary_color': None,
            'secondary_color': None,
            'theme_color': None,
            'fallback_reason': None
        }
        
        # Always define dom_colors and pixel_colors
        dom_colors = candidates.get('dom_colors', [])
        pixel_colors = candidates.get('pixel_colors', [])
        
        # Get theme color if available
        theme_color = candidates.get('theme_color')
        if theme_color:
            result['theme_color'] = theme_color
        
        # Try CSS variables first
        css_vars = candidates.get('css_vars', {})
        primary_var = next((v for k, v in css_vars.items() if 'primary' in k), None)
        if primary_var:
            result['primary_color'] = primary_var
            result['fallback_reason'] = 'css-variable'
        
        # Try DOM histogram
        if not result['primary_color']:
            for color in dom_colors:
                if not utils.colors.is_neutral(color):
                    result['primary_color'] = color
                    result['fallback_reason'] = 'computed-histogram'
                    break
        
        # Try pixel palette
        if not result['primary_color']:
            for color in pixel_colors:
                if not utils.colors.is_neutral(color):
                    result['primary_color'] = color
                    result['fallback_reason'] = 'pixel-palette'
                    break
        
        # If still no primary, use theme color
        if not result['primary_color'] and theme_color:
            result['primary_color'] = theme_color
            result['fallback_reason'] = 'theme-color'
        
        # Find secondary color
        if result['primary_color']:
            # Try CSS variable
            secondary_var = next((v for k, v in css_vars.items() 
                                if 'secondary' in k and v != result['primary_color']), None)
            if secondary_var and utils.colors.delta_e(secondary_var, result['primary_color']) > 5:
                result['secondary_color'] = secondary_var
            else:
                # Try other candidates
                all_colors = (dom_colors + pixel_colors)
                for color in all_colors:
                    if (color != result['primary_color'] and 
                        utils.colors.hsl_distance(color, result['primary_color']) >= 20):
                        result['secondary_color'] = color
                        break
        
        # Check theme color override
        if (theme_color and result['primary_color'] and 
            utils.colors.delta_e(theme_color, result['primary_color']) >= 10):
            result['secondary_color'] = result['primary_color']
            result['primary_color'] = theme_color
            result['fallback_reason'] = 'theme-color-override'
        
        return result

    def extract_brand_colors(self, soup: BeautifulSoup) -> dict:
        """Extract brand colors using multiple methods."""
        # Get candidates from different methods
        candidates = {
            'css_vars': self.extract_css_vars(),
            'dom_colors': [color for color, _ in self.dom_color_histogram().most_common(10)],
            'pixel_colors': self.pixel_palette(),
            'theme_color': self.extract_theme_color(soup)
        }
        
        # Select primary and secondary colors
        return self.select_primary_secondary(candidates)

    def analyze_site(self, url: str) -> dict:
        """Analyze a website and return structured data."""
        try:
            url = self.normalize_url(url)
            print("\nStep 1/8: Loading page...")
            # Load the page
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            print("Step 2/8: Parsing page content...")
            # Get the page source and create BeautifulSoup object
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            print("Step 3/8: Extracting basic information...")
            # Extract basic information
            company_name = self.extract_company_name(soup, url)
            description = self.extract_company_description(soup)
            tagline = self.generate_tagline(description, company_name)
            industry_niche = self.determine_industry_niche(soup, description)
            personality = self.extract_company_personality(description, company_name)
            
            print("Step 4/8: Analyzing main content...")
            # Get main content for detailed analysis
            main_content = self.extract_main_content(soup)
            
            print("Step 5/8: Crawling site...")
            # Crawl the site and get aggregated data
            crawl_data = self.crawl_site(url)
            
            print("Step 6/8: Extracting key services...")
            # Extract additional insights using full crawl data
            key_services = self.extract_key_services(description, company_name, main_content)
            
            print("Step 7/8: Analyzing additional aspects...")
            # Analyze additional aspects
            social_media = self.analyze_social_media(soup)
            tech_stack = self.analyze_tech_stack(soup)
            color_scheme = self.extract_brand_colors(soup)
            
            print("Step 8/8: Finalizing analysis...")
            return {
                'url': url,
                'company_name': company_name,
                'description': description,
                'tagline': tagline,
                'industry_niche': industry_niche,
                'personality': personality,
                'key_services': key_services,
                'social_media': social_media,
                'tech_stack': tech_stack,
                'color_scheme': color_scheme,
                'analysis_date': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error analyzing site: {str(e)}")
            return None

    def close(self):
        self.driver.quit()

    def extract_colors_from_css(self, soup: BeautifulSoup) -> list:
        """Extract color values from CSS in the page."""
        colors = set()
        
        # Look for inline styles
        for tag in soup.find_all(style=True):
            style = tag['style']
            # Find color values in various formats
            color_matches = re.findall(r'#[0-9a-fA-F]{3,6}|rgb\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)', style)
            for color in color_matches:
                if color.startswith('#'):
                    colors.add(color)
                else:
                    # Convert rgb to hex
                    rgb = re.findall(r'\d+', color)
                    if len(rgb) == 3:
                        hex_color = '#{:02x}{:02x}{:02x}'.format(
                            int(rgb[0]), int(rgb[1]), int(rgb[2])
                        )
                        colors.add(hex_color)
        
        # Look for background images that might contain logos
        for tag in soup.find_all(['div', 'header', 'nav']):
            style = tag.get('style', '')
            if 'background-image' in style:
                url_match = re.search(r'url\([\'"]?(.*?)[\'"]?\)', style)
                if url_match:
                    try:
                        img_url = url_match.group(1)
                        if not img_url.startswith(('http://', 'https://')):
                            img_url = urljoin(soup.url, img_url)
                        response = requests.get(img_url, timeout=5)
                        if response.status_code == 200:
                            img = Image.open(io.BytesIO(response.content))
                            color_thief = ColorThief(io.BytesIO(response.content))
                            # Get the dominant color
                            dominant_color = color_thief.get_color(quality=1)
                            hex_color = '#{:02x}{:02x}{:02x}'.format(*dominant_color)
                            colors.add(hex_color)
                    except Exception:
                        continue
        
        return list(colors)

    def get_contrast_ratio(self, color1: str, color2: str) -> float:
        """Calculate the contrast ratio between two colors."""
        def get_luminance(color):
            # Convert hex to RGB
            r, g, b = webcolors.hex_to_rgb(color)
            # Convert to relative luminance
            r = r / 255
            g = g / 255
            b = b / 255
            r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
            g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
            b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        l1 = get_luminance(color1)
        l2 = get_luminance(color2)
        lighter = max(l1, l2)
        darker = min(l1, l2)
        return (lighter + 0.05) / (darker + 0.05)

    def find_contrasting_color(self, primary_color: str, potential_colors: list) -> str:
        """Find a color from the list that contrasts well with the primary color."""
        best_contrast = 0
        best_color = None
        
        for color in potential_colors:
            contrast = self.get_contrast_ratio(primary_color, color)
            if contrast > best_contrast:
                best_contrast = contrast
                best_color = color
        
        return best_color


# ====================================================================== #
#  CLI
# ====================================================================== #
def main():
    if len(sys.argv) != 2:
        print("Usage: python site_analyzer.py <url>")
        sys.exit(1)
            
    url = sys.argv[1]
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
            
    print(f"\nAnalyzing website: {url}")
    print("This may take a few moments...\n")
    
    analyzer = SiteAnalyzer()
    try:
        results = analyzer.analyze_site(url)
        if results:
            print("\n=== Website Analysis Results ===\n")
            print(f"Company Name: {results['company_name']}")
            print(f"\nTagline: {results['tagline']}")
            print(f"\nDescription: {results['description']}")
            print(f"\nIndustry Niche: {results['industry_niche']}")
            print(f"\nPersonality/Tone: {results['personality']}")
            
            if results.get('key_services'):
                print("\nKey Services:")
                for service in results['key_services']:
                    if isinstance(service, dict):
                        print(f"- {service.get('service', 'Service')}: {service.get('description', '')}")
                    else:
                        print(f"- {service}")
            
            if results['social_media']:
                print("\nSocial Media Presence:")
                for platform, link in results['social_media'].items():
                    print(f"- {platform.title()}: {link}")
            
            if any(results['tech_stack'].values()):
                print("\nTechnology Stack:")
                for category, technologies in results['tech_stack'].items():
                    if technologies:
                        print(f"- {category.title()}: {', '.join(technologies)}")
            
            if results.get('color_scheme'):
                print("\nColor Scheme:")
                print(f"Primary Color: {results['color_scheme']['primary_color']}")
                print(f"Secondary Color: {results['color_scheme']['secondary_color']}")
            
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