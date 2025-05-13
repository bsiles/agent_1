# Site Properties Analyzer

This script analyzes websites to extract key information including company name, description, tagline, and industry niche. It uses headless browser automation to crawl the site and extract relevant information.

## Features

- Extracts company name from page title or domain
- Generates a company description from meta tags or main content
- Creates an engaging tagline based on the company's content
- Determines the industry niche based on content analysis
- Saves results to a timestamped text file
- Crawls the site to a depth of 1 to gather comprehensive information

## Requirements

- Python 3.7+
- Chrome browser installed
- Required Python packages (install using `pip install -r requirements.txt`):
  - requests
  - beautifulsoup4
  - selenium
  - webdriver-manager
  - python-dotenv

## Installation

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the script:
```bash
python site_analyzer.py
```

When prompted, enter the URL of the website you want to analyze. The script will:
1. Crawl the website
2. Extract relevant information
3. Save the results to a text file
4. Display the results in the console

## Output

The script generates a text file with the following information:
- URL
- Company Name
- Description
- Tagline
- Industry Niche
- Analysis Timestamp

## Notes

- The script uses a headless Chrome browser, so you don't need to see the browser window
- It respects robots.txt and implements basic rate limiting
- The analysis is based on the main page and immediate linked pages (depth=1) 