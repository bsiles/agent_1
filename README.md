# Site Analyzer

A powerful web crawler and analyzer that provides comprehensive technical analysis of websites.

## Features

- Depth-limited crawling (default depth 1)
- Technical SEO analysis
- Content structure analysis
- Link validation
- Sitemap validation
- Performance metrics
- Accessibility checks

## Requirements

- Python 3.7+
- Chrome/Chromium browser
- ChromeDriver

## Installation

1. Install Chrome/Chromium browser
2. Install ChromeDriver:
   ```bash
   # For Mac with Homebrew
   brew install chromedriver
   ```
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
python site_analyzer.py https://example.com
```

## Output

The script generates three JSON files:
- `{domain}-technical-discovery.json`: Technical analysis results
- `{domain}-issues.json`: Crawl and canonical issues summaries
- `{domain}-page-info.json`: Page metadata and structure info

## Analysis Details

### Technical Analysis
- URL status checking
- Redirect chain tracking
- Latency measurement
- Internal vs external link classification
- Sitemap validation

### SEO Analysis
- Title tag presence
- Meta description presence
- H1 tag presence
- Image alt text checking
- Canonical URL validation

### Content Structure
- Heading hierarchy analysis
- Image inventory
- Link structure
- Content organization

## License

MIT License 