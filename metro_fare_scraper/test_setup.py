"""
Test script to verify setup and check website accessibility
"""

import sys

def check_imports():
    """Check if all required packages are installed"""
    print("🔍 Checking Python packages...")
    
    packages = {
        'selenium': 'Selenium',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'plotly': 'Plotly',
        'requests': 'Requests',
        'beautifulsoup4': 'BeautifulSoup4'
    }
    
    missing = []
    installed = []
    
    for package, name in packages.items():
        try:
            if package == 'beautifulsoup4':
                __import__('bs4')
            else:
                __import__(package)
            installed.append(name)
            print(f"   ✅ {name}")
        except ImportError:
            missing.append(name)
            print(f"   ❌ {name} - NOT INSTALLED")
    
    print(f"\n📊 Summary: {len(installed)}/{len(packages)} packages installed")
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("   pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages are installed!")
        return True


def check_chrome():
    """Check if Chrome/Chromium is available"""
    print("\n🔍 Checking for Chrome/Chromium...")
    
    import subprocess
    
    browsers = [
        ('google-chrome', 'Google Chrome'),
        ('chromium-browser', 'Chromium'),
        ('chromium', 'Chromium')
    ]
    
    for cmd, name in browsers:
        try:
            result = subprocess.run([cmd, '--version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode == 0:
                print(f"   ✅ {name} found: {result.stdout.strip()}")
                return True
        except:
            pass
    
    print("   ⚠️  Chrome/Chromium not found")
    print("   Install: sudo apt install chromium-browser")
    return False


def check_network():
    """Check if website is accessible"""
    print("\n🔍 Checking website accessibility...")
    
    try:
        import requests
        
        url = "https://ltmetro.com/find-trip-details/"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print(f"   ✅ Website accessible: {url}")
            print(f"   Response time: {response.elapsed.total_seconds():.2f}s")
            return True
        else:
            print(f"   ⚠️  Website returned status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Cannot reach website: {str(e)}")
        print("   Check your internet connection")
        return False


def check_directories():
    """Check if required directories exist"""
    print("\n🔍 Checking project directories...")
    
    import os
    
    dirs = ['data', 'visualizations']
    
    for dir_name in dirs:
        if os.path.exists(dir_name):
            print(f"   ✅ {dir_name}/ exists")
        else:
            print(f"   ⚠️  {dir_name}/ missing - creating...")
            os.makedirs(dir_name)
            print(f"   ✅ Created {dir_name}/")
    
    return True


def main():
    """Run all checks"""
    print("="*70)
    print("  METRO FARE SCRAPER - SYSTEM CHECK")
    print("="*70)
    
    checks = []
    
    # Check packages
    checks.append(check_imports())
    
    # Check Chrome
    checks.append(check_chrome())
    
    # Check network
    checks.append(check_network())
    
    # Check directories
    checks.append(check_directories())
    
    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    
    if all(checks):
        print("✅ All checks passed! You're ready to scrape.")
        print("\nNext step:")
        print("   python scrape_metro_data.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("\nRun setup:")
        print("   ./setup.sh")
    
    print("="*70)


if __name__ == "__main__":
    main()
