#!/usr/bin/env python3
"""
Simple HTTP server to serve the network visualization files.
This avoids CORS issues when loading CSV files from JavaScript.
"""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

# Configuration
PORT = 8000
DIRECTORY = Path(__file__).parent

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler with CORS headers."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)
    
    def end_headers(self):
        """Add CORS headers to allow local file access."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()
    
    def log_message(self, format, *args):
        """Custom logging to show accessed files."""
        if args[1] == '200':
            print(f"✓ Served: {args[0]}")
        else:
            super().log_message(format, *args)

def main():
    """Start the HTTP server and open the browser."""
    
    # Check if required files exist
    required_files = ['nodes.csv', 'edges.csv', 'index.html']
    missing_files = [f for f in required_files if not (DIRECTORY / f).exists()]
    
    if missing_files:
        print(f"❌ Error: Missing required files: {', '.join(missing_files)}")
        print(f"   Current directory: {DIRECTORY}")
        sys.exit(1)
    
    print("=" * 60)
    print("🚀 Starting Hyderabad Network Visualization Server")
    print("=" * 60)
    print(f"📁 Serving files from: {DIRECTORY}")
    print(f"🌐 Server running at: http://localhost:{PORT}")
    print()
    print("📊 Comprehensive Dashboard:")
    print(f"   • Main Dashboard: http://localhost:{PORT}/index.html")
    print(f"   • Direct access:  http://localhost:{PORT}")
    print()
    print("💡 Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    # Create server
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        # Open browser
        url = f"http://localhost:{PORT}/index.html"
        print(f"🌍 Opening browser: {url}")
        webbrowser.open(url)
        
        print()
        print("✅ Server is ready! Your browser should open automatically.")
        print("   If not, manually open: http://localhost:{PORT}/index.html")
        print()
        
        # Serve forever
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n")
            print("=" * 60)
            print("🛑 Server stopped by user")
            print("=" * 60)
            sys.exit(0)

if __name__ == "__main__":
    main()
