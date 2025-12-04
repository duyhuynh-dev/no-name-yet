#!/usr/bin/env python
"""Simple HTTP server for the frontend."""

import http.server
import socketserver
import os
import sys
from pathlib import Path

# Change to frontend directory
frontend_dir = Path(__file__).parent.parent / "frontend"
os.chdir(frontend_dir)

PORT = 3000

class CORSHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with CORS headers."""
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

with socketserver.TCPServer(("", PORT), CORSHandler) as httpd:
    print(f"🚀 Frontend server running at http://localhost:{PORT}")
    print(f"📁 Serving files from: {frontend_dir}")
    print("Press Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        sys.exit(0)

