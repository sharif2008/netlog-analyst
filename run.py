"""
Flask development server with auto-reload enabled
Run this file instead of app.py for better auto-reload support
"""
from app import app
import os

if __name__ == '__main__':
    # Set environment variables for Flask
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = '1'
    
    # Run with auto-reload
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        use_reloader=True,
        use_debugger=True,
        reloader_type='stat'  # Use stat watcher (works better on Windows)
    )

