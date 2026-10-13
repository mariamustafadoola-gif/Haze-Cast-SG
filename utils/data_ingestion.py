"""
DATA TEAM (data branch): download raw data from the three APIs.

To build:
- download_pm25(date)    NEA PM2.5 API (data.gov.sg), hourly readings per region
- download_fires(date)   NASA FIRMS area API, Sumatra and Kalimantan box, needs FIRMS_MAP_KEY in .env
- download_weather(date) Open-Meteo, hourly wind speed, wind direction and rain
Each saves the response unchanged under data/raw/<source>/<date>.
"""
