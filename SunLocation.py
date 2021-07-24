# -*- coding: utf-8 -*-
import requests

response = requests.get('https://api.ipgeolocation.io/astronomy?apiKey=f83b30d358004c03ba575afcbf4cbaea&lat=41.933010&long=-87.640518')
if(response.status_code == 200):
    try:
        body = response.json()
        print(body['sun_altitude'])
        print(body['sun_azimuth'])
        print(body['sun_distance'])
    except (json.JSONDecodeError, ValueError):
        pass
