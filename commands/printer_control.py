from octorest import OctoRest
import requests
from requests import api

apikey = "55E925556E164A0B837E6E59EF335952"
octoprint_server = "https://alexqm.me/aquila"

def make_client():
    try:
        client = OctoRest(url=octoprint_server, apikey=apikey)
        return client
    except Exception as e:
        print(e)

def _control_PSU(command):
    try:
        r = requests.post(octoprint_server + "/api/plugin/psucontrol", headers={"X-Api-Key": apikey, "Authorization": "Basic YWxleDpBeF96ZDQ="}, json={"command": command})
        r.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(err)
        raise err

def turn_PSU_on():
    try:
        _control_PSU("turnPSUOn")
    except Exception as e:
        raise e

def turn_PSU_off():
    try:
        _control_PSU("turnPSUOff")
    except Exception as e:
        raise e