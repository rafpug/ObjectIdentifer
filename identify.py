import argparse
import base64
import contextlib
import threading
import json
import os
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from aiy.voice.tts import say
from aiy.vision.inference import CameraInference
from aiy.vision.models import image_classification
from picamera import PiCamera

from aiy.board import Board, Led

from click import prompt

ENDPOINT = "http://dgx03.ai.calpoly.edu:9300/v1/responses"
MODEL = "qwen3.6-27b"
IMAGE_PATH = "./photo.jpg"
PROMPT = "Based on this camera view mounted to a person's head, describe 5 objects in a paragraph as if you were a machine that identifies objects to blind people. Be clearcut without words unrelated to the objects. Any objects that can't be fully identified because of blur should be ignored. If there are more than 5 objects, you can group similar or a pile of objects together. If you have less than five, you can exclude them. The description of the location of objects should always be relative the to the blind person"

def make_request(api_key):
    with open(IMAGE_PATH, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("ascii")
    payload = {
        "model": MODEL,
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": PROMPT},
                {
                    "type": "input_image",
                    "image_url": "data:image/jpeg;base64," + encoded_image,
                    "detail": "high",
                },
            ],
        }],
    }

    request = Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
        },
        method="POST",
    )

    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))

def response_text(response):
    """Extract output text from a Responses API result."""
    for output in response.get("output", []):
        if output.get("type") != "message":
            continue
        for content in output.get("content", []):
            if content.get("type") == "output_text":
                return content.get("text", "")
    return "No text was returned. Full response:\n" + json.dumps(response, indent=2)

def main():
    api_key = os.environ.get("AI_API_KEY")
    if not api_key:
        sys.exit("Set AI_API_KEY first; do not put your API key in this script.")

    with PiCamera() as camera, Board() as board:
        while True:
            board.button.wait_for_press()
            camera.capture("photo.jpg")
            board.led.state = Led.ON
            try:
                print("Sending image to Qwen...")
                response = make_request(api_key)
            except (OSError, HTTPError, URLError) as error:
                sys.exit("Error: " + str(error))
            message = response_text(response)
            print("Qwen says: " + message)
            say(message)
            board.led.state = Led.OFF

if __name__ == '__main__':
    main()