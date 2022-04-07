import os
import time
import tkinter

import numpy as np
from PIL import Image
import pyscreenshot
from selenium import webdriver

# Removing fields from Google Maps
remove_from_view = [
    "var element = document.getElementById(\"omnibox-container\");element.remove();",
    "var element = document.getElementById(\"watermark\");element.remove();",
    "var element = document.getElementById(\"vasquette\");element.remove();",
    "var element = document.getElementsByClassName(\"app-viewcard-strip\");element[0].remove();",
    "var element = document.getElementsByClassName(\"scene-footer-container\");element[0].remove();",
    ]

# Removing labels from Google Maps Satellite View
remove_labels = [
    "document.getElementsByClassName(\"t9hXV-cdLCv-checkbox\")[1].click();",
]


def js_code_execute(driver, js_string: str):
    """Execute the JS code"""
    driver.execute_script(js_string)


def get_screen_resolution() -> tuple:
    """Return tuple of (width, height) of screen resolution in pixels."""
    root = tkinter.Tk()
    root.withdraw()
    return root.winfo_screenwidth(), root.winfo_screenheight()


def calc_latitude_shift(screen_height: int, percent_hidden: float, zoom: int) -> float: # up&down
    """Return the amount to shift latitude per row of screenshots."""
    return -0.000002968 * screen_height * (1 - percent_hidden) * (1 / 1.7 ** (zoom - 18)) # zoom=20
    # return -0.0000025235 * screen_height * (1 - percent_hidden) * (1 / 1.7 ** (zoom - 18)) # zoom=21


def calc_longitude_shift(screen_width: int, percent_hidden: float, zoom: int) -> float:# left&right
    """Return the amount to shift longitude per column of screenshots."""
    return 0.000003876 * screen_width * (1 - percent_hidden) * (1 / 1.7 ** (zoom - 18)) # zoom=10
    # return 0.0000032945 * screen_width * (1 - percent_hidden) * (1 / 1.7 ** (zoom - 18)) # zoom=21


def screenshot(screen_width: int, screen_height: int,
               offset_left: float, offset_top: float,
               offset_right: float, offset_bottom: float) -> Image:
    """Return a screenshot of only the pure maps area."""
    x1 = int(offset_left * screen_width)
    y1 = int(offset_top * screen_height)
    x2 = int((offset_right * -screen_width) + screen_width)
    y2 = int((offset_bottom * -screen_height) + screen_height)
    # image = pyscreenshot.grab()
    image = pyscreenshot.grab(bbox=(x1, y1, x2, y2))
    return image


def scale_image(image: Image, scale: float) -> Image:
    """Scale an Image by a proportion, maintaining aspect ratio."""
    width = round(image.width * scale)
    height = round(image.height * scale)
    image.thumbnail((width, height))
    return image


def create_map(lat_start: float, long_start: float, zoom: int,
               number_rows: int, number_cols: int,
               scale: float = 1, sleep_time: float = 0,
               offset_left: float = 0, offset_top: float = 0,
               offset_right: float = 0, offset_bottom: float = 0,
               outfile: str = None):
    # Create a map or satellite image given a waypoint
    """
    Args:
        lat_start: Top-left coordinate to start taking screenshots.
        long_start: Top-left coordinate to start taking screenshots.
        number_rows: Number of rows to take screenshot.
        number_cols: Number of columns to to create screenshot.
        scale: Percent to scale each image to reduce final resolution
            and filesize. Should be a float value between 0 and 1.
            Recommend to leave at 1 for production, and between 0.05
            and 0.2 for testing.
        sleep_time: Seconds to sleep between screenshots.
            Needed because Gmaps has some AJAX queries that will make
            the image better a few seconds after confirming page load.
            Recommend 0 for testing, and 3-5 seconds for production.
        offset_*: Percent of each side to crop from screenshots.
            Each should be a float value between 0 and 1. Offsets should
            account for all unwanted screen elements, including:
            taskbars, windows, multiple displays, and Gmaps UI (minimap,
            search box, compass/zoom buttons). Defaults are set for an
            Ubuntu laptop with left-side taskbar, and will need to be
            tuned to the specific machine and setup where it will be run.
        outfile: If provided, the program will save the final image to
            this filepath. Otherwise, it will be saved in the current
            working directory with name 'testing-<timestamp>.png'
        offset_right: Right offset.
        offset_top: Top offset.
        offset_bottom: Bottom offset.
        offset_left: Left offset.
    """

    # DRIVER Selection
    # Chromedriver should be in the current directory.
    # Modify these commands to find proper driver Chrome or Firefox
    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    
    driver = webdriver.Chrome(options=options, executable_path=r'./chromedriver.exe')

    driver.maximize_window()

    # Calculate amount to shift lat/long each screenshot
    screen_width, screen_height = get_screen_resolution()

    # Shifting values for lat and long
    lat_shift = calc_latitude_shift(screen_height, (offset_top + offset_bottom), zoom)
    long_shift = calc_longitude_shift(screen_width, (offset_left + offset_right), zoom)

    # Writing coordinates to the file
    f = open("./datasets/coordinates.txt", "w+")

    """
    i = 0 -> Map View
    i = 1 -> Satellite View
    """
    i = 1
    for row in range(number_rows):
        for col in range(number_cols):

            latitude = lat_start + (lat_shift * row)
            longitude = long_start + (long_shift * col)

            url = 'https://www.google.com/maps/'

            # Map URL
            if i == 0:
                url += '@{lat},{long},{z}z'.format(lat=latitude, long=longitude, z=zoom)
            # Satellite URL
            elif i == 1:
                url += '@{lat},{long},{z}z/data=!3m1!1e3'.format(lat=latitude, long=longitude, z=zoom)

            driver.get(url)
            time.sleep(5)

            # Remove labels from Satellite view
            if i == 1:
                js_code_execute(driver, remove_labels[0])
                time.sleep(3)
                # js_code_execute(driver, remove_labels[1])

            # Remove fields from Map view
            for j in remove_from_view:
                js_code_execute(driver, j)

            # Let the map load all assets before taking a screenshot
            time.sleep(sleep_time)
            image = screenshot(screen_width, screen_height, offset_left, offset_top, offset_right, offset_bottom)

            # Scale image up or down if desired, then save in memory
            image = scale_image(image, scale)
            map_path = './datasets/satellite_imgs/sat_%d_%d.png'%(row, col)
            image.save(map_path)
            f.write(f'{row}_{col}\tLat={latitude}\tLong={longitude}\n')

    # Close the browser
    driver.close()
    driver.quit()
    f.close()