from create_map import create_map


def take_screenshot(lat: float, long: float, row: int, col: int, number: int, file_name: str):
    """
    Args:
        lat: Latitude of the left corner
        long: Longitude of the left corner
        row: Row count
        col: Column count
        number: Numbering to output file
    Returns:
    """
    create_map(
        lat_start=lat,
        long_start=long,
        zoom=20,
        number_rows=row,
        number_cols=col,
        scale=1.0,
        sleep_time=2,
        offset_left=0.1666666,
        offset_top=0.1666666,
        offset_right=0.1666667,
        offset_bottom=0.1666667,
        outfile=file_name
    )



if __name__=='__main__':
    # Example: 5x5 -> 25 images
    Lat, Long = 40.000138757873195, -83.01825366047777  # 3x4, a smaller example map
    rows, cols = 3, 3
    
    # Lat, Long = 40.01835966827935, -83.03297664244631  # 30*17 Larger Map, 2.3km^2
    # rows, cols = 30, 17

    take_screenshot(
        lat=Lat,  # First image center latitude
        long=Long,  # First image center longitude
        row=rows,
        col=cols,
        file_name="image",  # Map image: "image-map-{number}.png"
        number=0
    )