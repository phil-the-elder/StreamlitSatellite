import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pystac_client import Client
from odc.stac import load
import hmac


def check_password():
    """
    Returns `True` if the user had a correct password.

    This function handles the user authentication process. It displays a login form using Streamlit widgets,
    collects the user's username and password, and checks if the entered credentials match the stored ones.
    If the credentials are correct, it returns True. Otherwise, it displays an error message and returns False.

    Parameters:
    None

    Returns:
    bool: True if the user's credentials are correct, False otherwise.
    """

    def login_form():
        """
        Form with widgets to collect user information.

        This function creates a Streamlit form with text input fields for the username and password.
        It also includes a submit button to trigger the password validation.
        """
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """
        Checks whether a password entered by the user is correct.

        This function compares the entered password with the stored password for the given username.
        If the passwords match, it sets the `password_correct` flag in the session state to True.
        Otherwise, it sets the flag to False.
        """
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the username or password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False


def search_satellite_images(collection="sentinel-2-l2a",
                            bbox=[-120.15, 38.93, -119.88, 39.25],
                            date="2023-06-01/2023-06-30",
                            cloud_cover=(0, 10)):
    """
    This function searches for satellite images using the provided parameters.

    Parameters:
    collection (str): The name of the satellite image collection to search. Default is "sentinel-2-l2a".
    bbox (list): A list of four values representing the bounding box coordinates in the format
        [min_longitude, min_latitude, max_longitude, max_latitude]. Default is [-120.15, 38.93, -119.88, 39.25].
    date (str): The date range for the search in the format "YYYY-MM-DD/YYYY-MM-DD". Default is "2023-06-01/2023-06-30".
    cloud_cover (tuple): A tuple of two values representing the minimum and maximum cloud cover percentage.
        Default is (0, 10).

    Returns:
    data (xarray.Dataset): An xarray Dataset containing the satellite image data.
    """

    # Define the search client
    client = Client.open("https://earth-search.aws.element84.com/v1")
    search = client.search(collections=[collection],
                           bbox=bbox,
                           datetime=date,
                           query=[f"eo:cloud_cover<{cloud_cover[1]}", f"eo:cloud_cover>{cloud_cover[0]}"])

    # Print the number of matched items
    print(f"Number of images found: {search.matched()}")

    data = load(search.items(), bbox=bbox, groupby="solar_day", chunks={}, crs="EPSG:4326", resolution=0.0000900)

    print(f"Number of days in data: {len(data.time)}")

    return data


def get_bbox_with_buffer(latitude=37.2502, longitude=-119.7513, buffer=0.01):
    """
    This function calculates a bounding box with a given buffer around a specified latitude and longitude.

    Parameters:
    latitude (float): The central latitude of the bounding box. Default is 37.2502.
    longitude (float): The central longitude of the bounding box. Default is -119.7513.
    buffer (float): The buffer distance in decimal degrees to add around the central coordinates. Default is 0.01.

    Returns:
    list: A list of four values representing the bounding box coordinates in the format
        [min_longitude, min_latitude, max_longitude, max_latitude].
    """
    min_lat = latitude - buffer
    max_lat = latitude + buffer
    min_lon = longitude - buffer
    max_lon = longitude + buffer
    bbox = [min_lon, min_lat, max_lon, max_lat]
    return bbox


def count_classified_pixels(data, num):
    """
    Count the number of classified pixels in a given satellite image.

    This function takes an xarray Dataset `data` containing satellite image data and an integer `num`
    representing the time index of the image to be analyzed. It then counts the number of pixels belonging
    to different classes based on the values in the "scl" (Scene Classification) band of the image.

    Parameters:
    data (xarray.Dataset): An xarray Dataset containing satellite image data. It should have a "scl" band
        representing the Scene Classification.
    num (int): The time index of the image to be analyzed.

    Returns:
    dict: A dictionary containing the count of pixels for each class. The keys of the dictionary represent
        the class names, and the values represent the corresponding counts.
    """
    scl_image = data[["scl"]].isel(time=num).to_array()
    # Count the classified pixels
    count_saturated = np.count_nonzero(scl_image == 1)        # Saturated or defective
    count_dark = np.count_nonzero(scl_image == 2)             # Dark Area Pixels
    count_cloud_shadow = np.count_nonzero(scl_image == 3)     # Cloud Shadows
    count_vegetation = np.count_nonzero(scl_image == 4)       # Vegetation
    count_soil = np.count_nonzero(scl_image == 5)             # Bare Soils
    count_water = np.count_nonzero(scl_image == 6)            # Water
    count_clouds_low = np.count_nonzero(scl_image == 7)        # Clouds Low Probability / Unclassified
    count_clouds_med = np.count_nonzero(scl_image == 8)       # Clouds Medium Probability
    count_clouds_high = np.count_nonzero(scl_image == 9)      # Clouds High Probability
    count_clouds_cirrus = np.count_nonzero(scl_image == 10)   # Cirrus
    count_clouds_snow = np.count_nonzero(scl_image == 11)     # Snow

    counts = {
        'Dark/Bright': count_cloud_shadow + count_dark + count_clouds_low + count_clouds_med
        + count_clouds_high + count_clouds_cirrus + count_clouds_snow + count_saturated,
        'Vegetation': count_vegetation,
        'Bare Soil': count_soil,
        'Water': count_water,
    }

    return counts


def run_button(collection, start_date, end_date, max_cloud_cover, longitude, latitude, buffer):
    """
    This function handles the user's request to run a satellite image analysis. It prepares a new DataFrame
    with the user's query parameters, updates the existing DataFrame, displays the DataFrame in Streamlit,
    performs a search for satellite images based on the provided parameters, and stores the retrieved data
    in the session state.

    Parameters:
    collection (str): The name of the satellite image collection to search.
    start_date (datetime.date): The start date for the image search.
    end_date (datetime.date): The end date for the image search.
    max_cloud_cover (int): The maximum allowed cloud cover percentage.
    longitude (float): The central longitude of the area of interest.
    latitude (float): The central latitude of the area of interest.
    buffer (float): The buffer distance in decimal degrees to add around the central coordinates.

    Returns:
    None
    """
    new_df = pd.DataFrame(
            [
                {
                    "collection": collection,
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "max_cloud_cover": max_cloud_cover,
                    "longitude": longitude,
                    "latitude": latitude,
                    "buffer": buffer,
                }

            ]
        )

    st.session_state.mdf = pd.concat([st.session_state.mdf, new_df], axis=0)
    st.dataframe(st.session_state.mdf)
    st.success("Your request successfully submitted!")

    data = search_satellite_images(collection=collection,
                                   date=f"{start_date}/{end_date}",
                                   cloud_cover=(0, max_cloud_cover),
                                   bbox=get_bbox_with_buffer(latitude=latitude, longitude=longitude, buffer=buffer))
    st.session_state.data = data

    date_labels = []
    # Determine the number of time steps
    numb_days = len(data.time)
    # Iterate through each time step
    for t in range(numb_days):
        scl_image = data[["scl"]].isel(time=t).to_array()
        dt = pd.to_datetime(scl_image.time.values)
        year = dt.year
        month = dt.month
        day = dt.day
        date_string = f"{year}-{month:02d}-{day:02d}"
        date_labels.append(date_string)

    st.session_state.date_labels = date_labels


def list_button():
    """
    This function handles the user's request to select an available image from the list.
    It displays a dropdown menu with the available image dates and updates the session state
    with the selected date and its index.

    Parameters:
    None

    Returns:
    None
    """
    user_date = st.selectbox("Available Images*", options=st.session_state.date_labels, index=None)
    if user_date:
        st.session_state.user_date = user_date
        st.session_state.user_date_index = user_date.index()


def submit_button():
    """
    This function handles the user's request to visualize the selected satellite image.
    It creates a figure with two subplots: one for the RGB image and another for the pie chart
    representing the distribution of classes. The figure is then displayed in Streamlit.

    Parameters:
    None

    Returns:
    None
    """
    date_string_title = "Sentinel-2 Image over AOI"
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    rgb = st.session_state.data[["red", "green", "blue"]].isel(time=st.session_state.user_date_index).to_array()
    rgb.plot.imshow(robust=True, ax=axs[0])
    axs[0].axis('off')  # Hide the axes ticks and labels
    axs[0].set_title(date_string_title)

    # Preparing data
    counts = count_classified_pixels(st.session_state.data, st.session_state.user_date_index)
    labels = list(counts.keys())
    values = list(counts.values())
    colors = ['DarkGrey', 'chartreuse', 'DarkOrange', 'cyan']
    explode = (0.3, 0.1, 0.1, 0.1)  # Exploding the first slice

    # Plotting the pie chart
    axs[1].pie(values, labels=labels, colors=colors, autopct='%1.0f%%', startangle=140, explode=explode)
    axs[1].legend(labels, loc='best', bbox_to_anchor=(1, 0.5))
    axs[1].axis('equal')  # Ensure the pie chart is a circle
    axs[1].set_title('Distribution of Classes')

    # Display the figure in Streamlit
    st.pyplot(fig)


def main():
    if not check_password():
        st.stop()

    # Main Streamlit app starts here
    st.write("Satellite Image Analysis Portal")

    # Display Title
    st.title("Satellite Map Portal")
    st.markdown("Please set the image query parameters below.")

    # Initialize session state for date_labels and user_date
    if 'date_labels' not in st.session_state:
        st.session_state.date_labels = []

    if 'data' not in st.session_state:
        st.session_state.data = None

    if 'user_date' not in st.session_state:
        st.session_state.user_date = None

    if 'user_date_index' not in st.session_state:
        st.session_state.user_date_index = 0

    collections = ["sentinel-2-l2a"]
    columns = ['collection', 'start_date', 'end_date', 'min_cloud_cover',
               'max_cloud_cover', 'longitude', 'latitude', 'buffer']

    # Create an empty DataFrame with these columns
    df = pd.DataFrame(columns=columns)

    if "mdf" not in st.session_state:
        st.session_state.mdf = pd.DataFrame(columns=df.columns)

    # New Data
    with st.form(key="test"):
        collection = st.selectbox("collection*", options=collections, index=None)
        start_date = st.date_input(label="start_date*")
        end_date = st.date_input(label="end_date*")
        max_cloud_cover = st.number_input(label="max_cloud_cover*", value=10)
        longitude = st.number_input(label="longitude*", format="%.4f", value=-119.7513)
        latitude = st.number_input(label="latitude*", format="%.4f", value=37.2502)
        buffer = st.number_input(label="buffer (0.01 = 1 km)*", format="%.2f", value=0.01)

        # Mark Mandatory fields
        st.markdown("**required*")

        submit_button_run = st.form_submit_button(label="Run")
        submit_button_list = st.form_submit_button(label="List Available Images")
        submit_button_viz = st.form_submit_button(label="Visualize")

        if submit_button_run:
            run_button(collection, start_date, end_date, max_cloud_cover, longitude, latitude, buffer)

        if submit_button_list:
            list_button()

        if submit_button_viz:
            submit_button()


if __name__ == "__main__":
    main()
