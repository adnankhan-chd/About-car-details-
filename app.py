"""Car Identifier - upload a car photo (or describe a car) and get its brand,
model and details, powered by Gemini Flash and Streamlit."""

import json
import os
from typing import List

import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field

# Change the model here, or set GEMINI_MODEL in env / Streamlit secrets.
DEFAULT_MODEL = "gemini-2.5-flash"
MAX_IMAGE_SIDE = 1600  # downscale big photos to keep requests fast and cheap


class CarInfo(BaseModel):
    is_car: bool = Field(description="True if a road vehicle (car, SUV, van or pickup) was identified")
    brand: str = Field(description="Manufacturer, e.g. Toyota. Empty string if unknown")
    model: str = Field(description="Model name, e.g. Corolla. Empty string if unknown")
    generation_or_years: str = Field(description="Generation or approximate production years, e.g. 'E210 (2018-2022)'")
    body_type: str = Field(description="Sedan, hatchback, SUV, coupe, pickup, etc.")
    fuel_type: str = Field(description="Petrol, diesel, hybrid, electric, etc. if determinable")
    engine_options: List[str] = Field(description="Typical engine/powertrain options for this model")
    drivetrain: str = Field(description="FWD, RWD, AWD, 4WD or unknown")
    country_of_origin: str = Field(description="Brand's home country or place of manufacture")
    notable_features: List[str] = Field(description="3-6 notable features or facts")
    confidence: str = Field(description="One of: high, medium, low")
    notes: str = Field(description="Short caveats, e.g. trim or year could not be determined")


PROMPT = (
    "You are an automotive expert. Identify the car from the provided image "
    "and/or description. Return the brand, model, generation/years and key "
    "details. If you are unsure, give your best estimate, set confidence to "
    "'low' or 'medium' and explain in notes. Never invent details; use an "
    "empty string or empty list when something is unknown. If the input does "
    "not show or describe a car, set is_car to false."
)


def get_secret(name: str, default: str = "") -> str:
    """Read from Streamlit secrets first, then environment variables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets file present
        pass
    return os.environ.get(name, default)


@st.cache_resource
def get_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def prepare_image(file) -> Image.Image:
    img = Image.open(file)
    img = img.convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return img


def identify_car(client, model_name: str, image=None, description: str = "") -> CarInfo:
    contents = [PROMPT]
    if description.strip():
        contents.append(f"User description: {description.strip()}")
    if image is not None:
        contents.append(image)

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CarInfo,
            temperature=0.2,
        ),
    )
    if not response.text:
        raise ValueError("The model returned an empty response (it may have been blocked).")
    return CarInfo(**json.loads(response.text))


def show_result(info: CarInfo) -> None:
    if not info.is_car:
        st.warning("I couldn't find a car in that input. Try a clearer photo or description.")
        return

    title = f"{info.brand} {info.model}".strip() or "Unknown car"
    st.subheader(title)
    st.caption(f"Confidence: **{info.confidence or 'unknown'}**")

    col1, col2 = st.columns(2)
    col1.metric("Brand", info.brand or "—")
    col2.metric("Model", info.model or "—")

    details = {
        "Generation / years": info.generation_or_years,
        "Body type": info.body_type,
        "Fuel type": info.fuel_type,
        "Drivetrain": info.drivetrain,
        "Country of origin": info.country_of_origin,
    }
    for label, value in details.items():
        st.markdown(f"**{label}:** {value or '—'}")

    if info.engine_options:
        st.markdown("**Engine options:**")
        for item in info.engine_options:
            st.markdown(f"- {item}")
    if info.notable_features:
        st.markdown("**Notable features:**")
        for item in info.notable_features:
            st.markdown(f"- {item}")
    if info.notes:
        st.info(info.notes)


def main() -> None:
    st.set_page_config(page_title="Car Identifier", page_icon="🚗", layout="centered")
    st.title("🚗 Car Identifier")
    st.write("Add a photo of a car and get its brand, model and details.")

    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        st.error(
            "GEMINI_API_KEY is not set. Add it to `.streamlit/secrets.toml` "
            "locally, or to your app's Secrets on Streamlit Community Cloud."
        )
        st.stop()
    model_name = get_secret("GEMINI_MODEL", DEFAULT_MODEL)

    tab_upload, tab_camera = st.tabs(["Upload photo", "Take photo"])
    with tab_upload:
        uploaded = st.file_uploader("Car photo", type=["jpg", "jpeg", "png", "webp"])
    with tab_camera:
        camera = st.camera_input("Take a photo of the car")
    description = st.text_input(
        "Optional: describe the car (helps when the photo is unclear)",
        placeholder="e.g. red 2015 hatchback with a three-door body",
    )

    source = uploaded or camera
    if source is not None:
        try:
            image = prepare_image(source)
        except Exception:
            st.error("That file couldn't be read as an image. Try a JPG or PNG.")
            st.stop()
        st.image(image)
    else:
        image = None

    if st.button("Identify car", type="primary"):
        if image is None and not description.strip():
            st.warning("Please add a photo or a description first.")
            return
        try:
            with st.spinner("Analyzing..."):
                info = identify_car(get_client(api_key), model_name, image, description)
            show_result(info)
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")


if __name__ == "__main__":
    main()
