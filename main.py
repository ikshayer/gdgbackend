# --- Imports ---
import os
import json
from flask import Flask, request, jsonify # Flask components
from flask_cors import CORS              # CORS handling
import google.generativeai as genai     # Google AI client library
import googlemaps                       # Google Maps client library
from dotenv import load_dotenv          # Loads environment variables

# --- Initialization ---

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

print("Loading environment variables...")
load_dotenv()

# --- Configure Google AI (Gemini) ---
google_api_key = os.getenv('GOOGLE_API_KEY')
gemini_model = None
if not google_api_key:
    print("ERROR: GOOGLE_API_KEY (for Gemini) environment variable not found.")
else:
    try:
        genai.configure(api_key=google_api_key)
        MODEL_NAME = 'gemini-1.5-flash-latest'
        gemini_model = genai.GenerativeModel(model_name=MODEL_NAME)
        print(f"Google AI SDK configured. Generative Model '{MODEL_NAME}' initialized.")
    except Exception as e:
        print(f"ERROR: Failed to configure Google AI SDK or initialize model: {e}")
        gemini_model = None # Ensure model is None if init fails


# --- Configure Google Maps ---
google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
gmaps = None
if not google_maps_api_key:
    print("ERROR: GOOGLE_MAPS_API_KEY environment variable not found.")
else:
    try:
        gmaps = googlemaps.Client(key=google_maps_api_key)
        print("Google Maps client initialized successfully.")
    except Exception as e:
        print(f"ERROR: Failed to initialize Google Maps client: {e}")
        gmaps = None # Ensure gmaps is None if init fails


# --- API Endpoint Definition ---
@app.route("/get_description", methods=["POST"])
def process_location_data():
    """
    Handles POST requests to /get_description endpoint.
    1. Receives lat, lon, alt, quaternion.
    2. Uses Google Maps Geocoding API to get location name/address from lat/lon.
    3. Queries Google AI (Gemini) using the location name and coordinates for context.
    4. Returns historical information as a JSON response.
    """
    print("\n--- Received request at /get_description ---")

    # Check if essential clients are initialized
    if gemini_model is None:
        print("ERROR: Gemini model not available. Cannot process request.")
        return jsonify({"error": "Backend configuration error: AI model not available"}), 500
    if gmaps is None:
        print("ERROR: Google Maps client not available. Cannot process request.")
        return jsonify({"error": "Backend configuration error: Maps client not available"}), 500

    # --- Request Body Parsing ---
    try:
        body = request.get_json()
        if not body:
            print("Error: Request body is empty or not valid JSON.")
            return jsonify({"error": "Request must contain a valid JSON body"}), 400
    except Exception as e:
        print(f"Error parsing request JSON: {e}")
        return jsonify({"error": f"Invalid JSON format in request body: {e}"}), 400

    # --- Extract Data ---
    latitude = body.get("latitude")
    longitude = body.get("longitude")
    altitude = body.get("altitude") # Received, but not used in this version
    quaternion = body.get("quaternion") # Received, but not used in this version

    print(f"Received Data: Lat={latitude}, Lon={longitude}, Alt={altitude}, Quat={quaternion}")

    if latitude is None or longitude is None:
        print("Error: Missing 'latitude' or 'longitude'.")
        return jsonify({"error": "Required fields 'latitude' and 'longitude' are missing"}), 400

    # --- Step 1: Google Maps Reverse Geocoding ---
    location_name = f"Coordinates {latitude}, {longitude}" # Default name
    geocoding_results = []
    try:
        print(f"Querying Google Maps Geocoding for: ({latitude}, {longitude})")
        # Perform reverse geocoding
        geocoding_results = gmaps.reverse_geocode((latitude, longitude))

        if geocoding_results:
            # Use the formatted address of the first result
            first_result = geocoding_results[0]
            location_name = first_result.get('formatted_address', location_name)
            print(f"Geocoding Result: Found location - {location_name}")
            # could potentially extract more specific info from 'address_components' if needed
            # e.g., find components with type 'point_of_interest' or 'establishment'
        else:
            print("Warning: Geocoding returned no results for the given coordinates.")
            # Keep the default location_name

    except googlemaps.exceptions.ApiError as e:
        print(f"ERROR: Google Maps API Error during geocoding: {e}")
        # Proceed with default location name
        location_name += " (Maps API Error)"
    except Exception as e:
        print(f"ERROR: Unexpected error during geocoding: {e}")
        # Proceed with default location name
        location_name += " (Geocoding Error)"


    # --- Step 2: AI Model Interaction with Context ---
    # Construct the prompt including the location name found via Geocoding
    prompt = (
        f"You are a historical geography expert. The user is observing a location identified as "
        f"'{location_name}' (precise coordinates: latitude={latitude}, longitude={longitude}). "
        f"Provide historical information and a concise summary about this specific location "
        f"or the nearest significant historical point of interest relevant to it. "
        f"Focus on information suitable for an Augmented Reality overlay. "
        f"Respond ONLY with a single, valid JSON object containing exactly two keys: "
        f'"summary" (string: a brief 1-2 sentence summary) and '
        f'"details" (string or list of strings: slightly more detailed historical points or facts). '
        "Strictly adhere to this JSON format: \{'summary': 'Example summary.', 'details': ['Detail 1.', 'Detail 2.']}"
    )

    print(f"\nSending Prompt to AI ({gemini_model.model_name}):\n{prompt}\n")

    try:
        # Send the prompt to the generative model
        response = gemini_model.generate_content(prompt)

        # --- AI Response Processing ---
        try:
            response_text = response.text
            print(f"Raw AI Response Text:\n{response_text}\n")
        except ValueError as e:
            print(f"Error accessing AI response text (Might be blocked): {e}")
            print(f"Prompt Feedback: {response.prompt_feedback}")
            return jsonify({"error": "AI response blocked or invalid", "details": str(e)}), 500
        except AttributeError:
            print("Error: AI response object missing 'text' attribute.")
            return jsonify({"error": "Unexpected AI response format"}), 500

        # --- JSON Parsing of AI Response ---
        try:
            # Basic cleanup for potential markdown wrappers
            cleaned_response_text = response_text.strip().removeprefix("```json").removesuffix("```").strip()
            data = json.loads(cleaned_response_text)
            print("Successfully parsed AI response as JSON:")
            print(json.dumps(data, indent=2))
            return jsonify(data), 200 # Success!

        except json.JSONDecodeError as json_error:
            print(f"Warning: AI response not valid JSON. Error: {json_error}")
            return jsonify({
                "raw_response": response_text,
                "warning": "AI response could not be parsed as valid JSON.",
                "error_details": str(json_error),
                "determined_location": location_name # Include the location we determined
            }), 200 # Return 200 but indicate format issue

    except Exception as e:
        print(f"An unexpected error occurred during AI interaction: {e}")
        return jsonify({"error": "An internal server error occurred processing the AI request"}), 500


# --- Server Execution ---
if __name__ == "__main__":
    print("\n--- Starting Flask Development Server ---")
    # Ensure both clients initialized before running
    if gemini_model and gmaps:
        app.run(host="0.0.0.0", port=5000, debug=True) # Set debug=False for production
    else:
        print("\n--- Server NOT started due to initialization errors ---")
    print("--- Flask Server Stopped ---")