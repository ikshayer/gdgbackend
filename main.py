# backend_server.py

# --- Imports ---
import os                           # For accessing environment variables
import json                         # For working with JSON data
import logging                      # For better logging practices
from flask import Flask, request, jsonify # Flask components for web server, requests, JSON responses
from flask_cors import CORS         # Handles Cross-Origin Resource Sharing
import google.generativeai as genai # Google AI (Gemini) client library
import googlemaps                   # Google Maps client library
from dotenv import load_dotenv      # Loads environment variables from a .env file

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialization ---
logging.info("Initializing Flask application...")
app = Flask(__name__)

# Configure CORS to allow requests from any origin (adjust for production)
CORS(app, resources={r"/*": {"origins": "*"}})
logging.info("CORS configured to allow all origins.")

# Load environment variables from .env file
logging.info("Loading environment variables from .env file...")
load_dotenv()

# --- Configure Google AI (Gemini) ---
google_api_key = os.getenv('GOOGLE_API_KEY')
gemini_model = None
if not google_api_key:
    logging.error("CRITICAL: GOOGLE_API_KEY (for Gemini) environment variable not found.")
else:
    try:
        genai.configure(api_key=google_api_key)
        # Use a known stable or latest recommended model
        MODEL_NAME = 'gemini-1.5-flash-latest'
        gemini_model = genai.GenerativeModel(model_name=MODEL_NAME)
        logging.info(f"Google AI SDK configured. Generative Model '{MODEL_NAME}' initialized.")
    except Exception as e:
        logging.error(f"CRITICAL: Failed to configure Google AI SDK or initialize model: {e}", exc_info=True)
        gemini_model = None # Ensure model is None if init fails


# --- Configure Google Maps ---
google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
gmaps_client = None
if not google_maps_api_key:
    logging.error("CRITICAL: GOOGLE_MAPS_API_KEY environment variable not found.")
else:
    try:
        gmaps_client = googlemaps.Client(key=google_maps_api_key)
        logging.info("Google Maps client initialized successfully.")
    except Exception as e:
        logging.error(f"CRITICAL: Failed to initialize Google Maps client: {e}", exc_info=True)
        gmaps_client = None # Ensure gmaps is None if init fails


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
    logging.info("--- Received request at /get_description ---")

    # --- Check if clients are ready ---
    if gemini_model is None:
        logging.error("Gemini model not available. Cannot process request.")
        return jsonify({"error": "Backend configuration error: AI model not available"}), 503 # Service Unavailable
    if gmaps_client is None:
        logging.error("Google Maps client not available. Cannot process request.")
        return jsonify({"error": "Backend configuration error: Maps client not available"}), 503 # Service Unavailable

    # --- Request Body Parsing ---
    try:
        body = request.get_json()
        if not body:
            logging.warning("Request body is empty or not valid JSON.")
            return jsonify({"error": "Request must contain a valid JSON body"}), 400 # Bad Request
    except Exception as e:
        logging.warning(f"Error parsing request JSON: {e}", exc_info=True)
        return jsonify({"error": f"Invalid JSON format in request body: {e}"}), 400 # Bad Request


    # --- Extract Data ---
    latitude = body.get("latitude")
    longitude = body.get("longitude")
    altitude = body.get("altitude") # Received, not used in prompts currently
    quaternion_list = body.get("quaternion") # Received, not used in prompts currently

    logging.info(f"Received Data: Lat={latitude}, Lon={longitude}, Alt={altitude}, Quat={quaternion_list}")

    # --- Validate Coordinates ---
    if latitude is None or longitude is None:
        logging.warning("Missing 'latitude' or 'longitude' in request.")
        return jsonify({"error": "Required fields 'latitude' and 'longitude' are missing"}), 400

    # Optional: Basic validation of coordinate values
    try:
        lat_f = float(latitude)
        lon_f = float(longitude)
        if not (-90 <= lat_f <= 90 and -180 <= lon_f <= 180):
            raise ValueError("Coordinates out of range")
    except (ValueError, TypeError):
        logging.warning(f"Invalid latitude or longitude values received: {latitude}, {longitude}")
        return jsonify({"error": "Invalid latitude or longitude values provided."}), 400


    # --- Step 1: Google Maps Reverse Geocoding ---
    location_name = f"Coordinates {latitude:.6f}, {longitude:.6f}" # Default name if geocoding fails
    geocoding_results = []
    try:
        logging.info(f"Querying Google Maps Geocoding for: ({latitude}, {longitude})")
        # Perform reverse geocoding using the initialized client
        geocoding_results = gmaps_client.reverse_geocode((latitude, longitude))

        if geocoding_results:
            # Use the formatted address of the first, most specific result
            first_result = geocoding_results[0]
            location_name = first_result.get('formatted_address', location_name)
            logging.info(f"Geocoding Result: Found location - '{location_name}'")
            # Optional: Log more details if needed for debugging
            # logging.debug(f"Full Geocoding Result[0]: {first_result}")
        else:
            logging.warning("Geocoding returned no results for the given coordinates.")
            # Keep the default location_name

    except googlemaps.exceptions.ApiError as e:
        logging.error(f"Google Maps API Error during geocoding: {e}", exc_info=False) # Avoid logging key in stack usually
        location_name += " (Maps API Error)" # Append note to name
    except Exception as e:
        logging.error(f"Unexpected error during geocoding: {e}", exc_info=True)
        location_name += " (Geocoding Error)"


    # --- Step 2: AI Model Interaction with Context ---
    # Construct the prompt using the location name from Geocoding for better context
    prompt = (
        f"You are a historical geography expert providing information for an Augmented Reality application. "
        f"The user is looking at a location identified as '{location_name}' "
        f"(precise coordinates: latitude={latitude}, longitude={longitude}). "
        f"Provide interesting historical information and a concise summary about this specific location "
        f"or the most relevant nearby historical point of interest. "
        f"Keep the language engaging and suitable for a brief AR overlay. "
        f"Respond ONLY with a single, valid JSON object containing exactly two keys: "
        f'"summary" (string: a very brief, 1-2 sentence summary for immediate display) and '
        f'"details" (string or list of strings: 2-4 slightly more detailed historical facts or points, suitable for expansion or secondary display). '
        f'Strictly adhere to this JSON format: {{"summary": "Example summary.", "details": ["Detail 1.", "Detail 2."]}}'
    )

    logging.info(f"\nSending Prompt to AI ({gemini_model.model_name}):\n{prompt}\n")

    try:
        # Send the prompt to the Gemini model
        response = gemini_model.generate_content(
            prompt,
            # Optional: Add safety settings if needed
            # safety_settings=[
            #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            # ]
        )

        # --- AI Response Processing ---
        try:
            # Check for safety blocks or other issues before accessing .text
            # Accessing parts might raise ValueError if blocked
            response_text = response.text
            logging.info(f"Raw AI Response Text:\n{response_text}\n")
        except ValueError as e:
            # This often indicates blocked content due to safety filters
            logging.error(f"Error accessing AI response text. Likely blocked due to safety settings: {e}")
            try:
                logging.warning(f"Prompt Feedback: {response.prompt_feedback}")
                # You might want to inspect response.candidates[0].finish_reason and .safety_ratings
                # logging.warning(f"Candidate Finish Reason: {response.candidates[0].finish_reason}")
                # logging.warning(f"Candidate Safety Ratings: {response.candidates[0].safety_ratings}")
            except Exception as feedback_e:
                logging.error(f"Error accessing prompt feedback details: {feedback_e}")
            return jsonify({"error": "AI response blocked due to safety settings or other issue.", "details": str(e)}), 400 # Bad Request might be appropriate if prompt was bad
        except AttributeError:
            logging.error("AI response object missing 'text' attribute. Unexpected format.")
            return jsonify({"error": "Unexpected AI response format from Google"}), 502 # Bad Gateway

        # --- JSON Parsing of AI Response ---
        try:
            # Attempt basic cleanup of markdown code fences sometimes added by models
            cleaned_response_text = response_text.strip()
            if cleaned_response_text.startswith("```json"):
                cleaned_response_text = cleaned_response_text[7:]
            if cleaned_response_text.endswith("```"):
                cleaned_response_text = cleaned_response_text[:-3]
            cleaned_response_text = cleaned_response_text.strip()

            # Parse the cleaned text as JSON
            data = json.loads(cleaned_response_text)

            # Optional: Validate the structure of the parsed JSON
            if not isinstance(data, dict) or "summary" not in data or "details" not in data:
                logging.warning("AI response parsed as JSON, but missing required 'summary' or 'details' keys.")
                raise json.JSONDecodeError("Missing required keys in AI JSON response", cleaned_response_text, 0)

            logging.info("Successfully parsed AI response as valid JSON:")
            logging.info(json.dumps(data, indent=2)) # Pretty print parsed JSON
            return jsonify(data), 200 # HTTP 200 OK - Success

        except json.JSONDecodeError as json_error:
            # Handle cases where the AI response text is not valid JSON
            logging.warning(f"AI response could not be parsed as valid JSON. Error: {json_error}")
            logging.warning(f"Original text was: {response_text}")
            # Return the raw text *and* the determined location, indicating the format issue
            return jsonify({
                "raw_response": response_text, # Send original problematic text
                "warning": "AI response could not be parsed as valid JSON.",
                "error_details": str(json_error),
                "determined_location": location_name # Include the location we found
            }), 200 # 200 OK because *we* processed it, but let client know AI format failed

    except Exception as e:
        # Catch any other unexpected errors during the AI call or processing
        logging.error(f"An unexpected error occurred during AI interaction: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred processing the AI request"}), 500 # Internal Server Error


# --- Server Execution ---
if __name__ == "__main__":
    logging.info("--- Starting Flask Development Server ---")
    # Ensure essential clients initialized before starting the server
    if gemini_model and gmaps_client:
        # host='0.0.0.0': Makes the server accessible on your network
        # port=5000: Standard Flask development port
        # debug=True: Enables auto-reload and detailed errors (DISABLE FOR PRODUCTION)
        app.run(host="0.0.0.0", port=5000, debug=True)
    else:
        logging.critical("\n--- Server NOT started due to critical initialization errors (API Keys or SDK setup) ---")
        logging.critical("--- Please check errors above and ensure .env file is correct ---")

    logging.info("--- Flask Server Stopped ---")