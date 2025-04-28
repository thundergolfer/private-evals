import json
import pathlib
import sys
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

RESULTS_FILE = "results.json"

def load_results() -> dict:
    f = pathlib.Path(RESULTS_FILE)
    if not f.exists():
        print(f"{RESULTS_FILE} does not exist", file=sys.stderr)
        return {"tests": []}
    
    return json.loads(f.read_text())


def save_results(data: dict) -> None:
    f = pathlib.Path(RESULTS_FILE)
    f.write_text(json.dumps(data, indent=4))

def get_test_id(test: dict, i: int) -> str:
    return f"{test.get('provider', 'unknown')}-{test.get('model', 'unknown')}-{test.get('test_slug', 'unknown')}-{i}"

@app.route("/")
def index():
    """Serves the main review page."""
    return render_template("review_template.html")

@app.route("/results")
def get_results():
    """Returns the current results data."""
    results_data = load_results()
    # Add a unique ID to each test for easier frontend handling
    for i, test in enumerate(results_data.get("tests", [])):
        test['unique_id'] = get_test_id(test, i)
    return jsonify(results_data)

@app.route("/update_rating", methods=["POST"])
def update_rating():
    """Updates the rating for a specific test."""
    data = request.get_json()
    if not data or "unique_id" not in data or "rating" not in data:
        return jsonify({"success": False, "message": "Missing unique_id or rating"}), 400

    unique_id = data["unique_id"]
    try:
        # Ensure rating is an int between 0 and 10, or None
        rating_input = data["rating"]
        if rating_input is None or rating_input == '':
            rating = None
        else:
            rating = int(rating_input)
            if not (0 <= rating <= 10):
                raise ValueError("Rating must be between 0 and 10.")
    except (ValueError, TypeError):
        return jsonify({"success": False, "message": "Invalid rating value"}), 400

    results_data = load_results()
    tests = results_data.get("tests", [])
    updated = False

    # Find the test by the unique ID we added
    for i, test in enumerate(tests):
         current_unique_id = get_test_id(test, i)
         if current_unique_id == unique_id:
             print(f"Updating rating for test {unique_id} to {rating}")
             test["rating_10"] = rating
             updated = True
             break

    if not updated:
        return jsonify({"success": False, "message": "Test not found"}), 404

    try:
        save_results(results_data)
        return jsonify({"success": True})
    except (IOError, TypeError) as e:
        return jsonify({"success": False, "message": f"Failed to save results: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
